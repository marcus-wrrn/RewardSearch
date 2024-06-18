import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.vector_search import VectorSearch
import utils.utilities as utils
from models.reranker import Reranker

class MORSpyManyPooled(nn.Module):
    def __init__(self, 
                 num_heads=3):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(3072, 4500),
            nn.Tanh(),
            nn.Linear(4500, 2000),
            nn.Tanh(),
            nn.Linear(2000, 1250),
        )

        self.num_heads = num_heads
        
        self.head_layers = nn.ModuleList([nn.Linear(1250, 768) for _ in range(self.num_heads)])
    
    def _get_combined_input(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor) -> Tensor:
        neg_emb = utils.cluster_embeddings(neg_embs)
        neut_emb = utils.cluster_embeddings(neut_embs)
        pos_emb = utils.cluster_embeddings(pos_embs)

        return torch.cat((neg_emb, assas_emb, neut_emb, pos_emb), dim=1)

    def process_heads(self, 
                      pos_embs: Tensor, 
                      neg_embs: Tensor, 
                      neut_embs: Tensor,
                      assas_emb: Tensor) -> Tensor:
        """Gets the output of all three heads"""
        concatenated = self._get_combined_input(pos_embs, neg_embs, neut_embs, assas_emb)
        intermediary_out = self.fc(concatenated)
        
        model_outs = [F.normalize(layer(intermediary_out), p=2, dim=1) for layer in self.head_layers]
        # Cluster embeddings together
        model_out_stacked = torch.stack(model_outs, dim=1)
        return model_out_stacked

    def forward(self, 
                pos_embs: Tensor, 
                neg_embs: Tensor, 
                neut_embs: Tensor, 
                assas_emb: Tensor):
        model_out_stacked = self.process_heads(pos_embs, neg_embs, neut_embs, assas_emb)

        # Pool heads
        model_out_pooled = torch.mean(model_out_stacked, dim=1)
        model_out_pooled = F.normalize(model_out_pooled, p=2, dim=1)
        
        # # Apply reranking
        # out = self._rerank_and_process(model_out_pooled, pos_embs, neg_embs, neut_embs, assas_emb)
        return model_out_pooled, model_out_stacked
    

class RetrievalTransformerNAR(nn.Module):
    def __init__(self, 
                 vocab: VectorSearch,
                 encoder: MORSpyManyPooled,
                 neg_weight=0.0,
                 neut_weight=1.0,
                 assas_weight=-10.0, 
                 vocab_size=80, 
                 head_num=3, 
                 device='cpu', 
                 freeze_encoder=True):
        super().__init__()

        self.vocab_size = vocab_size
        self.head_num = head_num

        self.value_gen = nn.Sequential(
            nn.Linear(768, 812),
            nn.ReLU(),
            nn.Linear(812, 768),
        )

        self.query_gen = nn.Sequential(
            nn.Linear(768, 812),
            nn.ReLU(),
            nn.Linear(812, 768)
        )

        # Keys are just the word embeddings using key gen leads to similar results with more computation
        # self.key_gen = nn.Sequential(
        #     nn.Linear(768, 800),
        #     nn.ReLU(),
        #     nn.Linear(800, 768)
        # )

        self.fc = nn.Sequential(
            nn.Linear(768, 812),
            nn.ReLU(),
            nn.Linear(812, 768),
        )

        self.encoder = encoder

        self.reranker = Reranker(
            vocab=vocab,
            vocab_size=vocab_size,
            neg_weight=neg_weight,
            neut_weight=neut_weight,
            assas_weight=assas_weight,
            device=device
        )
        self.device = device
        self.freeze_encoder = freeze_encoder

    def process_encoder(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor) -> tuple[Tensor, Tensor]:
        """Maps input embedding space, to response space"""
        encoder_logits, encoder_head_out = self.encoder(pos_embs, neg_embs, neut_embs, assas_emb)
        return encoder_logits, encoder_head_out
    
    def retrieve_embeddings(self, search_heads: Tensor, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor) -> tuple[Tensor, Tensor]:
        logits = []
        for i in range(self.head_num):
            logits.append(self.reranker.rerank_and_process(search_heads[:, i], pos_embs, neg_embs, neut_embs, assas_emb))
        
        word_embs = torch.cat([logs.word_embs for logs in logits], dim=1)
        scores = torch.cat([logs.emb_scores for logs in logits], dim=1).unsqueeze(-1)
        return word_embs, scores
    
    def _retrieve_and_score(self, search_emb: Tensor, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.reranker.rerank_and_process(search_emb, pos_embs, neg_embs, neut_embs, assas_emb)
        word_embs = logits.word_embs
        scores = logits.emb_scores.unsqueeze(-1)
        return word_embs, scores
    
    def _score_based_attn(self, queries: Tensor, word_embs: Tensor, scores: Tensor) -> Tensor:
        # Generate values
        values = self.value_gen(word_embs * scores)
        attn_weights = torch.matmul(queries, word_embs.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = torch.matmul(attn_weights, values)
        # Add weights, (number is relative to number of search heads)
        return attn_weights

    def forward(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor):
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_out_pooled, tri_out = self.process_encoder(pos_embs, neg_embs, neut_embs, assas_emb)
        else:
            encoder_out_pooled, tri_out = self.process_encoder(pos_embs, neg_embs, neut_embs, assas_emb)
        
        num_heads = tri_out.shape[1]
        if num_heads != self.head_num:
            raise ValueError(f"Number of heads must be the same size, expected {self.head_num} got {num_heads}")
        
        word_embs, scores = self._retrieve_and_score(encoder_out_pooled, pos_embs, neg_embs, neut_embs, assas_emb)
        
        # Generate Queries
        tri_out = F.normalize(tri_out, p=2, dim=2)
        queries = torch.stack([self.query_gen(tri_out[:, i]) for i in range(num_heads)], dim=1)

        attn_weights = self._score_based_attn(queries, word_embs, scores)

        # Add and norm with pooled encoder output as residual (leads to better results than using tri_head)
        attn_weights = attn_weights.sum(dim=1) 
        attn_weights = F.normalize(attn_weights + encoder_out_pooled, p=2, dim=1)
        out = self.fc(attn_weights)
        out = F.normalize(out, p=2, dim=1)

        # Find highest scoring results for comparison
        _, highest_scoring_index = torch.topk(scores, k=5, dim=1)
        highest_scoring_index = highest_scoring_index.squeeze(-1)
        batch_indices = torch.arange(out.shape[0], device=self.device).unsqueeze(1).repeat(1, 5)
        highest_scoring_embs = word_embs[batch_indices, highest_scoring_index]
        highest_scoring_embs = utils.cluster_embeddings(highest_scoring_embs)

        texts, search_embs, dist = self.reranker.vocab.search(out, num_results=5)
        search_embs = torch.tensor(search_embs, device=self.device).squeeze(1)
        
        return out, highest_scoring_embs, search_embs


