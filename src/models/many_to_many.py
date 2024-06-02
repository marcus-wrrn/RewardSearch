import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
#from transformers import DebertaConfig, DebertaModel, DebertaTokenizer
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
        
        # This should be in a list but I am getting strange results after loading and I am testing to see if it's due to pytorch loading the state dict incorrectly
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
    

class MORSpyFull(nn.Module):
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

        input_size = vocab_size * head_num
        self.fc = nn.Sequential(
            nn.Linear(input_size, int(input_size * 1.5)),
            nn.ReLU(),
            nn.Linear(int(input_size * 1.5), vocab_size),
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

        self.freeze_encoder = freeze_encoder

    def process_encoder(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor) -> tuple[Tensor, Tensor]:
        """Maps input embedding space, to response space"""
        encoder_logits, encoder_head_out = self.encoder(pos_embs, neg_embs, neut_embs, assas_emb)
        return encoder_logits, encoder_head_out

    def forward(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor):
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_out_pooled, tri_out = self.process_encoder(pos_embs, neg_embs, neut_embs, assas_emb)
        else:
            encoder_out_pooled, tri_out = self.process_encoder(pos_embs, neg_embs, neut_embs, assas_emb)

        logits = self.reranker.rerank_and_process(encoder_out_pooled, pos_embs, neg_embs, neut_embs, assas_emb)

        num_heads = tri_out.shape[1]
        if num_heads != self.head_num:
            raise ValueError(f"Number of heads must be the same size, expected {self.head_num} got {num_heads}")
        
        #sim_scores = []
        # TODO: Replace with attention mechanism (matmul)
        # for i in range(self.head_num):
        #     head = tri_out[:, i, :]
        #     head = head.unsqueeze(1)

        #     cos_sim = F.cosine_similarity(head, logits.word_embs, dim=2)
        #     sim_scores.append(cos_sim)
        layer_normed = F.normalize(tri_out, p=2, dim=2)
        sim_scores = torch.matmul(layer_normed, logits.word_embs.transpose(1, 2))


        #sim_scores = torch.stack(sim_scores, dim=2)
        sim_scores = sim_scores.view(sim_scores.shape[0], -1)
        out = self.fc(sim_scores)

        logits.reranker_out = out
        return logits


