import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
#from transformers import DebertaConfig, DebertaModel, DebertaTokenizer
from utils.vector_search import VectorSearch
from models.multi_objective_models import MORSpyMaster, MOROutObj

class ManyOutObj:
    """Output object of the ManytoNum models, contains all info needed for training and testing"""
    def __init__(self, embeddings: Tensor, 
                 emb_scores: Tensor, 
                 highest_scoring_embs: Tensor, 
                 emb_idx: Tensor, 
                 max_embs_pooled: Tensor, 
                 min_embs_pooled: Tensor) -> None:
        self.h_score_emb = highest_scoring_embs
        self.emb_ids = emb_idx
        self.word_embs = embeddings
        self.emb_scores = emb_scores
        self.max_embs_pooled = max_embs_pooled
        self.min_embs_pooled = min_embs_pooled
        self.texts = None
        self.dists = None
        self.encoder_out = None
    
    def add_text(self, texts: str):
        self.texts = texts
    
    def add_dist(self, dists: list):
        self.dists = dists

    def add_encoder_out(self, out: Tensor):
        self.encoder_out = out


class MORSpyManyPooled(MORSpyMaster):
    def __init__(self, vocab: VectorSearch, device: device, neutral_weight=1, negative_weight=0, assassin_weights=-10, vocab_size=80, search_pruning=False, num_heads=3):
        super().__init__(vocab, device, neutral_weight, negative_weight, assassin_weights, vocab_size, search_pruning)

        self.num_heads = num_heads
        self.head_layers = [nn.Linear(1000, 768, device=device) for i in range(num_heads)]

    
    def _find_scored_embeddings(self, reward: Tensor, word_embeddings: Tensor):
        # Find lowest scoring and highest scored indices
        index_max_vals, index_max = torch.topk(reward, k=word_embeddings.shape[1]//2, dim=1)
        index_min_vals, index_min = torch.topk(reward, k=word_embeddings.shape[1]//2, largest=False, dim=1)

        embeddings_max = torch.gather(word_embeddings, 1, index_max.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))
        embeddings_min = torch.gather(word_embeddings, 1, index_min.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))

        clustered_embs_max = self._cluster_embeddings(embeddings_max)
        clustered_embs_min = self._cluster_embeddings(embeddings_min)

        # Highest scoring word embedding
        highest_scoring_embedding = embeddings_max[:, 0]
        highest_scoring_embedding_index = index_max[:, 0]

        return highest_scoring_embedding, highest_scoring_embedding_index, clustered_embs_max, clustered_embs_min

    def find_search_embeddings(self, 
                               word_embeddings: Tensor, 
                               pos_encs: Tensor, 
                               neg_encs: Tensor, 
                               neut_encs: Tensor, 
                               assassin_encs: Tensor, 
                               reverse=False) -> ManyOutObj:
        
        word_embs_expanded = word_embeddings.unsqueeze(2)
        # Process encoding shapes
        pos_encs = self._expand_encodings_for_search(pos_encs)
        neg_encs = self._expand_encodings_for_search(neg_encs)
        neut_encs = self._expand_encodings_for_search(neut_encs)
        assas_encs = self._expand_encodings_for_search(assassin_encs.unsqueeze(1))

        tot_reward = self._get_total_reward(word_embs_expanded, pos_encs, neg_encs, neut_encs, assas_encs, reverse=reverse)

        highest_emb_scored, best_score_idx, max_embs_pooled, min_embs_pooled = self._find_scored_embeddings(tot_reward, word_embeddings)

        out_obj = ManyOutObj(word_embeddings, tot_reward, highest_emb_scored, best_score_idx, max_embs_pooled, min_embs_pooled)
        return out_obj

    
    def _rerank_and_process(self, model_out: Tensor, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_embs: Tensor) -> ManyOutObj:
        texts, embs, dist = self.vocab.search(model_out, num_results=self.vocab_size)
        embs = torch.tensor(embs).to(self.device).squeeze(1)
        out_obj = self.find_search_embeddings(embs, pos_embs, neg_embs, neut_embs, assas_embs)

        out_obj.add_text(texts)
        out_obj.add_dist(dist)
        out_obj.add_encoder_out(model_out)
        return out_obj
    
    def process_heads(self, 
                      pos_embs: Tensor, 
                      neg_embs: Tensor, 
                      neut_embs: Tensor,
                      assas_emb: Tensor) -> Tensor:
        """Gets the output of all three heads"""
        concatenated = self._get_combined_input(pos_embs, neg_embs, neut_embs, assas_emb)
        intermediary_out = self.fc(concatenated)
        
        model_outs = [layer(intermediary_out) for layer in self.head_layers]
        
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
        
        # Apply reranking
        out = self._rerank_and_process(model_out_pooled, pos_embs, neg_embs, neut_embs, assas_emb)
        return out, model_out_stacked
    