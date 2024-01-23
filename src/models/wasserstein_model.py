import torch
from torch._C import device
from torch import Tensor, device
from utils.vector_search import VectorSearch
from models.multi_objective_models import MORSpyMaster

"""I don't believe this method is feasible for various reasons, changing direction"""


class MORSpyWasserstein(MORSpyMaster):
    """Currently in development"""

    def __init__(self, vocab: VectorSearch, device: device, neutral_weight=1, negative_weight=0, assas_weights=-10, backbone='all-mpnet-base-v2', vocab_size=80, search_pruning=False):
        super().__init__(vocab, device, neutral_weight, negative_weight, assas_weights, backbone, vocab_size, search_pruning)
    
    def _normalize_scores(self, scores: Tensor):
        
        ...


    def _wasserstein_rot(self, pos_dist: Tensor, pos_weights: Tensor, neg_dist: Tensor, neg_weights: Tensor):
        """Wasserstein Rotation between the normalized scores of both positive and negative distributions"""
        # TODO: Normalize weights to be in line with a normal CDF
        # Normalize weights
        distance = torch.sum(torch.abs(neg_weights - pos_weights) * torch.abs(neg_dist - pos_dist), dim=1)
        return distance
    
    def find_search_embeddings(self, word_embeddings: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, assassin_encs: Tensor):
        word_embs_expanded = word_embeddings.unsqueeze(2)
        # Process encoding shapes
        pos_encs = self._expand_encodings_for_search(pos_encs)
        neg_encs = self._expand_encodings_for_search(neg_encs)
        neut_encs = self._expand_encodings_for_search(neut_encs)
        assas_encs = self._expand_encodings_for_search(assassin_encs.unsqueeze(1))

        pos_reward = self._get_total_reward(word_embs_expanded, pos_encs, neg_encs, neut_encs, assas_encs, reverse=False)
        neg_reward = self._get_total_reward(word_embs_expanded, pos_encs, neg_encs, neut_encs, assas_encs, reverse=True)

        # Normalize positive and negative rewards
        #pos_reward_sum = pos_reward.sum(dim=1, keepdim=True)
        pos_reward_sorted, pos_reward_indices = torch.sort((pos_reward), descending=True, dim=1)

        #neg_reward_sum = neg_reward.sum(dim=1, keepdim=True)
        neg_reward_sorted, neg_reward_indices = torch.sort((neg_reward), descending=True, dim=1)

        embeddings_pos = torch.gather(word_embeddings, 1, pos_reward_indices.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))
        embeddings_neg = torch.gather(word_embeddings, 1, neg_reward_indices.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))

        # Highest scoring word embedding
        highest_scoring_embedding = embeddings_pos[:, 0]
        highest_scoring_embedding_index = pos_reward_indices[:, 0]


        return highest_scoring_embedding, highest_scoring_embedding_index, (embeddings_pos, embeddings_neg), (pos_reward_sorted, neg_reward_sorted)