import torch
import torch.nn.functional as F
from torch import Tensor
from utils.vector_search import VectorSearch
import utils.utilities as utils
import numpy as np

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
        self.reranker_out = None
    
    def add_text(self, texts: str):
        self.texts = texts
    
    def add_dist(self, dists: list):
        self.dists = dists

    def add_encoder_out(self, out: Tensor):
        self.encoder_out = out

class Reranker:
    def __init__(self, 
                 vocab: VectorSearch,
                 vocab_size = 80, 
                 targ_weight=1.0,
                 neg_weight=0.0, 
                 neut_weight=1.0, 
                 assas_weight=-10.0, 
                 device='cpu'):
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.targ_w = targ_weight
        self.neg_w = neg_weight
        self.neut_w = neut_weight
        self.assas_w = assas_weight
        self.device = device

    
    def _expand_encodings_for_search(self, encs: Tensor):
        """Converts shape input encodings from [batch_size, num_encodings, embedding_size] -> [batch_size, vocab_size, num_encodings, embedding_size]"""
        return encs.unsqueeze(1).expand(-1, self.vocab_size, -1, -1)
    
    def _get_primary_reward(self, pos_scores: Tensor, neg_scores: Tensor, neut_scores: Tensor, assas_scores: Tensor, reverse=False):
        combined_scores = torch.cat((pos_scores, neg_scores, neut_scores, assas_scores), dim=2)
        _, indices = combined_scores.sort(dim=2, descending=True)

        # Create reward copies
        pos_reward = torch.zeros(pos_scores.shape[2]).to(self.device)   # Positive must be the only zero value
        neg_reward = torch.ones(neg_scores.shape[2]).to(self.device)
        neut_reward = torch.ones(neut_scores.shape[2]).to(self.device) 
        assas_reward = torch.ones(assas_scores.shape[2]).to(self.device)

        combined_rewards = torch.cat((pos_reward, neg_reward, neut_reward, assas_reward))
        combined_rewards = combined_rewards.expand((combined_scores.shape[0], self.vocab_size, combined_rewards.shape[0]))
        # Map rewards to their corresponding indices
        rewards = torch.gather(combined_rewards, 2, indices)

        # Find the total number of correct guesses, equal to the index of the first non-zero value
        num_correct = torch.argmax(rewards, dim=2)

        # Find the difference between the lowest correct guess and the highest incorrect guess
        difference = combined_scores.gather(2, num_correct.unsqueeze(2)).squeeze(2) - combined_scores.gather(2, (num_correct + 1).unsqueeze(2)).squeeze(2)
        difference = np.e**torch.abs(difference)


        batch_size, seq_length, word_shape = combined_scores.shape
        mask = torch.arange(word_shape, device=self.device).expand(batch_size, seq_length, word_shape) < num_correct.unsqueeze(2)
        
        # Apply the mask to combined_scores
        masked_combined_scores = combined_scores * mask

        # Calculate the sum of the scores with the mask applied
        sum_scores = masked_combined_scores.sum(dim=2)

        # Calculate the number of correct guesses, ensuring no division by zero
        num_correct_float = num_correct.float()
        num_correct_nonzero = torch.where(num_correct_float == 0, torch.ones_like(num_correct_float), num_correct_float)

        # Calculate the mean of the scores, avoiding NaN and Inf values
        mean_scores = (sum_scores / num_correct_nonzero) * 10

        if reverse:
            # Find the inverse of the positive reward (num incorrect)
            return (pos_reward.shape[0] - num_correct) - difference - mean_scores
        
        return num_correct + difference + mean_scores

    def _get_reward_tensor(self, size: int, weight: float, reverse: bool) -> Tensor:
        reward = torch.ones(size).to(self.device) * weight
        if reverse:
            return reward * -1
        return reward

    def _get_secondary_reward(self, neg_scores: Tensor, neut_scores: Tensor, assas_scores: Tensor, reverse=False):
        """Finds the highest value between both the negative, neutral and assassin scores"""
        combined = torch.cat((neg_scores, neut_scores, assas_scores), dim=2)
        _, indices = combined.sort(dim=2, descending=True)

        neg_reward = self._get_reward_tensor(neg_scores.shape[2], self.neg_w, reverse)
        neut_reward = self._get_reward_tensor(neut_scores.shape[2], self.neut_w, reverse)
        assas_reward = self._get_reward_tensor(assas_scores.shape[2], self.assas_w, reverse)

        combined_rewards = torch.cat((neg_reward, neut_reward, assas_reward))
        combined_rewards = combined_rewards.expand((combined.shape[0], self.vocab_size, combined_rewards.shape[0]))
        rewards = torch.gather(combined_rewards, 2, indices)

        # Find the values of the first index, equal to the given reward (score of the most similar unwanted embedding)
        # Due to the sequential nature of codenames only the first non-target guess matters
        reward = rewards[:, :, 0]
        return reward
    
    
    def _get_total_reward(self, word_encs: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, assas_encs: Tensor, reverse: bool) -> Tensor:
        # Find scores
        pos_scores = F.cosine_similarity(word_encs, pos_encs, dim=3)
        neg_scores = F.cosine_similarity(word_encs, neg_encs, dim=3)
        neut_scores = F.cosine_similarity(word_encs, neut_encs, dim=3)
        assas_scores = F.cosine_similarity(word_encs, assas_encs, dim=3)
        # Get reward
        primary_reward = self._get_primary_reward(pos_scores, neg_scores, neut_scores, assas_scores, reverse=reverse)
        secondary_reward = self._get_secondary_reward(neg_scores, neut_scores, assas_scores, reverse=reverse)

        return primary_reward + secondary_reward

    def _find_scored_embeddings(self, reward: Tensor, word_embeddings: Tensor):
        # Find lowest scoring and highest scored indices
        index_max_vals, index_max = torch.topk(reward, k=word_embeddings.shape[1]//2, dim=1)
        index_min_vals, index_min = torch.topk(reward, k=word_embeddings.shape[1]//2, largest=False, dim=1)

        embeddings_max = torch.gather(word_embeddings, 1, index_max.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))
        embeddings_min = torch.gather(word_embeddings, 1, index_min.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))

        clustered_embs_max = utils.cluster_embeddings(embeddings_max)
        clustered_embs_min = utils.cluster_embeddings(embeddings_min)

        # Highest scoring word embedding
        highest_scoring_embedding = embeddings_max[:, 0]
        highest_scoring_embedding_index = index_max[:, 0]

        return highest_scoring_embedding, highest_scoring_embedding_index, clustered_embs_max, clustered_embs_min


    def _find_search_embeddings(self, 
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
    
    def rerank_and_process(self, model_out: Tensor, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_embs: Tensor) -> ManyOutObj:
        texts, embs, dist = self.vocab.search(model_out, num_results=self.vocab_size)
        # embs = torch.tensor(embs).squeeze(1)
        # perm = torch.randperm(embs.shape[1])

        # texts = np.take_along_axis(texts, perm.unsqueeze(0).numpy(), axis=1)
        # embs = embs.index_select(1, perm).to(self.device)
        embs = torch.tensor(embs, device=self.device).squeeze(1)

        out_obj = self._find_search_embeddings(embs, pos_embs, neg_embs, neut_embs, assas_embs)

        out_obj.add_text(texts)
        out_obj.add_dist(dist)
        out_obj.add_encoder_out(model_out)
        return out_obj