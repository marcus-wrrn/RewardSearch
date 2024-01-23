import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
#from transformers import DebertaConfig, DebertaModel, DebertaTokenizer
from utils.vector_search import VectorSearch
from models.base_models import SentenceEncoder
from scipy.stats import wasserstein_distance

# Primary Codenames models, used in current tests

class MOROutObj:
    def __init__(self, words, model_out, search_out, search_out_max, search_out_min):
        self.words = words
        self.model_out = model_out
        self.search_out = search_out
        self.search_out_max = search_out_max
        self.search_out_min = search_out_min

class MORSpyMaster(nn.Module):
    """
    Multi-Objective Retrieval model for codenames with 4 competing objectives
    """
    def __init__(self, vocab: VectorSearch, device: torch.device, neutral_weight=1.0, negative_weight=0.0, assas_weights=-10.0, backbone='all-mpnet-base-v2', vocab_size=80, search_pruning=False):
        super().__init__()
        self.encoder = SentenceEncoder(backbone)
        self.vocab_size = vocab_size

        self.neut_weight = neutral_weight
        self.neg_weight = negative_weight
        self.assas_weights = assas_weights
        
        self.fc = nn.Sequential(
            nn.Linear(3072, 2304),
            nn.ReLU(),
            nn.Linear(2304, 1700),
            nn.ReLU(),
            nn.Linear(1700, 1000),
            nn.ReLU(),
            nn.Linear(1000, 768),
        )
        self.vocab = vocab
        self.device = device

        self.search_pruning = search_pruning

    def _process_embeddings(self, embs: Tensor):
        """Mean pool and normalize all embeddings"""
        out = torch.mean(embs,dim=1)
        out = F.normalize(out, p=2, dim=1)
        return out
    
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

        if reverse:
            # Find the inverse of the positive reward (num incorrect)
            return (pos_reward.shape[0] - num_correct)
        
        return num_correct 

    def _get_reward_tensor(self, size: int, weight: float, reverse: bool) -> Tensor:
        reward = torch.ones(size).to(self.device) * weight
        if reverse:
            return reward * -1
        return reward

    def _get_secondary_reward(self, neg_scores: Tensor, neut_scores: Tensor, assas_scores: Tensor, reverse=False):
        """Finds the highest value between both the negative, neutral and assassin scores"""
        combined = torch.cat((neg_scores, neut_scores, assas_scores), dim=2)
        _, indices = combined.sort(dim=2, descending=True)

        neg_reward = self._get_reward_tensor(neg_scores.shape[2], self.neg_weight, reverse)
        neut_reward = self._get_reward_tensor(neut_scores.shape[2], self.neut_weight, reverse)
        assas_reward = self._get_reward_tensor(assas_scores.shape[2], self.assas_weights, reverse)

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
    
    def _find_scored_embeddings(self, reward, word_embeddings):
        # Find lowest scoring and highest scored indices
        index_max_vals, index_max = torch.topk(reward, k=word_embeddings.shape[1]//2, dim=1)
        index_min_vals, index_min = torch.topk(reward, k=word_embeddings.shape[1]//2, largest=False, dim=1)

        embeddings_max = torch.gather(word_embeddings, 1, index_max.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))
        embeddings_min = torch.gather(word_embeddings, 1, index_min.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))

        max_embeddings_pooled = self._process_embeddings(embeddings_max)
        min_embeddings_pooled = self._process_embeddings(embeddings_min)

        # Highest scoring word embedding
        highest_scoring_embedding = embeddings_max[:, 0]
        highest_scoring_embedding_index = index_max[:, 0]

        return highest_scoring_embedding, highest_scoring_embedding_index, max_embeddings_pooled, min_embeddings_pooled

    def find_search_embeddings(self, word_embeddings: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, assassin_encs: Tensor):
        word_embs_expanded = word_embeddings.unsqueeze(2)
        # Process encoding shapes
        pos_encs = self._expand_encodings_for_search(pos_encs)
        neg_encs = self._expand_encodings_for_search(neg_encs)
        neut_encs = self._expand_encodings_for_search(neut_encs)
        assas_encs = self._expand_encodings_for_search(assassin_encs.unsqueeze(1))

        tot_reward = self._get_total_reward(word_embs_expanded, pos_encs, neg_encs, neut_encs, assas_encs, reverse=False)

        return self._find_scored_embeddings(tot_reward, word_embeddings)
    

    def _prune_word_embeddings(self, word_embeddings: Tensor, model_out: Tensor, sim_cutoff=0.08):
        model_out_expanded = model_out.unsqueeze(1)
        # Calculate similarity
        sim_scores = F.cosine_similarity(word_embeddings, model_out_expanded, dim=2)
        # Find std deviation of batch score
        std_dev = sim_scores.std(dim=1).mean()
        # Create pruning algorithm

        print() # NOP for breakpoint

    def _get_combined_input(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor) -> Tensor:
        neg_emb = self._process_embeddings(neg_embs)
        neut_emb = self._process_embeddings(neut_embs)
        pos_emb = self._process_embeddings(pos_embs)

        return torch.cat((neg_emb, assas_emb, neut_emb, pos_emb), dim=1)
    
    def _convert_word_embeddings_to_tensor(self, embs: Tensor):
        return torch.tensor(embs).to(self.device).squeeze(1)

    def forward(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor) -> MOROutObj | tuple:
        concatenated = self._get_combined_input(pos_embs, neg_embs, neut_embs, assas_emb)
        model_out = self.fc(concatenated)

        model_out = F.normalize(model_out, p=2, dim=1)

        # ANN Search
        words, word_embeddings, dist = self.vocab.search(model_out, num_results=self.vocab_size)
        word_embeddings = self._convert_word_embeddings_to_tensor(word_embeddings)

        if self.search_pruning:
            self._prune_word_embeddings(word_embeddings, model_out)

        search_out, search_out_index, search_out_max, search_out_min = self.find_search_embeddings(word_embeddings, pos_embs, neg_embs, neut_embs, assas_emb)

        if self.training:
            return model_out, search_out, search_out_max, search_out_min
        
        return MOROutObj(words[search_out_index.cpu()][:, :1], model_out, search_out, search_out_max, search_out_min)


class MORSpyDualHead(MORSpyMaster):
    def __init__(self, vocab: VectorSearch, device: device, neutral_weight=1, negative_weight=0, assassin_weights=-10, backbone='all-mpnet-base-v2', vocab_size=80, search_pruning=False):
        super().__init__(vocab, device, neutral_weight, negative_weight, assassin_weights, backbone, vocab_size, search_pruning)

        self.fc = nn.Sequential(
            nn.Linear(3072, 2304),
            nn.ReLU(),
            nn.Linear(2304, 1700),
            nn.ReLU(),
            nn.Linear(1700, 1000),
            nn.ReLU(),
        )

        self.pos_layer = nn.Linear(1000, 768)
        self.neg_layer = nn.Linear(1000, 768)

    def _find_scored_embeddings(self, reward: Tensor, word_embeddings: Tensor):
        # Find lowest scoring and highest scored indices
        index_max_vals, index_max = torch.topk(reward, k=word_embeddings.shape[1]//2, dim=1)

        embeddings_max = torch.gather(word_embeddings, 1, index_max.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))

        max_embeddings_pooled = self._process_embeddings(embeddings_max)

        # Highest scoring word embedding
        highest_scoring_embedding = embeddings_max[:, 0]
        highest_scoring_embedding_index = index_max[:, 0]

        return highest_scoring_embedding, highest_scoring_embedding_index, max_embeddings_pooled

    def find_search_embeddings(self, word_embeddings: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, assassin_encs: Tensor, reverse=False):
        word_embs_expanded = word_embeddings.unsqueeze(2)
        # Process encoding shapes
        pos_encs = self._expand_encodings_for_search(pos_encs)
        neg_encs = self._expand_encodings_for_search(neg_encs)
        neut_encs = self._expand_encodings_for_search(neut_encs)
        assas_encs = self._expand_encodings_for_search(assassin_encs.unsqueeze(1))

        tot_reward = self._get_total_reward(word_embs_expanded, pos_encs, neg_encs, neut_encs, assas_encs, reverse=reverse)


        return self._find_scored_embeddings(tot_reward, word_embeddings)

    def forward(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor, assas_emb: Tensor) -> MOROutObj | tuple:
        concatenated = self._get_combined_input(pos_embs, neg_embs, neut_embs, assas_emb)
        intermediary_out = self.fc(concatenated)

        model_out_pos = self.pos_layer(intermediary_out)
        model_out_neg = self.neg_layer(intermediary_out)

        model_out_pos = F.normalize(model_out_pos, p=2, dim=1)
        model_out_neg = F.normalize(model_out_neg, p=2, dim=1)

        # ANN Search
        pos_words, pos_word_embeddings, _ = self.vocab.search(model_out_pos, num_results=self.vocab_size)
        pos_word_embeddings = self._convert_word_embeddings_to_tensor(pos_word_embeddings)

        neg_words, neg_word_embeddings, _ = self.vocab.search(model_out_neg, num_results=self.vocab_size)
        neg_word_embeddings = self._convert_word_embeddings_to_tensor(neg_word_embeddings)

        # Scoring
        pos_search_out, pos_search_out_index, pos_embedding_pooled = self.find_search_embeddings(pos_word_embeddings, pos_embs, neg_embs, neut_embs, assas_emb, reverse=False)
        neg_search_out, neg_search_out_index, neg_embedding_pooled =  self.find_search_embeddings(neg_word_embeddings, pos_embs, neg_embs, neut_embs, assas_emb, reverse=True)

        return (model_out_pos, model_out_neg), (pos_embedding_pooled, neg_embedding_pooled), pos_search_out

class MORSpyMasterSmall(nn.Module):
    """
    Multi-Objective Retrieval model for codenames -> uses only positive, negative and neutral objectives (no assassin capability)
    """
    def __init__(self, vocab: VectorSearch, device: torch.device, neutral_weight=1.0, negative_weight=0.0, backbone='all-mpnet-base-v2', vocab_size=80):
        super().__init__()
        self.encoder = SentenceEncoder(backbone)
        self.vocab_size = vocab_size

        self.neut_weight = neutral_weight
        self.neg_weight = negative_weight
        
        self.fc = nn.Sequential(
            nn.Linear(2304, 1800),
            nn.ReLU(),
            nn.Linear(1800, 1250),
            nn.ReLU(),
            nn.Linear(1250, 900),
            nn.ReLU(),
            nn.Linear(900, 768),
        )
        self.vocab = vocab
        self.device = device

    def _process_embeddings(self, embs: Tensor):
        """Mean pool and normalize all embeddings"""
        out = torch.mean(embs,dim=1)
        out = F.normalize(out, p=2, dim=1)
        return out
    
    def _expand_encodings_for_search(self, encs: Tensor):
        """Converts shape input encodings from [batch_size, num_encodings, embedding_size] -> [batch_size, vocab_size, num_encodings, embedding_size]"""
        return encs.unsqueeze(1).expand(-1, self.vocab_size, -1, -1)
    
    def _get_positive_reward(self, pos_scores: Tensor, neg_scores: Tensor, neut_scores: Tensor):
        combined_scores = torch.cat((pos_scores, neg_scores, neut_scores), dim=2)
        _, indices = combined_scores.sort(dim=2, descending=True)

        # Create reward copies
        pos_reward = torch.zeros(pos_scores.shape[2]).to(self.device)
        neg_reward = torch.ones(neg_scores.shape[2]).to(self.device)
        neut_reward = torch.ones(neut_scores.shape[2]).to(self.device) 

        combined_rewards = torch.cat((pos_reward, neg_reward, neut_reward))
        combined_rewards = combined_rewards.expand((combined_scores.shape[0], self.vocab_size, combined_rewards.shape[0]))
        # Map rewards to their corresponding indices
        rewards = torch.gather(combined_rewards, 2, indices)

        # Mask all target embeddings to 0 and unwanted embeddings to 1
        non_zero_mask = torch.where(rewards != 0, 1., 0.)
        # Find the total number of correct guesses, equal to the index of the first non-zero value in the mask
        num_correct = torch.argmax(non_zero_mask, dim=2)

        return num_correct
    
    def _get_secondary_reward(self, neg_scores: Tensor, neut_scores: Tensor):
        """Finds the highest value between both the negative and neutral scores"""
        combined = torch.cat((neg_scores, neut_scores), dim=2)
        _, indices = combined.sort(dim=2, descending=True)

        neg_reward = torch.ones(neg_scores.shape[2]).to(self.device) * self.neg_weight
        neut_reward = torch.ones(neut_scores.shape[2]).to(self.device) * self.neut_weight

        combined_rewards = torch.cat((neg_reward, neut_reward))
        combined_rewards = combined_rewards.expand((combined.shape[0], self.vocab_size, combined_rewards.shape[0]))
        rewards = torch.gather(combined_rewards, 2, indices)
        
        # Find the values of the first index, equal to the given reward (score of the most similar unwanted embedding)
        # Due to the sequential nature of codenames only the first non-target guess matters
        reward = rewards[:, :, 0]
        return reward


    def find_search_embeddings(self, word_embeddings: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor):
        word_embs_expanded = word_embeddings.unsqueeze(2)
        # Process encoding shapes
        pos_encs = self._expand_encodings_for_search(pos_encs)
        neg_encs = self._expand_encodings_for_search(neg_encs)
        neut_encs = self._expand_encodings_for_search(neut_encs)

        # Find Similarity of all found embeddings 
        pos_scores = F.cosine_similarity(word_embs_expanded, pos_encs, dim=3)
        neg_scores = F.cosine_similarity(word_embs_expanded, neg_encs, dim=3)
        neut_scores = F.cosine_similarity(word_embs_expanded, neut_encs, dim=3)

        # Finds each word embeddings expected game reward
        primary_reward = self._get_positive_reward(pos_scores, neg_scores, neut_scores)
        secondary_reward = self._get_secondary_reward(neg_scores, neut_scores)

        tot_reward = primary_reward + secondary_reward

        # Find lowest scoring and highest scored indices
        index_max_vals, index_max = torch.topk(tot_reward, k=word_embeddings.shape[1]//2, dim=1)
        index_min_vals, index_min = torch.topk(tot_reward, k=word_embeddings.shape[1]//2, largest=False, dim=1)

        embeddings_max = torch.gather(word_embeddings, 1, index_max.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))
        embeddings_min = torch.gather(word_embeddings, 1, index_min.unsqueeze(-1).expand(-1, -1, word_embeddings.shape[2]))

        embeddings_max_pooled = self._process_embeddings(embeddings_max)
        embeddings_min_pooled = self._process_embeddings(embeddings_min)

        # Highest scoring word embedding
        highest_scoring_embedding = embeddings_max[:, 0]
        highest_scoring_embedding_index = index_max[:, 0]

        return highest_scoring_embedding, highest_scoring_embedding_index, embeddings_max_pooled, embeddings_min_pooled
    
    def forward(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor):
        neg_emb = self._process_embeddings(neg_embs)
        neut_emb = self._process_embeddings(neut_embs)
        pos_emb = self._process_embeddings(pos_embs)

        concatenated = torch.cat((neg_emb, neut_emb, pos_emb), dim=1)
        model_out = self.fc(concatenated)
        model_out = F.normalize(model_out, p=2, dim=1)

        words, word_embeddings, dist = self.vocab.search(model_out, num_results=self.vocab_size)
        word_embeddings = torch.tensor(word_embeddings).to(self.device).squeeze(1)
        
        search_out, search_out_index, search_out_max, search_out_min = self.find_search_embeddings(word_embeddings, pos_embs, neg_embs, neut_embs)

        if self.training:
            return model_out, search_out, search_out_max, search_out_min
        
        return words[search_out_index], search_out, dist

