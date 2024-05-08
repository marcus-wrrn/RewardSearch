from datasets.dataset import CodeGiverDataset
from sentence_transformers import util
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from loss_fns.loss import CombinedTripletLoss, RewardSearchLoss
import numpy as np
from models.many_to_many import ManyOutObj


"""
The following code is a collection of experimental loss functions that were originally intended to be used in the project. 
However, they did not provide desireable results and so have been moved to their own seperate file, in case they could be of use later.
"""

class CATLoss(CombinedTripletLoss):
    """Combined Asymmetric Triplet Loss"""
    def __init__(self, device, margin=1, weighting=1, neg_weighting=-2):
        super().__init__(margin)
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.device = device
        self.weighting = weighting
        self.neg_weighting = neg_weighting

    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        anchor_expanded = anchor.unsqueeze(1)

        pos_score = F.cosine_similarity(anchor_expanded, pos_encs, dim=2)
        neg_score = F.cosine_similarity(anchor_expanded, neg_encs, dim=2)
        scores = torch.cat((pos_score, neg_score), dim=1)

        scores, indices = scores.sort(dim=1, descending=True)
        # Set all positive values to -1 and negative values to 1
        # use a larger negative 
        modified_indices = torch.where(indices < 3, self.neg_weighting, self.weighting)
        scores = torch.mul(scores, modified_indices)
    
        weights = 1/torch.arange(1, scores.shape[1] + 1).to(self.device) + 1
        scores = torch.mul(scores, weights)
        scores = scores.mean(dim=1)
        loss = F.relu(scores + self.margin)
        return loss.mean()
    
class CATLossNormalDistribution(CATLoss):
    """Uses a normal distribution for the weighting function"""
    def __init__(self, stddev: float, mean=0.0, device="cpu", margin=1, weighting=1, neg_weighting=-1, constant=7, list_size=6):
        super().__init__(device, margin, weighting, neg_weighting)
        self.mean = mean
        self.std = stddev
        self.constant = constant
        
        self.negative_weighting = neg_weighting
        self._norm_distribution = Normal(self.mean, self.std)
        
    def norm_sample(self, indicies):
        norm_dist = torch.exp(self._norm_distribution.log_prob(indicies))
        mean = norm_dist.mean()
        return (norm_dist - mean) / self.std

    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        anchor_expanded = anchor.unsqueeze(1)

        pos_score = F.cosine_similarity(anchor_expanded, pos_encs, dim=2)
        neg_score = F.cosine_similarity(anchor_expanded, neg_encs, dim=2)
        scores = torch.cat((pos_score, neg_score), dim=1)

        sorted_scores, indices = scores.sort(dim=1, descending=True)
        
        # Create loss mask
        size = scores.shape[1]
        mask = torch.ones(size).to(self.device)
        mask[size//2:] = -1
        # mask = mask.unsqueeze(1)
        # Apply mask
        scores = torch.mul(scores, mask)
        # Find normal distribution 
        norm_distribution = self.norm_sample(indices)
        # Apply weights 
        scores = torch.mul(scores, norm_distribution).sum(1)
        loss = F.relu(scores + self.margin)
        return loss.mean()

class CATCluster(CATLoss):
    def __init__(self, dataset: CodeGiverDataset, device: torch.device, margin=1, weighting=1, neg_weighting=-2):
        super().__init__(device, margin, weighting, neg_weighting)
        # Use the codename vocab (word inputs) not the guess word vocab (word outputs)
        self.vocab = dataset.get_vocab(guess_data=False)
        

class CombinedTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CombinedTripletLoss, self).__init__()
        self.margin = margin

    @property
    def name(self):
        return "ConmbinedTripletLoss"

    def _calc_cos_scores(self, anchor, pos_encs, neg_encs):
        """Finds cosine similarity between all word embeddings and the model output"""
        pos_score = F.cosine_similarity(anchor, pos_encs, dim=2)
        neg_score = F.cosine_similarity(anchor, neg_encs, dim=2)
        return pos_score, neg_score

    def forward(self, anchor, pos_encs, neg_encs):
        # Add extra dimension to anchor to align with the pos and neg encodings shape
        anchor_expanded = anchor.unsqueeze(1)   # [batch, emb_size] -> [batch, 1, emb_size]

        pos_score, neg_score = self._calc_cos_scores(anchor_expanded, pos_encs, neg_encs)
        # Combine scores
        pos_score = torch.mean(pos_score, dim=1)
        neg_score = torch.mean(neg_score, dim=1) * 3

        loss = neg_score - pos_score + self.margin
        return F.relu(loss).mean()

class TripletMeanLossL2Distance(CombinedTripletLoss):
    def __init__(self, margin=1.0):
        super(TripletMeanLossL2Distance, self).__init__(margin)

    @property
    def name(self):
        return "TripletMeanLossL2Distance"
    
    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        # Add extra dimension to anchor to align with the pos and neg encodings shape
        anchor_expanded = anchor.unsqueeze(1)  # [batch, emb_size] -> [batch, 1, emb_size]

        # Calculate L2 distance for positive and negative pairs
        pos_distance = torch.norm(anchor_expanded - pos_encs, p=2, dim=2)
        neg_distance = torch.norm(anchor_expanded - neg_encs, p=2, dim=2)

        # Calculate mean of distances
        avg_pos_distance = torch.mean(pos_distance, dim=1)
        avg_neg_distance = torch.mean(neg_distance, dim=1) # Attempting to weight the negative more highly,

        # Calculate triplet loss
        loss = F.relu(avg_pos_distance - avg_neg_distance + self.margin)

        return loss.mean()

class ScoringLoss(CombinedTripletLoss):
    def __init__(self, margin=1, device='cpu', normalize=True):
        super().__init__(margin)
        self.normalize = normalize
        self.device = device
    
    @property
    def name(self):
        return "ScoringLossWithModelSearch"

    def _process_shape(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor):
        pos_dim = pos_tensor.shape[1]
        neg_dim = neg_tensor.shape[1]

        if pos_dim > neg_dim:
            dif = pos_dim - neg_dim
            pos_tensor = pos_tensor[:, dif:]
        elif neg_dim > pos_dim:
            dif = neg_dim - pos_dim
            neg_tensor = neg_tensor[:, dif:]
        
        return pos_tensor, neg_tensor
    
    def _calc_score(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        """
        Compares the values of negative triplet scores to positive triplet scores.
        Returns the total number of negative scores greater than positivefor each batch
        """
        pos_sorted, _ = pos_scores.sort(descending=True, dim=1)
        neg_sorted, _ = neg_scores.sort(descending=False, dim=1)
        
        pos_sorted, neg_sorted = self._process_shape(pos_sorted, neg_sorted)
        
        comparison = torch.where(neg_sorted > pos_sorted, 1.0, 0.0).to(self.device)
        
        final_score = comparison.sum(1, keepdim=True)

        if self.normalize:
            final_score = final_score * 1/comparison.shape[1]

        return final_score

    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        anchor_expanded = anchor.unsqueeze(1)
        pos_scores, neg_scores = self._calc_cos_scores(anchor_expanded, pos_encs, neg_encs)

        total_score = self._calc_score(pos_scores, neg_scores)
        loss = F.relu(neg_scores - pos_scores + total_score + self.margin)
        return loss.mean(), total_score.mean(dim=0)
        

class ScoringLossWithModelSearch(ScoringLoss):
    def __init__(self, margin=1, device='cpu', normalize=True):
        super().__init__(margin, device, normalize)

    @property
    def name(self):
        return "ScoringLossWithModelSearch"
    
    def forward(self, model_out: torch.Tensor, search_out: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        model_expanded = model_out.unsqueeze(1)
        search_expanded = search_out.unsqueeze(1)
        # calculate scores compared to the search output, not the model output
        s_pos_scores, s_neg_scores = self._calc_cos_scores(search_expanded, pos_encs, neg_encs)
        m_pos_scores, m_neg_scores = self._calc_cos_scores(model_expanded, pos_encs, neg_encs)

        total_score = self._calc_score(s_pos_scores, s_neg_scores)

        loss = F.relu(m_neg_scores.mean(dim=1) - m_pos_scores.mean(dim=1) + total_score + self.margin)
        # loss_select = F.relu(s_neg_scores - s_pos_scores + total_score + self.margin)
        # loss = loss + loss_select
        return loss.mean(), total_score.mean(dim=0)

class MultiObjectiveScoringLoss(nn.Module):
    def __init__(self, margin=0.2, device='cpu', normalize=True):
        super().__init__()
        self.margin = margin
        self.device = device
        self.normalize = normalize
    
    @property
    def name(self):
        return "MultiObjectiveScoringLoss"
    
    def _calc_cos_sim(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, neutral: torch.Tensor):
        pos_score = F.cosine_similarity(anchor, pos, dim=2)
        neg_score = F.cosine_similarity(anchor, neg, dim=2)
        neutral_score = F.cosine_similarity(anchor, neutral, dim=2)
        return pos_score, neg_score, neutral_score
    
    def _find_results(self, pos_scores, neg_scores, neutral_scores):
        combined_scores = torch.cat((pos_scores, neg_scores, neutral_scores), dim=1)
        # Find sorted indices
        _, indices = combined_scores.sort(dim=1, descending=True)
        # create reward copies
        pos_reward = torch.zeros(pos_scores.shape[1]).to(self.device)
        neg_reward = torch.ones(neg_scores.shape[1]).to(self.device) * 2
        neut_reward = torch.ones(neutral_scores.shape[1]).to(self.device) 

        combined_rewards = torch.cat((pos_reward, neg_reward, neut_reward))
        combined_rewards = combined_rewards.expand((combined_scores.shape[0], combined_rewards.shape[0]))
        rewards = torch.gather(combined_rewards, 1, indices)

        non_zero_mask = torch.where(rewards != 0, 1., 0.)
        first_non_pos_index = torch.argmax(non_zero_mask, dim=1)
        first_vals = rewards[torch.arange(rewards.size(0)), first_non_pos_index]
        return first_non_pos_index, first_vals
    
    def _calc_reward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, neutral_scores: torch.Tensor):
        num_positive, word_class = self._find_results(pos_scores, neg_scores, neutral_scores)
        target_loss = 1- num_positive/pos_scores.shape[1]

        return target_loss, word_class - 1

    
    def forward(self, model_out: torch.Tensor, search_out: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor, neutral_encs: torch.Tensor):
        model_out_expanded = model_out.unsqueeze(1)
        search_out_expanded = search_out.unsqueeze(1)

        m_pos_score, m_neg_score, m_neut_score = self._calc_cos_sim(model_out_expanded, pos_encs, neg_encs, neutral_encs)
        s_pos_score, s_neg_score, s_neut_score = self._calc_cos_sim(search_out_expanded, pos_encs, neg_encs, neutral_encs)

        target_loss, results = self._calc_reward(s_pos_score, s_neg_score, s_neut_score)
        neg_score = torch.cat((m_neg_score, m_neut_score), dim=1).mean(dim=1)
        loss = F.relu((neg_score - m_pos_score.mean(dim=1)) * target_loss + self.margin)
        return loss.mean(), target_loss.mean(), results.mean()
    


class MultiKeypointLoss(RewardSearchLoss):
    def __init__(self, model_marg=0.2, search_marg=0.7, device='cpu', normalize=True, exp_score=False, num_heads=3):
        super().__init__(model_marg, search_marg, device, normalize, exp_score)
        self.num_heads = num_heads
    
    def _cluster_embeddings(self, embs: Tensor, dim=1):
        embs = embs.mean(dim=dim)
        embs = F.normalize(embs, p=2, dim=dim)
        return embs
    
    def forward(self, model_logits: list[ManyOutObj], pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, assas_encs: Tensor, backward=True):
        # Pool positive embeddings
        pos_cluster = self._cluster_embeddings(pos_encs)
        assas_encs = assas_encs.unsqueeze(1)
        neg_vals = torch.cat((neg_encs, neut_encs, assas_encs), dim=1).to(self.device)
        neg_cluster = self._cluster_embeddings(neg_vals)

        losses = []
        for i, model_log in enumerate(model_logits):
            stable_loss = F.triplet_margin_loss(model_log.encoder_out, pos_cluster, neg_cluster, margin=self.margin)
            search_loss = F.triplet_margin_loss(model_log.encoder_out, model_log.max_embs_pooled, model_log.min_embs_pooled, margin=self.search_marg)
            # Get the two different values
            model_out1 = None
            model_out2 = None
            if i == 0:
                model_out1 = model_logits[-1].encoder_out
            else:
                model_out1 = model_logits[i - 1].encoder_out
            
            if i == len(model_logits) - 1:
                model_out2 = model_logits[0].encoder_out
            else:
                model_out2 = model_logits[i + 1].encoder_out
            
            dist_loss = F.triplet_margin_loss(model_log.encoder_out, model_out1, model_out2, margin=0.7)

            total_loss = stable_loss + np.e**search_loss + dist_loss
            if backward:
                total_loss.backward(retain_graph=True)

            losses.append(total_loss.mean())
        
        return losses