from datasets.dataset import CodeGiverDataset
from sentence_transformers import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from loss_fns.loss import CombinedTripletLoss
import numpy as np
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
        
