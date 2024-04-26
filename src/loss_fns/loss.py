from sentence_transformers import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch import Tensor
from models.many_to_many import ManyOutObj

def triplet_loss(out_embedding, pos_embeddings, neg_embeddings, margin=0):
    pos_sim = util.cos_sim(out_embedding, pos_embeddings)/len(pos_embeddings)
    neg_sim = util.cos_sim(out_embedding, neg_embeddings)/len(neg_embeddings)
    return max(pos_sim - neg_sim + margin, 0)

class RewardSearchLossSmall(nn.Module):
    def __init__(self, margin=0.2, device='cpu', normalize=True, exp_score=False):
        super().__init__()
        self.margin = margin
        self.device = device
        self.normalize = normalize

        self.exp_score = exp_score

    @property
    def name(self):
        return "RewardSearchLossSmall"
    
    def _calc_cos_sim(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, neutral: torch.Tensor):
        pos_score = F.cosine_similarity(anchor, pos, dim=2)
        neg_score = F.cosine_similarity(anchor, neg, dim=2)
        neutral_score = F.cosine_similarity(anchor, neutral, dim=2)
        return pos_score, neg_score, neutral_score
    
    def _calc_final_scores(self, pos_score: torch.Tensor, neg_score: torch.Tensor, neut_score: torch.Tensor):
        neg_score = torch.cat((neg_score, neut_score), dim=1).mean(dim=1)
        pos_score = pos_score.mean(dim=1)

        if self.exp_score:
            pos_score = pos_score.exp2()
            neg_score = neg_score.exp2()
        
        return pos_score, neg_score
    
    def forward(self, model_out: torch.Tensor, search_max: torch.Tensor, search_min: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor, neutral_encs: torch.Tensor):
        model_out_expanded = model_out.unsqueeze(1)

        m_pos_score, m_neg_score, m_neut_score = self._calc_cos_sim(model_out_expanded, pos_encs, neg_encs, neutral_encs)
        pos_score, neg_score = self._calc_final_scores(m_pos_score, m_neg_score, m_neut_score)
        loss_model = F.relu((neg_score - pos_score) + self.margin).mean()
        loss_search = F.triplet_margin_loss(model_out, search_min, search_max, margin=0.9)

        return loss_model + loss_search

class RewardSearchLoss(RewardSearchLossSmall):
    """Current best performing loss function for the full game of Codenames"""
    def __init__(self, model_marg=0.2, search_marg=0.7, device='cpu', normalize=True, exp_score=False):
        super().__init__(model_marg, device, normalize, exp_score)

        self.search_marg = search_marg
    
    @property
    def name(self):
        return "RewardSearchLoss"
    
    def _calc_cos_sim(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, neutral: torch.Tensor, assassin: torch.Tensor):
        pos_score = F.cosine_similarity(anchor, pos, dim=2)
        neg_score = F.cosine_similarity(anchor, neg, dim=2)
        neutral_score = F.cosine_similarity(anchor, neutral, dim=2)
        assassin_score = F.cosine_similarity(anchor, assassin, dim=2)
        return pos_score, neg_score, neutral_score, assassin_score
    
    def _calc_final_scores(self, pos_score: torch.Tensor, neg_score: torch.Tensor, neut_score: torch.Tensor, assas_score: torch.Tensor):
        neg_score = torch.cat((neg_score, neut_score, assas_score), dim=1).mean(dim=1) 
        pos_score = pos_score.mean(dim=1)

        if self.exp_score:
            pos_score = pos_score.exp2()
            neg_score = neg_score.exp2()
        
        return pos_score, neg_score
    
    def forward(self, model_out: torch.Tensor, search_max: torch.Tensor, search_min: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor, neutral_encs: torch.Tensor, assas_encs: torch.Tensor):
        model_out_expanded = model_out.unsqueeze(1)
        assas_expanded = assas_encs.unsqueeze(1)

        m_pos_score, m_neg_score, m_neut_score, m_assas_score = self._calc_cos_sim(model_out_expanded, pos_encs, neg_encs, neutral_encs, assas_expanded)
        pos_score, neg_score = self._calc_final_scores(m_pos_score, m_neg_score, m_neut_score, m_assas_score)
        loss_model = F.relu((neg_score - pos_score) + self.margin).mean()

        loss_search = F.triplet_margin_loss(model_out, search_max, search_min, margin=self.search_marg)

        return loss_model + np.e**(loss_search)

class RewardSearchWithWassersteinLoss(RewardSearchLoss):
    def __init__(self, model_marg=0.2, search_marg=0.7, device='cpu', normalize=True, exp_score=False):
        super().__init__(model_marg, search_marg, device, normalize, exp_score)
    
    def forward(self, model_out: Tensor, wasserstein_rot: Tensor, pos_encs: Tensor, neg_encs: Tensor, neutral_encs: Tensor, assas_encs: Tensor):
        model_out_expanded = model_out.unsqueeze(1)
        assas_expanded = assas_encs.unsqueeze(1)

        m_pos_score, m_neg_score, m_neut_score, m_assas_score = self._calc_cos_sim(model_out_expanded, pos_encs, neg_encs, neutral_encs, assas_expanded)
        pos_score, neg_score = self._calc_final_scores(m_pos_score, m_neg_score, m_neut_score, m_assas_score)
        loss_model = F.relu((neg_score - pos_score) + self.margin).mean()

        return loss_model + wasserstein_rot.mean()
    
class KeypointTriangulationLoss(RewardSearchLoss):
    def __init__(self, model_marg=0.2, search_marg=0.7, device='cpu', normalize=True, exp_score=False):
        super().__init__(model_marg, search_marg, device, normalize, exp_score)
    
    def forward(self, model_out: tuple[Tensor, Tensor], search_outs: tuple[Tensor, Tensor], pos_encs: Tensor, neg_encs: Tensor, neutral_encs: Tensor, assas_encs: Tensor):
        m_pos_out, m_neg_out = model_out

        assas_expanded = assas_encs.unsqueeze(1)
        m_pos_expanded = m_pos_out.unsqueeze(1)
        
        # Calculate topic loss for positive output
        p_pos_score, p_neg_score, p_neut_score, p_assas_score = self._calc_cos_sim(m_pos_expanded, pos_encs, neg_encs, neutral_encs, assas_expanded)
        p_pos_sim, p_neg_sim = self._calc_final_scores(p_pos_score, p_neg_score, p_neut_score, p_assas_score)
        
        p_topic_loss = F.relu((p_neg_sim - p_pos_sim) + self.margin).mean()

        # Calculate triplet loss between 4 points
        s_pos_out, s_neg_out = search_outs
        loss_positive = F.triplet_margin_loss(m_pos_out, s_pos_out, m_neg_out, margin=0.1)
        loss_negative = F.triplet_margin_loss(m_neg_out, s_neg_out, m_pos_out, margin=0.7)

        return p_topic_loss + loss_positive + loss_negative


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
            stable_loss = F.triplet_margin_loss(model_log.model_out, pos_cluster, neg_cluster, margin=self.margin)
            search_loss = F.triplet_margin_loss(model_log.model_out, model_log.max_embs_pooled, model_log.min_embs_pooled, margin=self.search_marg)
            # Get the two different values
            model_out1 = None
            model_out2 = None
            if i == 0:
                model_out1 = model_logits[-1].model_out
            else:
                model_out1 = model_logits[i - 1].model_out
            
            if i == len(model_logits) - 1:
                model_out2 = model_logits[0].model_out
            else:
                model_out2 = model_logits[i + 1].model_out
            
            dist_loss = F.triplet_margin_loss(model_log.model_out, model_out1, model_out2, margin=0.7)

            total_loss = stable_loss + np.e**search_loss + dist_loss
            if backward:
                total_loss.backward(retain_graph=True)

            losses.append(total_loss.mean())
        
        return losses

