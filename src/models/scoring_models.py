import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import utils.utilities as utils


class ScoringModel(nn.Module):
    def __init__(self):
        super(ScoringModel, self).__init__()

        self.reducer = nn.Sequential(
            nn.Linear(3072, 2056),
            nn.Tanh(),
            nn.Linear(2056, 1024),
            nn.Tanh(),
            nn.Linear(1024, 768)
        )

        self.classifier_layer = nn.Sequential(
            nn.Linear(1536, 768),
            nn.GELU(),
            nn.Linear(768, 250),
            nn.GELU(),
            nn.Linear(250, 1)
        )

    def forward(self, 
                pos_embs: Tensor, 
                neg_embs: Tensor, 
                neut_embs: Tensor, 
                assas_emb: Tensor,
                search_out: Tensor) -> Tensor:
        # Cluster embeddings
        pos_emb = utils.cluster_embeddings(pos_embs, dim=0)
        neg_emb = utils.cluster_embeddings(neg_embs, dim=0)
        neut_emb = utils.cluster_embeddings(neut_embs, dim=0)

        cat_emb = torch.cat((pos_emb, neg_emb, neut_emb, assas_emb), dim=0)
        cat_emb = cat_emb.expand((search_out.shape[0], -1))

        reduced = self.reducer(cat_emb)
        cat_emb = torch.cat((reduced, search_out), dim=1)
        pred = self.classifier_layer(cat_emb)
        return pred
