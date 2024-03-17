from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import DebertaConfig, DebertaModel, DebertaTokenizer
from utils.vector_search import VectorSearch
import numpy as np
import os
from models.base_models import SimpleCodeGiver

# Environment variable need to be set to remove 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class ConvLayer(nn.Module):
    """
    Convolution layer for the pooling of input embeddings

    - Multiple combinations of conv layers with varying channel lengths and kernel sizes were attempted
    - Results were only marginally better than guessing, pooling appears to be the best option
    """
    def __init__(self, in_channels=9) -> None:
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=100, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=100, out_channels=300, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=300, out_channels=500, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=500, out_channels=768, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool1d(768),
        )
    def forward(self, x):
        out = self.conv_layer(x)
        return out.squeeze()


class ObjectiveEncoder(SimpleCodeGiver):
    def __init__(self, in_channels=9, model_name='all-mpnet-base-v2'):
        super().__init__(model_name)

        self.pos_encoder = ConvLayer(in_channels)
        self.neg_encoder = ConvLayer(in_channels)
    
    def forward(self, pos_embeddings: torch.Tensor, neg_embeddings: torch.Tensor):
        pos_out = self.pos_encoder(pos_embeddings)
        neg_out = self.neg_encoder(neg_embeddings)
        # Remove excess dim
        pos_out = pos_out.squeeze(1)
        neg_out = neg_out.squeeze(1)

        concatenated = torch.cat((pos_out, neg_out), 1)
        out = self.fc(concatenated)
        return F.normalize(out, p=2, dim=1)

class ConvolutionSearch(nn.Module):
    def __init__(self, vector_db: VectorSearch, device: torch.device, in_channels=9, model_name='all-mpnet-base-v2'):
        super().__init__()
        self.encoder = ObjectiveEncoder(in_channels, model_name)
        self.vector_db = vector_db
        self.device = device
    
    def forward(self, pos_embeddings: torch.Tensor, neg_embeddings: torch.Tensor):
        out = self.encoder(pos_embeddings, neg_embeddings)
        words, embeddings, dist = self.vector_db.search(out, num_results=1)
        embeddings = torch.tensor(embeddings).to(self.device).squeeze(1)
        if self.training:
            return out, embeddings
        return words, embeddings, dist







        

