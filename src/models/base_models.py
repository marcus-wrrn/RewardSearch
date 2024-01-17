from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import DebertaConfig, DebertaModel, DebertaTokenizer
from utils.vector_search import VectorSearch
import numpy as np
import os

# Environment variable need to be set to remove 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Take attention mask into account for correct averaging
def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def mean_pooling_ein(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    mask = attention_mask.unsqueeze(-1)  # Add an extra dimension for broadcasting
    sum_embeddings = torch.einsum('ijk,ij->ik', token_embeddings, mask.float())
    sum_mask = mask.sum(1)
    return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

class SentenceEncoderRaw(nn.Module):
    """
    Sentence Transformer used for encoding input sentences. Does not use sentence_transformers library
    Allows for fine-tuning
    """
    def __init__(self, 
                 device="cpu",
                 tokenizer_path="sentence-transformers/all-mpnet-base-v2", 
                 model_path="sentence-transformers/all-mpnet-base-v2" 
                ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.device = device

    def tokenize_sentences(self, sentences):
        return self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
    
    def get_token_embeddings(self, tokenized_sentences):
        return self.model(**tokenized_sentences)

    def forward(self, sentences, normalize=True) -> torch.Tensor:
        tokenized_sents = self.tokenize_sentences(sentences)
        token_embeddings = self.model(**tokenized_sents) 
        sentence_embeddings = mean_pooling(token_embeddings, tokenized_sents['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1) if normalize else sentence_embeddings
    
    
class SentenceEncoder(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2'):
        super(SentenceEncoder, self).__init__()

        self.name = model_name
        self.encoder = SentenceTransformer(self.name)
        
    def forward(self, text: torch.Tensor):
        encodings = self.encoder.encode(text, convert_to_tensor=True)
        if encodings.ndim == 1:
            encodings = encodings.unsqueeze(0)
        # out = self.fc(encodings)
        return encodings


class EncoderLayer(nn.Module):
    """
    A neural network module for encoding sentences using the SentenceTransformer model.

    This class supports encoding sentences with different labels (e.g., positive, negative, neutral, assassin).
    Each label type can be optionally included in the encoding process.

    Attributes:
        pos_encoder (SentenceTransformer): Encoder for positive sentences.
        neg_encoder (SentenceTransformer): Encoder for negative sentences.
        neutral_encoder (SentenceTransformer): Encoder for neutral sentences.
        assassin_encoder (SentenceTransformer): Encoder for assassin words.
    """

    def __init__(self, has_pos=True, has_neg=True, has_neutral=True, has_assassin=False) -> None:
        super().__init__()

        model_name = 'all-mpnet-base-v2'
        self.encoders = {
            'pos': SentenceEncoder(model_name) if has_pos else None,
            'neg': SentenceEncoder(model_name) if has_neg else None,
            'neutral': SentenceEncoder(model_name) if has_neutral else None,
            'assassin': SentenceEncoder(model_name) if has_assassin else None
        }

    def forward(self, pos_sents='', neg_sents='', neutral_sents='', assassin_word=''):
        encoded = {
            'pos': self.encoders['pos'](pos_sents) if self.encoders['pos'] else None,
            'neg': self.encoders['neg'](neg_sents) if self.encoders['neg'] else None,
            'neutral': self.encoders['neutral'](neutral_sents) if self.encoders['neutral'] else None,
            'assassin': self.encoders['assassin'](assassin_word) if self.encoders['assassin'] else None
        }

        return encoded['pos'], encoded['neg'], encoded['neutral'], encoded['assassin']

class OldCodeGiver(nn.Module):
    """
    Only encodes positive and negative sentences

    Currently most stable model
    """
    def __init__(self, model_name='all-mpnet-base-v2'):
        super().__init__()
        self.name = model_name
        
        self.pos_encoder = SentenceEncoder(model_name)
        self.neg_encoder = SentenceEncoder(model_name)
        self.fc = nn.Sequential(
            nn.Linear(1536, 1250),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1250, 1000),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 768)
        )
    
    def forward(self, pos_texts: str, neg_texts: str):
        pos_emb = self.pos_encoder(pos_texts)
        neg_emb = self.neg_encoder(neg_texts)
        
        concatenated = torch.cat((pos_emb, neg_emb), 1)
        out = self.fc(concatenated)
        return F.normalize(out, p=2, dim=1)

class SimpleCodeGiver(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2'):
        super().__init__()
        self.name = model_name
        
        self.encoder = SentenceEncoder(model_name)
        self.fc = nn.Sequential(
            nn.Linear(1536, 1250),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1250, 1000),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 768)
        )
    
    def forward(self, pos_texts: str, neg_texts: str):
        pos_emb = self.encoder(pos_texts)
        neg_emb = self.encoder(neg_texts)
        
        concatenated = torch.cat((pos_emb, neg_emb), 1)
        out = self.fc(concatenated)
        return F.normalize(out, p=2, dim=1)

class SimpleCodeGiverPooled(SimpleCodeGiver):
    """
    Like SimpleCodeGiver but mean pools the input embeddings using text strings
    """
    def __init__(self, model_name='all-mpnet-base-v2'):
        super().__init__(model_name)
    
    def forward(self, pos_embeddings: torch.Tensor, neg_embeddings: torch.Tensor):
        pos_emb = pos_embeddings.mean(dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)
        neg_emb = neg_embeddings.mean(dim=1)
        neg_emb = F.normalize(neg_emb, p=2, dim=1)
        
        concatenated = torch.cat((pos_emb, neg_emb), 1)
        out = self.fc(concatenated)
        return F.normalize(out, p=2, dim=1)
    
class CodeGiverRaw(nn.Module):
    """Uses fine-tunable encoder for its backbone"""
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # Initialize encoders
        self.encoder = SentenceEncoderRaw(device)

        self.fc = nn.Sequential(
            nn.Linear(1536, 1250),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1250, 1000),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 768)
        )

    def forward(self, pos_texts: str, neg_texts: str):
        pos_embs = self.encoder(pos_texts)
        neg_embs = self.encoder(neg_texts)

        concatenated = torch.cat((pos_embs, neg_embs), dim=1)
        out = self.fc(concatenated)
        return F.normalize(out, p=2, dim=1) 


class CodeSearch(OldCodeGiver):
    """Implements vector search as part of its forward pass"""
    def __init__(self, vector_db: VectorSearch, device: torch.device, model_name='all-mpnet-base-v2'):
        super().__init__(model_name)
        self.vector_db = vector_db
        self.device = device
        self.selector_model = SentenceEncoderRaw(device=self.device)

        self.to(self.device)
    
    def forward(self, pos_texts: str, neg_texts: str):
        out = super().forward(pos_texts, neg_texts)

        # Perform search
        words, embeddings, _ = self.vector_db.search(out, num_results=1)
        words = tuple([word[0] for word in words])
        #words = torch.tensor(words).to(self.device)
        # search_out = torch.tensor(embeddings).to(self.device)
        search_out = self.selector_model(words)

        return out, search_out

class CodeSearchMeanPool(SimpleCodeGiverPooled):
    def __init__(self, vector_db: VectorSearch, device: torch.device, model_name='all-mpnet-base-v2'):
        super().__init__(model_name)
        self.vector_db = vector_db
        self.device = device

    def forward(self, pos_embeddings: torch.Tensor, neg_embeddings: torch.Tensor):
        out = super().forward(pos_embeddings, neg_embeddings)
        words, embeddings, dist = self.vector_db.search(out, num_results=1)
        embeddings = torch.tensor(embeddings).to(self.device).squeeze(1)
        if self.training:
            return out, embeddings
        return words, embeddings, dist

class CodeSearchDualNet(SimpleCodeGiver):
    def __init__(self, vector_db: VectorSearch, device: torch.device, model_name='all-mpnet-base-v2'):
        super().__init__(model_name)
        self.vector_db = vector_db
        self.device = device

        self.search_encoder = SentenceEncoderRaw(device=self.device)

        self.search_fc = nn.Sequential(
            nn.Linear(768, 512),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 512),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 768),
        )
    
    @torch.no_grad
    def infer(self, pos_texts: str, neg_texts: str):
        out = super().forward(pos_texts, neg_texts)
        words, embeddings, dist = self.vector_db.search(out, num_results=1)
        embeddings = torch.tensor(embeddings).to(self.device).squeeze(1)

        return words, embeddings, dist

    def forward(self, pos_texts: str, neg_texts: str):
        out = super().forward(pos_texts, neg_texts)
        # Search for and process embeddings
        words, embeddings, _ = self.vector_db.search(out, num_results=1)
        embeddings = torch.tensor(embeddings).to(self.device).squeeze(1)
        return out, embeddings



def main():
    # testing
    positive_words = "cat dog house"
    negative_words = "car train sun"
    device = torch.device('cuda')

    encoder = OldCodeGiver()
    encoder.to(device)
    vals = encoder(positive_words, negative_words)

    print(vals.shape)
    print("Done")

if __name__ == "__main__":
    main()