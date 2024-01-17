import numpy as np
from datasets.dataset import CodeGiverDataset
import faiss
import torch


class VectorSearch():
    def __init__(self, dataset: CodeGiverDataset, prune=False, n_dim=768, n_neighbours=32, useGuessData=True) -> None:
        self.vocab_words, self.vocab_embeddings = dataset.get_vocab(guess_data=useGuessData) if not prune else dataset.get_pruned_vocab()
        self.vocab_words = np.array(self.vocab_words)
        # Process embeddings
        self.vocab_embeddings = np.array(self.vocab_embeddings).astype(np.float32)
        # Initialize index + add embeddings
        self.index = faiss.IndexHNSWFlat(n_dim, n_neighbours)
        self.index.add(self.vocab_embeddings)
    
    def search(self, logits: torch.Tensor, num_results=20):
        # detach tensor from device and convert it to numpy for faiss compatibility
        search_input = logits.detach().cpu().numpy()
        # D: L2 distance from input, I: index of result
        D, I = self.index.search(search_input, num_results)
        # Map index values to words
        words = self.vocab_words[I]
        embeddings = self.vocab_embeddings[I]
        return words, embeddings, D

    def index_to_words(self, index):
        return self.vocab_words[index]
    
