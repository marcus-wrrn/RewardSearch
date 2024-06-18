import numpy as np
from datasets.dataset import CodeGiverDataset
import faiss
import torch


class VectorSearch:
    def __init__(self, dataset: CodeGiverDataset, prune=False, n_dim=768, n_neighbours=32, useGuessData=True) -> None:
        self.vocab_texts, self.vocab_embeddings = dataset.get_vocab(guess_data=useGuessData) if not prune else dataset.get_pruned_vocab()
        self.vocab_texts = np.array(self.vocab_texts)
        # Process embeddings
        self.vocab_embeddings = np.array(self.vocab_embeddings).astype(np.float32)
        # Initialize index + add embeddings
        self.index = faiss.IndexHNSWFlat(n_dim, n_neighbours)
        self.index.add(self.vocab_embeddings)

    def vocab_add(self, text: str, emb: torch.Tensor):
        emb.detach().cpu().numpy()
        self.index.add(emb)
        self.vocab_embeddings.append(emb)
        self.vocab_words(text)
    
    def search(self, logits: torch.Tensor, num_results=20):
        # detach tensor from device and convert it to numpy for faiss compatibility
        search_input = logits.detach().cpu().numpy()
        # D: L2 distance from input, I: index of result
        D, I = self.index.search(search_input, num_results)
        # Map index values to words
        words = self.vocab_texts[I]
        embeddings = self.vocab_embeddings[I]
        return words, embeddings, D

    def index_to_texts(self, index):
        return self.vocab_texts[index]
    
    def save_index(self, filedir: str):
        index_path = filedir + 'index'
        faiss.write_index(self.index, index_path)
        vocab_path = filedir + 'vocab.npy'
        np.save(vocab_path, self.vocab_embeddings)
        text_path = filedir + 'vocab_texts.npy'
        np.save(text_path, self.vocab_texts)
    
    def load_index(self, filedir: str):
        index_path = filedir + 'index'
        self.index = faiss.read_index(index_path)
        vocab_path = filedir + 'vocab.npy'
        self.vocab_embeddings = np.load(vocab_path)
        text_path = filedir + 'vocab_texts.npy'
        self.vocab_texts = np.load(text_path)
    
