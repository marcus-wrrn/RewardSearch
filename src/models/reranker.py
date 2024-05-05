import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Reranker(nn.Module):
    def __init__(self, vocab_size=80, head_num=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.head_num = head_num

        input_size = vocab_size * head_num
        self.fc = nn.Sequential(
            nn.Linear(input_size, int(input_size * 0.7)),
            nn.ReLU(),
            nn.Linear(int(input_size * 0.7), input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 3)
        )

    def forward(self, tri_out: Tensor, text_embs: Tensor):
        num_heads = tri_out.shape[1]
        if num_heads != self.head_num:
            raise ValueError(f"Number of heads must be the same size, expected {self.head_num} got {num_heads}")
        
        sim_scores = []
        for i in range(self.head_num):
            head = tri_out[:, i, :]
            head = head.unsqueeze(1)

            cos_sim = F.cosine_similarity(head, text_embs, dim=2)
            sim_scores.append(cos_sim)

        sim_scores = torch.stack(sim_scores, dim=2)
        sim_scores = sim_scores.view(sim_scores.shape[0], -1)
        
        out = self.fc(sim_scores)

        return out





if __name__ == "__main__":
    batch_size = 500
    num_heads = 3
    emb_size = 768
    vocab_size = 80

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Reranker(vocab_size, num_heads)
    
    model.to(device)
    model.train()

    tri_out = torch.rand(batch_size, num_heads, emb_size).to(device)
    text_embs = torch.rand(batch_size, vocab_size, emb_size).to(device)

    model(tri_out, text_embs)

