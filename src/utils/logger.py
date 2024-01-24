from torch import Tensor
from utils.utilities import calc_codenames_score

class EpochLogger:
    def __init__(self, data_size: int, batch_size: int, device='cpu', name="Training"):
        self.name = name
        self.total_loss = 0.0

        self.num_correct = 0
        self.neut_sum = 0
        self.neg_sum = 0
        self.assas_sum = 0

        self.data_size = data_size
        self.batch_size = batch_size
        self.device = device
    
    @property
    def avg_correct(self):
        return self.num_correct / self.batch_size
    
    @property
    def avg_loss(self):
        return self.total_loss / self.batch_size
    
    def update_loss(self, loss: Tensor):
        self.total_loss += loss.item()

    def update_results(self, emb: Tensor, pos_emb: Tensor, neg_emg: Tensor, neut_emb: Tensor, assas_emb: Tensor):
        num_correct, neg_sum, neut_sum, assas_sum = calc_codenames_score(emb, pos_emb, neg_emg, neut_emb, assas_emb, self.device)

        self.num_correct += num_correct.item()
        self.neg_sum += neg_sum.item()
        self.neut_sum += neut_sum.item()
        self.assas_sum += assas_sum.item()

    def to_string(self):
        out_str = f"{self.name} Log\n"
        out_str += f"Loss: {self.avg_loss}, Total Score: {self.avg_correct}\n"
        out_str += f"Neutral Guesses: {self.neut_sum}/{self.data_size}, Negative Guesses: {self.neg_sum}/{self.data_size}\n"        
        out_str += f"Assassin Guesses: {self.assas_sum}/{self.data_size}"
        return out_str
    
    def print_log(self):
        output = self.to_string()
        print(output)