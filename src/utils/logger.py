from torch import Tensor
from utils.utilities import calc_codenames_score
from collections import Counter
from models.multi_objective_models import MORSpyMaster
import json
import os

class EpochLogger:
    def __init__(self, data_size: int, batch_size: int, num_targets=9, device='cpu', name="Training"):
        self.name = name
        self.total_loss = 0.0

        self.target_perc = 0
        self.target_percent = 0
        self.num_targets = []

        self.neut_sum = 0
        self.neg_sum = 0
        self.assas_sum = 0

        self.data_size = data_size
        self.batch_size = batch_size
        self.device = device

        self.count = 0
    
    @property
    def avg_target_perc(self):
        return self.target_perc / self.count
    
    @property
    def avg_loss(self):
        return self.total_loss / self.batch_size

    @property
    def json(self) -> json:
        obj = {"Avg Targets": self.avg_target_perc, "Neutral": self.neut_sum, "Negative": self.neg_sum, "Assassin": self.assas_sum, "Loss": self.avg_loss}
        return json(obj)
    
    def update_loss(self, loss: Tensor):
        self.total_loss += loss.item()

    def update_results(self, emb: Tensor, pos_emb: Tensor, neg_emg: Tensor, neut_emb: Tensor, assas_emb: Tensor):
        self.count += 1
        num_correct, neg_sum, neut_sum, assas_sum = calc_codenames_score(emb, pos_emb, neg_emg, neut_emb, assas_emb, self.device)

        self.target_perc += num_correct.item() / pos_emb.shape[1]
        self.num_targets.append(pos_emb.shape[1]) 
        self.neg_sum += neg_sum.item()
        self.neut_sum += neut_sum.item()
        self.assas_sum += assas_sum.item()

        

    def to_string(self):
        out_str = f"{self.name} Log\n"
        out_str += f"Loss: {self.avg_loss}, Target Selection: {self.avg_target_perc}\n"
        out_str += f"Neutral Guesses: {self.neut_sum}/{self.data_size}, Negative Guesses: {self.neg_sum}/{self.data_size}\n"        
        out_str += f"Assassin Guesses: {self.assas_sum}/{self.data_size}\n"
        return out_str
    
    def print_log(self):
        output = self.to_string()
        print(output)


class TrainLogger:
    def __init__(self, num_epochs: int) -> None:
        self.train_loggers_model = []
        self.train_loggers_search = []

        self.valid_loggers_model = []
        self.valid_loggers_search = []
    
    def add_loggers(self, tmodel_log, tsearch_log, vmodel_log, vsearch_log):
        self.train_loggers_model.append(tmodel_log)
        self.train_loggers_search.append(tsearch_log)
        self.valid_loggers_model.append(vmodel_log)
        self.valid_loggers_search.append(vsearch_log)

    def _combine_loggers(self, loggers: list[EpochLogger]):
        loss = []
        targ_rate = []
        neut_rate = []
        neg_rate = []
        assas_rate = []

        for logger in loggers:
            loss.append(logger.avg_loss)

            targ_rate.append(logger.avg_target_perc)
            neut_rate.append(logger.neut_sum / logger.data_size)
            neg_rate.append(logger.neg_sum / logger.data_size)
            assas_rate.append(logger.assas_sum / logger.data_size)
        
        # TODO: Fix scuffed name 
        obj = {"Name": loggers[0].name, "Loss": loss, "Target Rate": targ_rate, "Neutral Rate": neut_rate, "Negative Rate": neg_rate, "Assassin Rate": assas_rate}
        return obj

    def save_results(self, directory: str):
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # TODO: Refactor

        # Save train logs
        train_path = directory + "training.json"

        train_model = self._combine_loggers(self.train_loggers_model)
        train_search = self._combine_loggers(self.train_loggers_search)

        obj = {"Model": train_model, "Search": train_search}
        with open(train_path, 'w') as file:
            json.dump(obj, file)
        
        # Save validation logs
        valid_path = directory + "validation.json"
        valid_model = self._combine_loggers(self.valid_loggers_model)
        valid_search = self._combine_loggers(self.valid_loggers_search)

        obj = {"Model": valid_model, "Search": valid_search}
        with open(valid_path, 'w') as file:
            json.dump(obj, file)


class TestLogger(EpochLogger):
    def __init__(self, data_size: int, batch_size: int, device='cpu', name="Training"):
        super().__init__(data_size, batch_size, device=device, name=name)
        self.words = []
    
    def update_results(self, words: list, emb: Tensor, pos_emb: Tensor, neg_emg: Tensor, neut_emb: Tensor, assas_emb: Tensor):
        super().update_results(emb, pos_emb, neg_emg, neut_emb, assas_emb)

        self.words.extend([word[0] for word in words])

    def print_word_distribution(self):
        word_dist = Counter(self.words)
        print(f"Word Distribution: ")
        print(word_dist)