from torch import Tensor
from utils.utilities import calc_codenames_score
from collections import Counter
from models.multi_objective_models import MORSpyMaster
import json
import os

class EpochLogger:
    """Logs the result of the codenames model for each epoch, used to record all important information during model training"""
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
    def targ_rate(self):
        return self.target_perc / self.count
    
    @property
    def avg_loss(self):
        return self.total_loss / self.batch_size

    @property
    def json(self) -> json:
        obj = {"Avg Targets": self.targ_rate, "Neutral": self.neut_sum, "Negative": self.neg_sum, "Assassin": self.assas_sum, "Loss": self.avg_loss}
        return json(obj)
    
    @property
    def neut_rate(self) -> float:
        return self.neut_sum / self.data_size

    @property
    def neg_rate(self) -> float:
        return self.neg_sum / self.data_size
    
    @property
    def assas_rate(self) -> float:
        return self.assas_sum / self.data_size
    
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
        out_str += f"Loss: {self.avg_loss}, Target Selection: {self.targ_rate}\n"
        out_str += f"Neutral Guesses: {self.neut_sum}/{self.data_size}, Negative Guesses: {self.neg_sum}/{self.data_size}\n"        
        out_str += f"Assassin Guesses: {self.assas_sum}/{self.data_size}\n"
        return out_str
    
    def print_log(self):
        output = self.to_string()
        print(output)

class EpochLoggerCombined:
    def __init__(self, data_size: int, batch_size: int, num_targets=9, device='cpu', name_model="Model", name_search="Search") -> None:
        self.log_model = EpochLogger(data_size=data_size, batch_size=batch_size, num_targets=num_targets, device=device, name=name_model)
        self.log_search = EpochLogger(data_size=data_size, batch_size=batch_size, num_targets=num_targets, device=device, name=name_search)
    
    def update_results(self, 
                       model_out: Tensor, 
                       search_out: Tensor, 
                       pos_emb: Tensor, 
                       neg_emb: Tensor, 
                       neut_emb: Tensor, 
                       assas_emb: Tensor):
        self.log_model.update_results(model_out, pos_emb, neg_emb, neut_emb, assas_emb)
        self.log_search.update_results(search_out, pos_emb, neg_emb, neut_emb, assas_emb)
    
    def update_loss(self, loss: Tensor):
        self.log_model.update_loss(loss)
        self.log_search.update_loss(loss)
    
    def print_log(self):
        output_model = self.log_model.to_string()
        print(output_model)

        output_search = self.log_search.to_string()
        print(output_search)

class LogInfo:
    def __init__(self, name: str) -> None:
        self.loss = []
        self.targ_rate = []
        self.neut_rate = []
        self.neg_rate = []
        self.assas_rate = []
        self.name = name
    
    def update_log(self, logger: EpochLogger):
        self.loss.append(logger.avg_loss)
        self.targ_rate.append(logger.targ_rate)
        self.neut_rate.append(logger.neut_rate)
        self.neg_rate.append(logger.neg_rate)
        self.assas_rate.append(logger.assas_rate)

    def json(self):
        obj = {"Name": self.name, 
               "Loss": self.loss, 
               "Target Rate": self.targ_rate, 
               "Neutral Rate": self.neut_rate, 
               "Negative Rate": self.neg_rate, 
               "Assassin Rate": self.assas_rate
               }
        return obj


class TrainLogger:
    def __init__(self) -> None:
        self.train_loggers = []
        self.valid_loggers = []
    
    def add_loggers(self, train_log, val_log):
        self.train_loggers.append(train_log)
        self.valid_loggers.append(val_log)

    def _update_log(self, log: LogInfo, logger: EpochLogger):
        log.loss.append(logger.avg_loss)


    def _combine_loggers(self, loggers: list[EpochLoggerCombined]) -> tuple[LogInfo, LogInfo]:
        model_log = LogInfo(loggers[0].log_model.name)
        search_log = LogInfo(loggers[0].log_search.name)

        for logger in loggers:
            model_log.update_log(logger.log_model)
            search_log.update_log(logger.log_search)
        
        return model_log, search_log

    def save_results(self, directory: str):
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # TODO: Refactor

        # Save train logs
        train_path = directory + "training.json"
        train_model, train_search = self._combine_loggers(self.train_loggers)

        obj = {"Model": train_model.json(), "Search": train_search.json()}
        with open(train_path, 'w') as file:
            json.dump(obj, file)
        
        # Save validation logs
        valid_path = directory + "validation.json"
        valid_model, valid_search = self._combine_loggers(self.valid_loggers)

        obj = {"Model": valid_model.json(), "Search": valid_search.json()}
        with open(valid_path, 'w') as file:
            json.dump(obj, file)

class TrainLoggerMany:
    def __init__(self) -> None:
        self.epoch_loggers_train = []
        self.epoch_loggers_valid = []
    
    def add_loggers(self, train_log: list[EpochLoggerCombined], valid_log: list[EpochLoggerCombined]):
        self.epoch_loggers_train.append(train_log)
        self.epoch_loggers_valid.append(valid_log)
    
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