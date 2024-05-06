from utils.utilities import convert_args_str_to_bool
import json

class HyperParameters:
    """Hyperparameter object to store all important information used during model training"""
    def __init__(self, args):
        self.model_marg = args.m_marg
        self.search_marg = args.s_marg
        self.learning_rate = args.lr
        self.gamma = args.gamma
        self.weight_decay = args.w_decay
        self.n_epochs = args.e
        self.batch_size = args.b
        
        self.search_window = args.sw
        self.neut_weight = args.neut_weight
        self.neg_weight = args.neg_weight
        self.assas_weight = args.assas_weight
        self.using_sentences = convert_args_str_to_bool(args.sentences)
        self.bias = convert_args_str_to_bool(args.bias)
        self.dynamic_board = convert_args_str_to_bool(args.dynamic_board)
        self.backbone = args.backbone
        self.emb_size = self._get_emb_size()
        self.seperator = args.sep
        self.burst_counter = False
    
    def _get_emb_size(self):
        if (self.backbone == "all-mpnet-base-v2"):
            return 768
        if (self.backbone == "all-MiniLM-L6-v2"):
            return 384
        return 768

    
    def save_params(self, filepath: str):
        data = {
            "model_margin": self.model_marg,
            "search_margin": self.search_marg,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "weight_decay": self.weight_decay,
            "num_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "vocab_size": self.search_window,
            "neut_weight": self.neut_weight,
            "neg_weight": self.neg_weight,
            "assas_weight": self.assas_weight,
            "sentence_trained": self.using_sentences,
            "bias": self.bias,
            "backbone": self.backbone
        }
        with open(filepath, 'w') as file:
            json.dump(data, file)


class RerankerHyperParameter:
    def __init__(self, args) -> None:
        self.n_epochs = args.e
        self.batch_size = args.b


        self.learning_rate = args.lr
        self.gamma = args.gamma
        self.weight_decay = args.w_decay
        self.search_window = args.sw
        self.dynamic_board = convert_args_str_to_bool(args.dynamic_board)
    
    def save_params(self, filepath: str):
        data = {
            "epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "weight_decay": self.weight_decay,
            "search_window": self.search_window,
            "dynamic_board": self.dynamic_board
        }

        with open(filepath, 'w') as file:
            json.dump(data, file)
