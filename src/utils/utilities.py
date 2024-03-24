import torch
from torch import Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import random

def get_device(is_cuda: str):
    if (is_cuda.lower() == 'y' and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")

def convert_args_str_to_bool(arg: str):
    return True if arg.lower() == 'y' else False
 
def save_loss_plot(losses_train: list, losses_test: list, save_path: str):
    # Plot training losses
    plt.plot([i for i in range(len(losses_train))], losses_train, label='Training Loss')
    plt.plot([i for i in range(len(losses_test))], losses_test, label='Test Loss')
    # Set labels
    #plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()

def calc_game_scores_no_assasin(model_out: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, device: torch.device):
    model_out_expanded = model_out.unsqueeze(1)
    pos_scores = F.cosine_similarity(model_out_expanded, pos_encs, dim=2)
    neg_scores = F.cosine_similarity(model_out_expanded, neg_encs, dim=2)
    neut_scores = F.cosine_similarity(model_out_expanded, neut_encs, dim=2)

    combined_scores = torch.cat((pos_scores, neg_scores, neut_scores), dim=1)
    _, indices = combined_scores.sort(dim=1, descending=True)

    # create reward copies
    pos_reward = torch.zeros(pos_scores.shape[1]).to(device)
    neg_reward = torch.ones(neg_scores.shape[1]).to(device) * 2
    neut_reward = torch.ones(neut_scores.shape[1]).to(device) 

    combined_rewards = torch.cat((pos_reward, neg_reward, neut_reward))
    # Make shape [batch_size, total_number_of_embeddings]
    combined_rewards = combined_rewards.expand((combined_scores.shape[0], combined_rewards.shape[0]))
    # Retrieve the ordered number of rewards, in the order of highest cossine similarity
    rewards = torch.gather(combined_rewards, 1, indices)
    # set all target embeddings to 0 and unwanted embeddings to 1
    non_zero_mask = torch.where(rewards != 0, 1., 0.)
    # Find the total number of correct guesses, equal to the index of the first non-zero value in the mask
    num_correct = torch.argmax(non_zero_mask, dim=1)
    # Find the first incorrect value
    first_incorrect_value = rewards[torch.arange(rewards.size(0)), num_correct]
    return num_correct.float().mean(), first_incorrect_value.mean() - 1

def calc_codenames_score(model_out: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, assas_encs: Tensor, device: torch.device):
    model_out_expanded = model_out.unsqueeze(1)
    assas_expanded = assas_encs.unsqueeze(1)

    pos_scores = F.cosine_similarity(model_out_expanded, pos_encs, dim=2)
    neg_scores = F.cosine_similarity(model_out_expanded, neg_encs, dim=2)
    neut_scores = F.cosine_similarity(model_out_expanded, neut_encs, dim=2)
    assas_scores = F.cosine_similarity(model_out_expanded, assas_expanded, dim=2)

    combined_scores = torch.cat((pos_scores, neg_scores, neut_scores, assas_scores), dim=1)
    _, indices = combined_scores.sort(dim=1, descending=True)

    # create reward copies
    pos_reward = torch.zeros(pos_scores.shape[1]).to(device)
    neg_reward = torch.ones(neg_scores.shape[1]).to(device) * 2
    neut_reward = torch.ones(neut_scores.shape[1]).to(device) 
    assas_reward = torch.ones(assas_scores.shape[1]).to(device) * 3

    combined_rewards = torch.cat((pos_reward, neg_reward, neut_reward, assas_reward))
    # Make shape [batch_size, total_number_of_embeddings]
    combined_rewards = combined_rewards.expand((combined_scores.shape[0], combined_rewards.shape[0]))
    # Retrieve the ordered number of rewards, in the order of highest cosine similarity
    rewards = torch.gather(combined_rewards, 1, indices)
    # set all target embeddings to 0 and unwanted embeddings to 1
    non_zero_mask = torch.where(rewards != 0, 1., 0.)
    # Find the total number of correct guesses, equal to the index of the first non-zero value in the mask
    num_correct = torch.argmax(non_zero_mask, dim=1)
    # Find the first incorrect value
    first_incorrect_value = rewards[torch.arange(rewards.size(0)), num_correct]

    assassin_sum = torch.sum(first_incorrect_value == 3, dim=0)
    neg_sum = torch.sum(first_incorrect_value == 2, dim=0)
    neut_sum = torch.sum(first_incorrect_value == 1, dim=0)

    return num_correct.float().mean(), neg_sum, neut_sum, assassin_sum

class HyperParameters:
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


def slice_board_embeddings(embs: Tensor):
    """Used to randomly remove words/collections of text, from the game board"""
    rand_num = random.randint(1, embs.shape[1])
    return embs[:, :rand_num, :]
