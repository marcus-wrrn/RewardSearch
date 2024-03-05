import torch
from torch.optim.lr_scheduler import ExponentialLR
from loss_fns.loss import  RewardSearchLossSmall
from torch.utils.data import DataLoader
from models.multi_objective_models import MORSpyMasterSmall
from datasets.dataset import CodeDatasetMultiObjective
import numpy as np
import datetime
import argparse
import utils.utilities as utils
from utils.vector_search import VectorSearch
from utils.hidden_vars import BASE_DIR
from utils.utilities import calc_game_scores_no_assasin
import torch.nn.functional as F

def init_hyperparameters(model: MORSpyMasterSmall, device, normalize_reward):
    loss_fn = RewardSearchLossSmall(margin=0.6, device=device, normalize=normalize_reward)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    return loss_fn, optimizer, scheduler

class LossResults:
    def __init__(self, data_size: int) -> None:
        self.tot_pos = 0.0
        self.tot_neg = 0.0
        self.tot_neut = 0.0
        self.size = data_size
    
    def add_results(self, results: tuple):
        assert len(results) == 3
        pos, neg, neut = results

        self.tot_pos += pos.mean(0).item()
        self.tot_neg += neg.mean(0).item()
        self.tot_neut += neut.mean(0).item()
    
    @property
    def results_str(self) -> str:
        return f"Positive: {self.tot_pos/self.size}, Negative: {self.tot_neg/self.size}, Neutral: {self.tot_neut/self.size}"



@torch.no_grad()
def validate(model: MORSpyMasterSmall, valid_loader: DataLoader, loss_fn: RewardSearchLossSmall, device: torch.device):
    total_loss = 0.0
    total_score = 0.0
    incorrect_guess = 0.0
    for i, data in enumerate(valid_loader, 0):
        pos_sents, neg_sents, neut_sents, pos_embeddings, neg_embeddings, neut_embeddings = data
        pos_embeddings, neg_embeddings, neut_embeddings = pos_embeddings.to(device), neg_embeddings.to(device), neut_embeddings.to(device)

        model_out, search_out, search_max, search_min = model(pos_embeddings, neg_embeddings, neut_embeddings)

        #loss = F.triplet_margin_loss(model_out, search_min, search_max, margin=0.2)
        loss = loss_fn(model_out, search_max, search_min, pos_embeddings, neg_embeddings, neut_embeddings)
        score, results = calc_game_scores_no_assasin(search_out, pos_embeddings, neg_embeddings, neut_embeddings, device)

        incorrect_guess += results.item()
        total_loss += loss.item()
        total_score += score.item()
    avg_loss = total_loss / len(valid_loader)
    avg_score = total_score / len(valid_loader)
    avg_incorrect_guess = incorrect_guess / len(valid_loader)

    return avg_loss, avg_score, avg_incorrect_guess

def train(n_epochs: int, model: MORSpyMasterSmall, train_loader: DataLoader, valid_dataloader: DataLoader, device: torch.device, model_path: str, normalize_reward: bool):
    loss_fn, optimizer, scheduler = init_hyperparameters(model, device, normalize_reward)
    print("Training")
    model.train()

    losses_train = []
    losses_valid = []

    print(f"Starting training at: {datetime.datetime.now()}")
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")
        loss_train = 0.0
        total_score = 0.0

        total_incorrect_guess = 0.0
        for i, data in enumerate(train_loader, 0):
            if (i % 100 == 0):
                print(f"{datetime.datetime.now()}: Iteration: {i}/{len(train_loader)}")
            pos_sents, neg_sents, neutral_sents, pos_embeddings, neg_embeddings, neut_embeddings = data
            # Put embeddings on device
            pos_embeddings = pos_embeddings.to(device)
            neg_embeddings = neg_embeddings.to(device)
            neut_embeddings = neut_embeddings.to(device)
            
            optimizer.zero_grad()
            model_out, search_out, search_max, search_min = model(pos_embeddings, neg_embeddings, neut_embeddings)


            loss = loss_fn(model_out, search_max, search_min, pos_embeddings, neg_embeddings, neut_embeddings)
            score, results = calc_game_scores_no_assasin(search_out, pos_embeddings, neg_embeddings, neut_embeddings, device)
            #loss, score, results = loss_fn(logits_out, logits_search, pos_embeddings, neg_embeddings, neut_embeddings)

            total_incorrect_guess += results.item()
            total_score += score.item()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        scheduler.step()
        avg_loss = loss_train / len(train_loader)
        avg_score = total_score / len(train_loader) # technically average of average scores
        avg_incorrect_guess = total_incorrect_guess / len(train_loader)
        losses_train.append(avg_loss)
        # Validate model output
        validation_loss, validation_score, validation_answers = validate(model, valid_dataloader, loss_fn, device)
        losses_valid.append(validation_loss)
        # Log and print save model parameters
        training_str = f"{datetime.datetime.now()}, Epoch: {epoch}\nTraining Loss: {avg_loss}, Training Score: {avg_score}\nValidation Loss: {validation_loss}, Validation Score: {validation_score}\n"
        training_str += f"Incorrect Guess Train: {avg_incorrect_guess}, Validation: {validation_answers}"
        print(training_str)
        if len(losses_train) == 1 or losses_train[-1] < losses_train[-2]:
                torch.save(model.state_dict(), model_path)
        
    return losses_train, losses_valid

def main(args):
    device = utils.get_device(args.cuda)
    # extract cli arguments
    code_data = args.code_data
    guess_data = args.guess_data
    val_guess_data = args.val_guess_data
    model_out = args.model_out
    loss_out = args.loss_out

    normalize_reward = True if args.norm.lower() == 'y' else False

    print(f"Device: {device}")
    train_dataset = CodeDatasetMultiObjective(code_dir=code_data, game_dir=guess_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=4)

    valid_dataset = CodeDatasetMultiObjective(code_dir=code_data, game_dir=val_guess_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size=50, num_workers=4)

    vector_db = VectorSearch(train_dataset, prune=True)
    model = MORSpyMasterSmall(vector_db, device)
    model.to(device)

    losses_train, losses_valid = train(n_epochs=args.e, model=model, train_loader=train_dataloader, valid_dataloader=valid_dataloader, device=device, model_path=model_out, normalize_reward=normalize_reward)
    utils.save_loss_plot(losses_train, losses_valid, save_path=loss_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, help="Number of epochs", default=10)
    parser.add_argument('-b', type=int, help="Batch Size", default=400)
    parser.add_argument('-code_data', type=str, help="Codenames dataset path", default=BASE_DIR + "data/words.json")
    parser.add_argument('-guess_data', type=str, help="Geuss words dataset path", default=BASE_DIR + "data/codewords_16neg_data_valid.json")
    parser.add_argument('-norm', type=str, help="Whether to normalize reward function, [Y/n]", default='Y')
    parser.add_argument('-val_guess_data', type=str, help="Filepath for the validation dataset", default=BASE_DIR + "data/codewords_full_data_mini.json")
    parser.add_argument('-model_out', type=str, default=BASE_DIR + "/saved_models/test.pth")
    parser.add_argument('-loss_out', type=str, default=BASE_DIR + "/saved_models/test.png")
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    args = parser.parse_args()
    main(args)