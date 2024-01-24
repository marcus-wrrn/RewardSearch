import torch
from torch.optim.lr_scheduler import ExponentialLR
from loss_fns.loss import  RewardSearchLoss, KeypointTriangulationLoss
from torch.utils.data import DataLoader
from models.multi_objective_models import MORSpyDualHead
from datasets.dataset import CodeNamesDataset
import numpy as np
import datetime
import argparse
import utils.utilities as utils

from utils.vector_search import VectorSearch
from utils.hidden_vars import BASE_DIR
from utils.logger import EpochLogger

def init_hyperparameters(model: MORSpyDualHead, device, normalize_reward):
    loss_fn = KeypointTriangulationLoss(model_marg=0.7, search_marg=0.8, device=device, normalize=normalize_reward)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    return loss_fn, optimizer, scheduler

@torch.no_grad()
def validate(model: MORSpyDualHead, valid_loader: DataLoader, loss_fn: RewardSearchLoss, device: torch.device):
    val_logger = EpochLogger(len(valid_loader.dataset), len(valid_loader), device, "Validation")
    for i, data in enumerate(valid_loader, 0):
        pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings = data[1]
        pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings = pos_embeddings.to(device), neg_embeddings.to(device), neut_embeddings.to(device), assas_embeddings.to(device)

        model_logits, search_logits, word_emb_output = model(pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)

        loss = loss_fn(model_logits, search_logits, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)
        val_logger.update_results(word_emb_output, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)
        val_logger.update_loss(loss)

    return val_logger

def train(n_epochs: int, model: MORSpyDualHead, train_loader: DataLoader, valid_dataloader: DataLoader, device: torch.device, model_path: str, normalize_reward: bool, use_model_out: bool):
    loss_fn, optimizer, scheduler = init_hyperparameters(model, device, normalize_reward)
    print("Training")
    model.train()

    losses_train = []
    losses_valid = []

    print(f"Starting training at: {datetime.datetime.now()}")
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")
        train_logger_search = EpochLogger(len(train_loader.dataset), len(train_loader), device, name="Training with Search Output")
        train_logger_model = EpochLogger(len(train_loader.dataset), len(train_loader), device, name="Training with Model Output")
        for i, data in enumerate(train_loader, 0):
            if (i % 100 == 0):
                print(f"{datetime.datetime.now()}: Iteration: {i}/{len(train_loader)}")
            pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings = data[1]

            # Put embeddings on device
            pos_embeddings = pos_embeddings.to(device)
            neg_embeddings = neg_embeddings.to(device)
            neut_embeddings = neut_embeddings.to(device)
            assas_embeddings = assas_embeddings.to(device)
            
            optimizer.zero_grad()
            model_logits, search_logits, word_emb_output = model(pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)

            loss = loss_fn(model_logits, search_logits, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)

            train_logger_search.update_results(word_emb_output, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)
            # Currently looks at the positive output, use index 1 to look at negative output
            train_logger_model.update_results(model_logits[0], pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)
            
            loss.backward()
            optimizer.step()
            train_logger_search.update_loss(loss)
            train_logger_model.update_loss(loss)
    
        scheduler.step()
        losses_train.append(train_logger_search.avg_loss)
        train_logger_search.print_log()
        train_logger_model.print_log()

        # Validate model output
        validation_logger = validate(model, valid_dataloader, loss_fn, device)
        
        validation_logger.print_log()
        losses_valid.append(validation_logger.avg_loss)
        
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
    vocab_size = args.vocab

    use_model_output = utils.convert_args_str_to_bool(args.use_model_out)
    search_pruning = utils.convert_args_str_to_bool(args.prune_search)
    normalize_reward = utils.convert_args_str_to_bool(args.norm)

    print(f"Device: {device}")
    train_dataset = CodeNamesDataset(code_dir=code_data, game_dir=guess_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=4)
    print(f"Training Length: {len(train_dataset)}")
    valid_dataset = CodeNamesDataset(code_dir=code_data, game_dir=val_guess_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size=50, num_workers=4)

    vector_db = VectorSearch(train_dataset, prune=True)
    model = MORSpyDualHead(vector_db, device, vocab_size=vocab_size, search_pruning=search_pruning)
    model.to(device)

    losses_train, losses_valid = train(n_epochs=args.e, model=model, train_loader=train_dataloader, valid_dataloader=valid_dataloader, device=device, model_path=model_out, normalize_reward=normalize_reward, use_model_out=use_model_output)
    utils.save_loss_plot(losses_train, losses_valid, save_path=loss_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, help="Number of epochs", default=10)
    parser.add_argument('-b', type=int, help="Batch Size", default=400)
    parser.add_argument('-code_data', type=str, help="Codenames dataset path", default=BASE_DIR + "data/words.json")
    parser.add_argument('-guess_data', type=str, help="Geuss words dataset path", default=BASE_DIR + "data/codewords_full_w_assassin_valid.json")
    parser.add_argument('-vocab', type=int, default=80)
    parser.add_argument('-weight_decay', type=float, default=0.1)
    parser.add_argument('-prune_search', type=str, help="Prunes the search window based on average similarity [Y/n]", default='N')
    parser.add_argument('-use_model_out', type=str, help="Determines whether to use the model output for scoring or the search output (highest scoring word embedding) [Y/n]", default='N')
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-norm', type=str, help="Whether to normalize reward function, [Y/n]", default='Y')
    parser.add_argument('-val_guess_data', type=str, help="Filepath for the validation dataset", default=BASE_DIR + "data/codewords_full_w_assassin_mini.json")
    parser.add_argument('-model_out', type=str, default=BASE_DIR + "test.pth")
    parser.add_argument('-loss_out', type=str, default=BASE_DIR + "test.png")
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    args = parser.parse_args()
    main(args)