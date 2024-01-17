import torch
from torch.optim.lr_scheduler import ExponentialLR
from loss_fns.loss import CombinedTripletLoss, ScoringLoss
from torch.utils.data import DataLoader
from models.base_models import OldCodeGiver, CodeGiverRaw
from datasets.dataset import CodeGiverDataset
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import utils.utilities as utils
from utils.hidden_vars import BASE_DIR

def init_hyperparameters(model: OldCodeGiver, device):
    #loss_fn = CombinedTripletLoss(margin=0.8)
    #loss_fn = CATLoss(device, margin=0.8, weighting=2)
    #loss_fn = CATLossNormalDistribution(stddev=5.2, margin=0.01, device=device, constant=10)
    loss_fn = ScoringLoss(margin=0.6, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.5)
    return loss_fn, optimizer, scheduler

@torch.no_grad()
def validate(model: OldCodeGiver, valid_loader: DataLoader, loss_fn: CombinedTripletLoss, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_score = 0.0
    for i, data in enumerate(valid_loader, 0):
        pos_sents, neg_sents, neutral, pos_embeddings, neg_embeddings = data
        pos_embeddings, neg_embeddings = pos_embeddings.to(device), neg_embeddings.to(device)

        logits = model(pos_sents, neg_sents, neutral)
        loss, score = loss_fn(logits, pos_embeddings, neg_embeddings)
        total_loss += loss.item()
        total_score += score.item()
    model.train()
    avg_loss = total_loss / len(valid_loader)
    avg_score = total_score / len(valid_loader)

    return avg_loss, avg_score

def train(n_epochs: int, model: OldCodeGiver, train_loader: DataLoader, valid_dataloader: DataLoader, device: torch.device, model_path: str):
    loss_fn, optimizer, scheduler = init_hyperparameters(model, device)
    print("Training")
    model.train()

    losses_train = []
    losses_valid = []
    print(f"Starting training at: {datetime.datetime.now()}")
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")
        loss_train = 0.0
        total_score = 0.0
        for i, data in enumerate(train_loader, 0):
            if (i % 100 == 0):
                print(f"{datetime.datetime.now()}: Iteration: {i}/{len(train_loader)}")
            pos_sents, neg_sents, pos_embeddings, neg_embeddings = data
            # Put embeddings on device
            pos_embeddings = pos_embeddings.to(device)
            neg_embeddings = neg_embeddings.to(device)
            
            optimizer.zero_grad()
            logits = model(pos_sents, neg_sents)

            loss, score = loss_fn(logits, pos_embeddings, neg_embeddings)
            total_score += score.item()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        scheduler.step()
        avg_loss = loss_train / len(train_loader)
        avg_score = total_score / len(train_loader) # technically average of average scores
        losses_train.append(avg_loss)
        # Validate model output
        validation_loss, validation_score = validate(model, valid_dataloader, loss_fn, device)
        losses_valid.append(validation_loss)
        # Log and print save model parameters
        training_str = f"{datetime.datetime.now()}, Epoch: {epoch}\nTraining Loss: {avg_loss}, Training Score: {avg_score}\nValidation Loss: {validation_loss}, Validation Score: {validation_score}\n"
        print(training_str)
        if len(losses_train) == 1 or losses_train[-1] < losses_train[-2]:
                torch.save(model.state_dict(), model_path)
        
    return losses_train, losses_valid

def main(args):
    device = utils.get_device(args.cuda)
    code_data = args.code_data
    guess_data = args.guess_data

    val_guess_data = args.val_guess_data
    model_out = args.model_out
    loss_out = args.loss_out

    print(f"Device: {device}")
    train_dataset = CodeGiverDataset(code_dir=code_data, game_dir=guess_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=4)

    valid_dataset = CodeGiverDataset(code_dir=code_data, game_dir=val_guess_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size=50, num_workers=4)

    model = OldCodeGiver()
    model.to(device)

    losses_train, losses_valid = train(n_epochs=args.e, model=model, train_loader=train_dataloader, valid_dataloader=valid_dataloader, device=device, model_path=model_out)
    utils.save_loss_plot(losses_train, losses_valid, save_path=loss_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, help="Number of epochs", default=10)
    parser.add_argument('-b', type=int, help="Batch Size", default=400)
    parser.add_argument('-code_data', type=str, help="Codenames dataset path", default=BASE_DIR + "data/words.json")
    parser.add_argument('-guess_data', type=str, help="Geuss words dataset path", default=BASE_DIR + "data/five_word_data_medium.json")
    parser.add_argument('-val_guess_data', type=str, help="Filepath for the validation dataset", default=BASE_DIR + "data/five_word_data_validation.json")
    parser.add_argument('-model_out', type=str, default=BASE_DIR + "/saved_models/cat_model_10e_400b.pth")
    parser.add_argument('-loss_out', type=str, default=BASE_DIR + "saved_models/cat_model_10e_400b.png")
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    args = parser.parse_args()
    main(args)