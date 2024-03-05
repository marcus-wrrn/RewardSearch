import torch
from torch.optim.lr_scheduler import ExponentialLR
from loss_fns.loss import CombinedTripletLoss, TripletMeanLossL2Distance
from torch.utils.data import DataLoader
from models.base_models import OldCodeGiver, SentenceEncoderRaw
from datasets.dataset import CodeGiverDatasetCombinedSent
import datetime
import argparse
import utils.utilities as utils

def init_hyperparameters(model: OldCodeGiver):
    loss_fn = CombinedTripletLoss(margin=0.8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    return loss_fn, optimizer, scheduler

@torch.no_grad()
def validate(model: OldCodeGiver, valid_loader: DataLoader, loss_fn: CombinedTripletLoss, device: torch.device):
    model.eval()
    total_loss = 0.0
    for i, data in enumerate(valid_loader, 0):
        input_sents, pos_embeddings, neg_embeddings = data
        pos_embeddings, neg_embeddings = pos_embeddings.to(device), neg_embeddings.to(device)

        logits = model(input_sents)
        loss = loss_fn(logits, pos_embeddings, neg_embeddings)
        total_loss += loss.item()
    model.train()
    return total_loss / len(valid_loader)

def train(n_epochs: int, model: SentenceEncoderRaw, train_loader: DataLoader, valid_dataloader: DataLoader, device: torch.device, model_path: str):
    loss_fn, optimizer, scheduler = init_hyperparameters(model)
    print("Training")
    model.train()

    losses_train = []
    losses_valid = []
    print(f"Starting training at: {datetime.datetime.now()}")
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")
        loss_train = 0.0
        for i, data in enumerate(train_loader, 0):
            if (i % 100 == 0):
                print(f"{datetime.datetime.now()}: Iteration: {i}/{len(train_loader)}")
            input_sents, pos_embeddings, neg_embeddings = data
            # Put embeddings on device
            pos_embeddings = pos_embeddings.to(device)
            neg_embeddings = neg_embeddings.to(device)
            
            optimizer.zero_grad()
            logits = model(input_sents)

            loss = loss_fn(logits, pos_embeddings, neg_embeddings)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        scheduler.step()
        avg_loss = loss_train / len(train_loader)
        losses_train.append(avg_loss)
        # Validate model output
        validation_loss = validate(model, valid_dataloader, loss_fn, device)
        losses_valid.append(validation_loss)
        # Log and print save model parameters
        training_str = f"{datetime.datetime.now()}, Epoch: {epoch}, Training Loss: {avg_loss}, Validation Loss: {validation_loss}"
        print(training_str)
        if len(losses_train) == 1 or losses_train[-1] < losses_train[-2]:
                torch.save(model.state_dict(), model_path)
        
    return losses_train, losses_valid

def main(args):
    device = utils.get_device(args.cuda)
    code_data = args.code_data
    guess_data = args.guess_data
    val_guess_data = args.val_guess_data
    model_out = args.out
    loss_out = args.loss_out

    print(f"Device: {device}")
    train_dataset = CodeGiverDatasetCombinedSent(code_dir=code_data, game_dir=guess_data)
    train_dataloader = DataLoader(train_dataset, batch_size=300, num_workers=4)

    valid_dataset = CodeGiverDatasetCombinedSent(code_dir=code_data, game_dir=val_guess_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size=50, num_workers=4)

    model = SentenceEncoderRaw(device=device)
    model.to(device)

    losses_train, losses_valid = train(n_epochs=10, model=model, train_loader=train_dataloader, valid_dataloader=valid_dataloader, device=device, model_path=model_out)
    utils.save_loss_plot(losses_train, losses_valid, save_path=loss_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=300)
    parser.add_argument('-e', type=int, default=10)
    parser.add_argument('-code_data', type=str, help="Codenames dataset path", default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json")
    parser.add_argument('-guess_data', type=str, help="Geuss words dataset path", default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/three_word_data_medium.json")
    parser.add_argument('-val_guess_data', type=str, help="Filepath for the validation dataset", default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/three_word_data.json")
    parser.add_argument('-out', type=str, default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/saved_models/model.pth")
    parser.add_argument('-loss_out', type=str, default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/saved_models/model_loss.png")
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    args = parser.parse_args()
    main(args)