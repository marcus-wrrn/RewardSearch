import torch
import torch.nn as nn
from loss_fns.loss import CATLoss
from torch.optim.lr_scheduler import ExponentialLR
from models.base_models import OldCodeGiver
from torch.utils.data import DataLoader
import datetime

def init_hyperparameters(model, device):
    loss_fn = CATLoss(device, margin=0.8, weighting=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    return loss_fn, optimizer, scheduler

class HyperParams:
    def __init__(self, loss_fn, optimizer, scheduler):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
    

class TrainManager():
    def __init__(self,  model: OldCodeGiver,  
                 model_name: str,
                 out_path: str,
                 device="cuda"):
        self.model = model
        self.name = model_name
        self.out_path = out_path
        self.device = device

    def fit(self, n_epochs: int, train_loader: DataLoader, val_loader: DataLoader, hparams: HyperParams, verbose=True):
        loss_fn, optimizer, scheduler = hparams.loss_fn, hparams.optimizer, hparams.scheduler

        
        self.model.train()
        device = self.device

        losses_train = []
        losses_valid = []
        
        if verbose:
            print(f"Device: {self.device}")
            print(f"Starting Training at: {datetime.datetime.now()}")
        
        for epoch in range(1, n_epochs + 1):
            if verbose: print(f"Epoch: {epoch}")
            loss_train = 0.0
            for i, data in enumerate(train_loader, 0):
                if (i % 100 == 0):
                    print(f"{datetime.datetime.now()}: Iteration: {i}/{len(train_loader)}")
                pos_sents, neg_sents, pos_embeddings, neg_embeddings = data
                # Put embeddings on device
                pos_embeddings = pos_embeddings.to(self.device)
                neg_embeddings = neg_embeddings.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(pos_sents, neg_sents)

                loss = loss_fn(logits, pos_embeddings, neg_embeddings)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                
            scheduler.step()
            avg_loss = loss_train / len(train_loader)
            self.losses_train.append(avg_loss)
            # Validate model output
            validation_loss = validate(self.model, val_loader, loss_fn, device)
            self.losses_valid.append(validation_loss)
            # Log and print save model parameters
            training_str = f"{datetime.datetime.now()}, Epoch: {epoch}, Training Loss: {avg_loss}, Validation Loss: {validation_loss}"
            print(training_str)
            if len(losses_train) == 1 or losses_train[-1] < losses_train[-2]:
                    torch.save(model.state_dict(), model_path)


            
        
    
