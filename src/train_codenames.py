import torch
from torch.optim.lr_scheduler import ExponentialLR
from loss_fns.loss import  RewardSearchLoss
from torch.utils.data import DataLoader
from models.multi_objective_models import MORSpyMaster
from datasets.dataset import CodeNamesDataset, SentenceNamesDataset
import datetime
import argparse
import utils.utilities as utils
from utils.vector_search import VectorSearch
from utils.hidden_vars import BASE_DIR
import utils.utilities as utils
from utils.logger import EpochLogger, TrainLogger

# This is the main Codenames Model with the best recorded performance -> main training script

def init_hyperparameters(hp: utils.HyperParameters, model: MORSpyMaster, device, normalize_reward):
    loss_fn = RewardSearchLoss(model_marg=hp.model_marg, search_marg=hp.search_marg, device=device, normalize=normalize_reward)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=hp.gamma)
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
def validate(model: MORSpyMaster, valid_loader: DataLoader, loss_fn: RewardSearchLoss, device: torch.device):
    val_logger_search = EpochLogger(len(valid_loader.dataset), len(valid_loader), device=device, name="Validation Search")
    val_logger_model = EpochLogger(len(valid_loader.dataset), len(valid_loader), device=device, name="Validation Model")

    for i, data in enumerate(valid_loader, 0):
        pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings = data[1]
        pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings = pos_embeddings.to(device), neg_embeddings.to(device), neut_embeddings.to(device), assas_embeddings.to(device)

        model_out, search_out, search_out_max, search_out_min = model(pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)

        #loss = F.triplet_margin_loss(model_out, search_min, search_max, margin=0.2)
        loss = loss_fn(model_out, search_out_max, search_out_min, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)
        #score, results, neut_sum, assas_sum = utils.calc_codenames_score(search_out, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings, device)

        
        val_logger_search.update_results(search_out, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)
        val_logger_search.update_loss(loss)

        val_logger_model.update_results(model_out, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)
        val_logger_model.update_loss(loss)
        

    return val_logger_model, val_logger_search

def train(hyperparams: utils.HyperParameters, model: MORSpyMaster, train_loader: DataLoader, valid_loader: DataLoader, device: torch.device, normalize_reward: bool) -> TrainLogger:

    loss_fn, optimizer, scheduler = init_hyperparameters(hyperparams, model, device, normalize_reward)
    print("Training")
    model.train()

    train_logger = TrainLogger(hyperparams.n_epochs)

    print(f"Starting training at: {datetime.datetime.now()}")
    for epoch in range(1, hyperparams.n_epochs + 1):
        print(f"Epoch: {epoch}")
        train_logger_search = EpochLogger(len(train_loader.dataset), len(train_loader), device=device, name="Training Search")
        train_logger_model = EpochLogger(len(train_loader.dataset), len(train_loader), device=device, name="Training Model")
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
            model_out, search_out, search_out_max, search_out_min = model(pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)

            loss = loss_fn(model_out, search_out_max, search_out_min, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)
            
            train_logger_model.update_results(model_out, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)
            train_logger_search.update_results(search_out, pos_embeddings, neg_embeddings, neut_embeddings, assas_embeddings)


            train_logger_model.update_loss(loss)
            train_logger_search.update_loss(loss)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Validate model output
        valid_logger_model, valid_logger_search = validate(model, valid_loader, loss_fn, device)
        train_logger.add_loggers(train_logger_model, train_logger_search, valid_logger_model, valid_logger_search)

        # TODO: Implement this in the train logger object
        print()
        train_logger_model.print_log()
        train_logger_search.print_log()
        
        valid_logger_model.print_log()
        valid_logger_search.print_log()
    
    return train_logger
        

def main(args):
    device = utils.get_device(args.cuda)
    # extract cli arguments
    code_data = args.code_data
    guess_data = args.guess_data
    val_guess_data = args.val_guess_data

    hyperparams = utils.HyperParameters(args)

    normalize_reward = utils.convert_args_str_to_bool(args.norm)

    print(f"Device: {device}")
    if hyperparams.using_sentences:
        train_dataset = SentenceNamesDataset(code_dir=code_data, game_dir=guess_data, vocab_dir=args.vocab_dir)
        valid_dataset = SentenceNamesDataset(code_dir=code_data, game_dir=val_guess_data)
    else:
        train_dataset = CodeNamesDataset(code_dir=code_data, game_dir=guess_data)
        valid_dataset = CodeNamesDataset(code_dir=code_data, game_dir=val_guess_data)
        
    train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=50, num_workers=4)

    print(f"Training Length: {len(train_dataset)}")
    vector_db = VectorSearch(train_dataset, prune=True)
    model = MORSpyMaster(vector_db, device, neutral_weight=hyperparams.neut_weight, negative_weight=hyperparams.neg_weight, assas_weights=hyperparams.assas_weight, vocab_size=hyperparams.vocab_size)
    model.to(device)

    logger = train(hyperparams=hyperparams, model=model, train_loader=train_dataloader, valid_loader=valid_dataloader, device=device, normalize_reward=normalize_reward)
    
    # Save log results
    logger.save_results(args.dir)

    # Save model information
    model_path = args.dir + args.name + ".pth"
    torch.save(model.state_dict(), model_path)

    # Save hyperparameters
    hp_path = args.dir + "hyperparameters.json"
    hyperparams.save_params(hp_path)

if __name__ == "__main__":
    # Most default values can be kept the same, but can be changed if needed
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, help="Number of epochs", default=10)
    parser.add_argument('-b', type=int, help="Batch Size", default=400)
    parser.add_argument('-code_data', type=str, help="Codenames dataset path", default=BASE_DIR + "data/words.json")
    parser.add_argument('-guess_data', type=str, help="Geuss words dataset path", default=BASE_DIR + "data/codewords_full_w_assassin_valid.json")
    parser.add_argument('-vocab_dir', type=str, help="Vocab directory for sentences dataset", default=BASE_DIR + "data/news_vocab.json")
    parser.add_argument('-vocab', type=int, default=80)
    parser.add_argument('-w_decay', type=float, default=0.1)
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-m_marg', type=float, default=0.7)
    parser.add_argument('-s_marg', type=float, default=0.8)
    parser.add_argument('-lr', type=float, default=0.00001)

    parser.add_argument('-prune_search', type=str, help="Prunes the search window based on average similarity [Y/n]", default='N')
    parser.add_argument('-use_model_out', type=str, help="Determines whether to use the model output for scoring or the search output (highest scoring word embedding) [Y/n]", default='N')
    parser.add_argument('-sentences', type=str, help="Whether the model is being trained on longer texts [Y/n]", default='N')
    
    parser.add_argument('-neut_weight', type=float, default=1.0)
    parser.add_argument('-neg_weight', type=float, default=0.0)
    parser.add_argument('-assas_weight', type=float, default=-10.0)
    parser.add_argument('-norm', type=str, help="Whether to normalize reward function, [Y/n]", default='Y')
    parser.add_argument('-val_guess_data', type=str, help="Filepath for the validation dataset", default=BASE_DIR + "data/codewords_full_w_assassin_mini.json")
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    parser.add_argument('-dir', type=str, help="Directory to save all results of the model", default=BASE_DIR + "model_data/testing/")
    parser.add_argument('-name', type=str, help="Name of Model", default="test")
    args = parser.parse_args()
    main(args)