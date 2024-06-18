import argparse
import torch
from models.many_to_many import MORSpyManyPooled, RetrievalTransformerNAR
from datasets.dataset import CodeNamesDataset
from torch.utils.data import DataLoader
import utils.utilities as utils
from utils.hidden_vars import BASE_DIR
from utils.vector_search import VectorSearch
from utils.logger import TrainLogger, EpochLoggerCombined
from loss_fns.loss import RerankerLoss
from torch.optim.lr_scheduler import ExponentialLR
from utils.hyperparameters import RerankerHyperParameter
import logging
import random


def init_hyperparameters(hp: RerankerHyperParameter, model: RetrievalTransformerNAR):
    loss_fn_rerank = RerankerLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=hp.gamma)
    return loss_fn_rerank, optimizer, scheduler

@torch.no_grad()
def validate(model: RetrievalTransformerNAR,
             valid_loader: DataLoader, 
             loss_fn: RerankerLoss, 
             device: torch.device) -> EpochLoggerCombined:
    val_logger = EpochLoggerCombined(
            len(valid_loader.dataset), 
            len(valid_loader), 
            name_model="Validation Model Reranker",
            name_search="Validation Clustered Search Embs",
            device=device
        )
    for data in valid_loader:
        pos_embs, neg_embs, neut_embs, assas_emb = data[1]

        pos_embs = pos_embs.to(device)
        neg_embs = neg_embs.to(device)
        neut_embs = neut_embs.to(device)
        assas_emb = assas_emb.to(device)

        out, labels, search_embs = model(pos_embs, neg_embs, neut_embs, assas_emb)
        search_embs = search_embs.mean(dim=1)

        loss = loss_fn(out, labels, search_embs)

        val_logger.update_results(
            model_out=out,
            search_out=search_embs,
            pos_emb=pos_embs,
            neg_emb=neg_embs,
            neut_emb=neut_embs,
            assas_emb=assas_emb
        )

        val_logger.update_loss(loss)
    return val_logger

def train(hprams: RerankerHyperParameter, 
          model: RetrievalTransformerNAR, 
          train_loader: DataLoader,
          valid_loader: DataLoader,
          console_logger: logging.Logger,
          device='cpu') -> TrainLogger:
    loss_fn, optimizer, scheduler = init_hyperparameters(hprams, model)
    
    
    model.train()

    train_logger = TrainLogger()
    console_logger.info("Started Training")
    for epoch in range(1, hprams.n_epochs + 1):
        console_logger.info(f"Epoch: {epoch}")

        epoch_logger = EpochLoggerCombined(
            len(train_loader.dataset), 
            len(train_loader), 
            name_model="Train Model Reranker",
            name_search="Train Clustered Search Embs",
            device=device
        )

        for i, data in enumerate(train_loader, 0):
            if ((i + 1) % 100 == 0):
                console_logger.info(f"Iteration: {i + 1}/{len(train_loader)}")
            pos_embs, neg_embs, neut_embs, assas_emb = data[1]

            # Put embeddings on device
            pos_embs = pos_embs.to(device)
            neg_embs = neg_embs.to(device)
            neut_embs = neut_embs.to(device)
            assas_emb = assas_emb.to(device)

            # Randomize board state
            if hprams.dynamic_board:
                pos_num = random.randint(1, pos_embs.shape[1])
                neg_num = random.randint(1, neg_embs.shape[1])
                neut_num = random.randint(1, neut_embs.shape[1])

                pos_embs = pos_embs[:, :pos_num, :]
                neg_embs = neg_embs[:, :neg_num, :]
                neut_embs = neut_embs[:, :neut_num, :]

            optimizer.zero_grad()
            
            out, labels, search_embs = model(pos_embs, neg_embs, neut_embs, assas_emb)
            search_embs = search_embs.mean(dim=1)

            loss = loss_fn(out, labels, search_embs)

            loss.backward()

            epoch_logger.update_results(
                model_out=out, 
                search_out=search_embs, 
                pos_emb=pos_embs, 
                neg_emb=neg_embs, 
                neut_emb=neut_embs, 
                assas_emb=assas_emb,
            )
            epoch_logger.update_loss(loss)

            optimizer.step()

        # Validate model output
        console_logger.info("Validating model: Note that validation set uses static board size, if training on dynamic board, results may differ significantly")
        valid_logger = validate(model, valid_loader, loss_fn, device)
        train_logger.add_loggers(epoch_logger, valid_logger)

        scheduler.step()

        print(f"========================================================================================")
        epoch_logger.print_log()
        print(f"========================================================================================")
        valid_logger.print_log()

    return train_logger
            

def main(args):

    hprams = RerankerHyperParameter(args)

    encoder_dir = args.enc
    output_dir = args.out
    device = utils.get_device(args.cuda)

    board_dir = args.b_dir
    board_dir_valid = args.b_dir_valid
    vocab_dir = args.v_dir

    console_logger = utils.console_logger(logger_name="reranker_log")
    console_logger.info(f"Device: {device}")

    train_dataset = CodeNamesDataset(code_dir=vocab_dir, game_dir=board_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=hprams.batch_size, shuffle=True)

    valid_dataset = CodeNamesDataset(code_dir=vocab_dir, game_dir=board_dir_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=10, shuffle=False)

    console_logger.info("Indexing Search Graph")
    vocab = VectorSearch(train_dataset, prune=True)

    encoder = MORSpyManyPooled(num_heads=args.num_heads)
    encoder.load_state_dict(torch.load(f"{encoder_dir}model.pth"))
    encoder.to(device)

    model = RetrievalTransformerNAR(
        vocab=vocab,
        encoder=encoder,
        head_num=args.num_heads,
        neg_weight=hprams.neg_weight,
        neut_weight=hprams.neut_weight,
        assas_weight=hprams.assas_weight,
        device=device,
        freeze_encoder=True,
    )
    model.to(device)

    results = train(
        hprams=hprams,
        model=model,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        console_logger=console_logger,
        device=device
    )

    results.save_results(output_dir)
    model_path = f"{output_dir}model.pth"
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-enc', type=str, help="Encoder directory", default=f"{BASE_DIR}model_data/encoder_portfolio/")
    parser.add_argument('-cuda', type=str, help="Use CUDA [Y/n]", default='Y')
    parser.add_argument('-out', type=str, help="Output directory path", default=f"{BASE_DIR}model_data/testing/")

    # Training parameters
    parser.add_argument('-e', type=int, help="Number of epochs", default=5)
    parser.add_argument('-b', type=int, help="Batch Size", default=200)

    # Dataset directories
    parser.add_argument('-v_dir', type=str, help="Vocab Directory", default=f"{BASE_DIR}data/words.json")
    parser.add_argument('-b_dir', type=str, help="Train Board Directory", default=f"{BASE_DIR}data/codewords_full_w_assassin_valid.json")
    parser.add_argument('-b_dir_valid', type=str, help="Validation Board Directory", default=f"{BASE_DIR}data/codewords_full_w_assassin_mini.json")

    # Reward Weighting
    parser.add_argument('-neut_weight', type=float, default=2.0)
    parser.add_argument('-neg_weight', type=float, default=0.0)
    parser.add_argument('-assas_weight', type=float, default=-10.0)

    # Hyperparameters
    parser.add_argument('-lr', type=float, help="Learning Rate", default=0.0001)
    parser.add_argument('-gamma', type=float, help="Gamma", default=0.9)
    parser.add_argument('-w_decay', type=float, help="Weight Decay", default=0.1)
    parser.add_argument('-sw', type=int, help="Search Window", default=80)
    parser.add_argument('-num_heads', type=int, help="Number of encoder heads", default=3)
    parser.add_argument('-dynamic_board', type=str, help="Randomize Board States: [Y/n]", default='n')
    args = parser.parse_args()
    main(args)
