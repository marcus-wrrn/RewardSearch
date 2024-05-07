import argparse
import torch
from models.many_to_many import MORSpyManyPooled
from models.reranker import Reranker
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


def init_hyperparameters(hp: RerankerHyperParameter, model: Reranker):
    loss_fn_rerank = RerankerLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=hp.gamma)
    return loss_fn_rerank, optimizer, scheduler

@torch.no_grad()
def validate(encoder: MORSpyManyPooled, 
             reranker: Reranker, 
             valid_loader: DataLoader, 
             loss_fn: RerankerLoss, 
             device: torch.device) -> EpochLoggerCombined:
    val_logger = EpochLoggerCombined(
            len(valid_loader.dataset), 
            len(valid_loader), 
            name_model="Validation Model Reranker",
            name_search="Validation Algorithmic Reranker",
            device=device
        )
    for data in valid_loader:
        pos_embs, neg_embs, neut_embs, assas_emb = data[1]

        pos_embs = pos_embs.to(device)
        neg_embs = neg_embs.to(device)
        neut_embs = neut_embs.to(device)
        assas_emb = assas_emb.to(device)

        encoder_logits, encoder_heads = encoder(pos_embs, neg_embs, neut_embs, assas_emb)

        out = reranker(encoder_heads, encoder_logits.word_embs)

        target_idx = encoder_logits.emb_ids
            
        # Build target tensor
        target_tensor = torch.zeros(target_idx.shape[0], out.shape[1], dtype=torch.float32, device=device)
        target_tensor.scatter_(1, target_idx.unsqueeze(1), 1)

        loss = loss_fn(out, target_tensor)

        # Find most similar output
        word_indices = torch.max(out, dim=1).indices

        # No need to expand for every embedding dimension if gathering across num_outputs
        word_indices = word_indices.unsqueeze(1).expand(-1, 768)  # Prepare for gather

        # Gather embeddings
        word_embeddings = torch.gather(encoder_logits.word_embs, 1, word_indices.unsqueeze(1)).squeeze(1)

        val_logger.update_results(
            model_out=word_embeddings,
            search_out=encoder_logits.h_score_emb,
            pos_emb=pos_embs,
            neg_emb=neg_embs,
            neut_emb=neut_embs,
            assas_emb=assas_emb
        )

        val_logger.update_loss(loss)
    return val_logger

def train(hprams: RerankerHyperParameter, 
          encoder: MORSpyManyPooled, 
          reranker: Reranker, 
          train_loader: DataLoader,
          valid_loader: DataLoader,
          console_logger: logging.Logger,
          device='cpu') -> TrainLogger:
    loss_fn, optimizer, scheduler = init_hyperparameters(hprams, reranker)
    
    
    reranker.train()
    encoder.eval()

    train_logger = TrainLogger()
    console_logger.info("Started Training")
    for epoch in range(1, hprams.n_epochs + 1):
        console_logger.info(f"Epoch: {epoch}")

        epoch_logger = EpochLoggerCombined(
            len(train_loader.dataset), 
            len(train_loader), 
            name_model="Train Model Reranker",
            name_search="Train Algorithmic Reranker",
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
            
            with torch.no_grad():
                encoder_logits, encoder_heads = encoder(pos_embs, neg_embs, neut_embs, assas_emb)

            out = reranker(encoder_heads, encoder_logits.word_embs)
           

            target_idx = encoder_logits.emb_ids
            
            # Build target tensor
            target_tensor = torch.zeros(target_idx.shape[0], out.shape[1], dtype=torch.float32, device=device)
            target_tensor.scatter_(1, target_idx.unsqueeze(1), 1)

            loss = loss_fn(out, target_tensor)

            loss.backward()

            # Find most similar output
            word_indices = torch.max(out, dim=1).indices

            # No need to expand for every embedding dimension if gathering across num_outputs
            word_indices = word_indices.unsqueeze(1).expand(-1, 768)  # Prepare for gather

            # Gather embeddings
            word_embeddings = torch.gather(encoder_logits.word_embs, 1, word_indices.unsqueeze(1)).squeeze(1)

            epoch_logger.update_results(
                model_out=word_embeddings, 
                search_out=encoder_logits.h_score_emb, 
                pos_emb=pos_embs, 
                neg_emb=neg_embs, 
                neut_emb=neut_embs, 
                assas_emb=assas_emb,
            )
            epoch_logger.update_loss(loss)

            optimizer.step()

        # Validate model output
        console_logger.info("Validating model: Note that validation set uses static board size, if training on dynamic board, results may differ significantly")
        valid_logger = validate(encoder, reranker, valid_loader, loss_fn, device)
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
    encoder = MORSpyManyPooled(vocab, device, vocab_size=80)
    reranker = Reranker(vocab_size=80, head_num=3)

    encoder.load_state_dict(torch.load(f"{encoder_dir}model.pth"))
    encoder.to(device)
    
    reranker.to(device)

    results = train(
        hprams=hprams,
        encoder=encoder,
        reranker=reranker,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        console_logger=console_logger,
        device=device
    )

    results.save_results(output_dir)
    model_path = f"{output_dir}model.pth"
    torch.save(reranker.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-enc', type=str, help="Encoder directory", default=f"{BASE_DIR}model_data/many_test_two_no_spacing_loss/")
    parser.add_argument('-cuda', type=str, help="Use CUDA [Y/n]", default='Y')
    parser.add_argument('-out', type=str, help="Output directory path", default=f"{BASE_DIR}model_data/testing/")

    # Training parameters
    parser.add_argument('-e', type=int, help="Number of epochs", default=5)
    parser.add_argument('-b', type=int, help="Batch Size", default=200)

    # Dataset directories
    parser.add_argument('-v_dir', type=str, help="Vocab Directory", default=f"{BASE_DIR}data/words.json")
    parser.add_argument('-b_dir', type=str, help="Train Board Directory", default=f"{BASE_DIR}data/codewords_full_w_assassin_valid.json")
    parser.add_argument('-b_dir_valid', type=str, help="Validation Board Directory", default=f"{BASE_DIR}data/codewords_full_w_assassin_mini.json")

    # Model Directories
    parser.add_argument('-save_dir', type=str, help="Directory to save model path and training info", default=f"{BASE_DIR}model_data/testing/")

    # Hyperparameters
    parser.add_argument('-lr', type=float, help="Learning Rate", default=0.0001)
    parser.add_argument('-gamma', type=float, help="Gamma", default=0.9)
    parser.add_argument('-w_decay', type=float, help="Weight Decay", default=0.1)
    parser.add_argument('-sw', type=int, help="Search Window", default=80)
    parser.add_argument('-dynamic_board', type=str, help="Randomize Board States: [Y/n]", default='n')
    args = parser.parse_args()
    main(args)
