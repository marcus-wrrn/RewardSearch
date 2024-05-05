import argparse
import torch
from models.many_to_many import MORSpyManyPooled
from models.reranker import Reranker
from datasets.dataset import CodeNamesDataset
from torch.utils.data import DataLoader
import utils.utilities as utils
from utils.hidden_vars import BASE_DIR
from utils.vector_search import VectorSearch
from utils.logger import TrainLogger
from loss_fns.loss import RerankerLoss
from torch.optim.lr_scheduler import ExponentialLR
from utils.hyperparameters import RerankerHyperParameter

import logging
import random


def init_hyperparameters(hp: RerankerHyperParameter, model: Reranker):
    loss_fn = RerankerLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=hp.gamma)
    return loss_fn, optimizer, scheduler



class ManyOutData:
    def __init__(self) -> None:
        self.words = []
        self.embeddings = [] 

def train(hprams: RerankerHyperParameter, 
          encoder: MORSpyManyPooled, 
          reranker: Reranker, 
          train_loader: DataLoader,
          valid_loader: DataLoader,
          console_logger: logging.Logger,
          device='cpu') -> TrainLogger:
    loss_fn, optimizer, scheduler = init_hyperparameters(hprams, reranker, device)
    
    
    reranker.train()
    encoder.eval()

    train_logger = TrainLogger()
    console_logger.info("Started Training")
    for epoch in range(1, hprams.n_epochs + 1):
        console_logger.info(f"Epoch: {epoch}")

        for i, data in enumerate(train_loader, 0):
            if ((i + 1) % 100 == 0):
                console_logger.info(f"Iteration: {i+1}/{len(train_loader)}")
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




def main(args):

    hprams = RerankerHyperParameter(args)

    model_dir = args.enc
    output_dir = args.out
    device = utils.get_device(args.cuda)
    model_path = f"{model_dir}model.pth"
    board_dir = args.b_dir
    vocab_dir = args.v_dir

    console_logger = logging.getLogger('reranker_logger')
    console_logger.setLevel(logging.DEBUG)
    
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    console_logger.addHandler(stream_handler)
    
    
    console_logger.info(f"Device: {device}")
    

    dataset = CodeNamesDataset(code_dir=board_dir, game_dir=vocab_dir)
    vocab = VectorSearch(dataset, prune=True)
    encoder = MORSpyManyPooled(vocab, device, vocab_size=80)
    reranker = Reranker(vocab_size=80, head_num=3)

    encoder.load_state_dict(torch.load(model_path))
    encoder.to(device)
    reranker.to(device)

    train(hprams, encoder, reranker, dataset, console_logger, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-enc', type=str, help="Encoder directory", default=f"{BASE_DIR}model_data/many_test_two_no_spacing_loss/")
    parser.add_argument('-cuda', type=str, help="Use CUDA [Y/n]", default='Y')
    parser.add_argument('-b_dir', type=str, help="Board Directory", default=f"{BASE_DIR}data/words.json")
    parser.add_argument('-v_dir', type=str, help="Vocab Directory", default=f"{BASE_DIR}data/codewords_full_w_assassin_mini.json")
    parser.add_argument('-out', type=str, help="Output path", default=f"{BASE_DIR}data/many_head_output.json")

    # Hyperparameters
    parser.add_argument('-lr', type=float, help="Learning Rate", default=0.00001)
    parser.add_argument('-gamma', type=float, help="Gamma", default=0.9)
    parser.add_argument('-w_decay', type=float, help="Weight Decay", default=0.1)
    parser.add_argument('-sw', type=int, help="Search Window", default=80)
    parser.add_argument('-dynamic_board', type=str, help="Randomize Board States: [Y/n]", default='n')
    args = parser.parse_args()
    main(args)
