from models.multi_objective_models import MORSpyMaster, MORSpyMPNet, MORSpyMiniLM, MORSpyIntegratedRewards
from datasets.dataset import CodeNamesDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils.vector_search import VectorSearch
import utils.utilities as utils
from utils.logger import TestLogger
from utils.hidden_vars import BASE_DIR

def calc_cos_score(anchor, pos_encs, neg_encs):
    anchor = anchor.unsqueeze(1)
    pos_score = F.cosine_similarity(anchor, pos_encs, dim=2)
    neg_score = F.cosine_similarity(anchor, neg_encs, dim=2)
    
    return pos_score, neg_score

def process_shape( pos_tensor: torch.Tensor, neg_tensor: torch.Tensor):
        pos_dim = pos_tensor.shape[1]
        neg_dim = neg_tensor.shape[1]
        dif = 0
        if pos_dim > neg_dim:
            dif = pos_dim - neg_dim
            pos_tensor = pos_tensor[:, dif:]
        elif neg_dim > pos_dim:
            dif = neg_dim - pos_dim
            neg_tensor = neg_tensor[:, dif:]
            dif = 0 # set difference back to zero to avoid it being counted in the score function
        
        return pos_tensor, neg_tensor, dif

def print_results(sentences: tuple, words, tot_score: float, neg_score, neut_score: float, assas_score: float):
    pos_sents, neg_sents, neutral_sents, assas_word = sentences

    num_sents = len(pos_sents)
    print(f"Negative Score: {neg_score}/{num_sents}")
    print(f"Neutral Score: {neut_score}/{num_sents}")
    print(f"Assassin Score: {assas_score}/{num_sents}")
    print(f"Total Score: {tot_score}/9")
    print()



@torch.no_grad()
def test_loop(model: MORSpyMaster, dataloader: DataLoader, device: torch.device, verbose=False, use_model_out=False, pos_split=9, neg_split=9, neut_split=6):

    model.eval()
    logger_model = TestLogger(len(dataloader.dataset), len(dataloader), device=device, name="Testing Model Output")
    logger_search = TestLogger(len(dataloader.dataset), len(dataloader), device=device, name="Testing Search Output")

    model.update_weights(0.0, 2.0, -10.0)
    print(f"Pos Split: {pos_split}\nNeg Split: {neg_split}\nNeut Split: {neut_split}\n")
    for i, data in enumerate(dataloader):
        pos_sents, neg_sents, neutral_sents, assasin_word = data[0]
        pos_embs, neg_embs, neut_embs, assas_embs = data[1]

        pos_embs = pos_embs.to(device)
        neg_embs = neg_embs.to(device)
        neut_embs = neut_embs.to(device)
        assas_embs = assas_embs.to(device)

        pos_embs = pos_embs[:, :pos_split]
        neg_embs = neg_embs[:, :neg_split]
        neut_embs = neut_embs[:, :neut_split]
        

        words, search_out_index, model_out, search_out = model(pos_embs, neg_embs, neut_embs, assas_embs)

        logger_model.update_results(words, model_out, pos_embs, neg_embs, neut_embs, assas_embs)
        logger_search.update_results(words, search_out, pos_embs, neg_embs, neut_embs, assas_embs)
        
    logger_model.print_log()
    logger_search.print_log()

    #logger_search.print_word_distribution()


        
def main(args):
    device = utils.get_device(args.cuda)
    verbose = True if args.v.lower() == 'y' else False
    use_model_out = True if args.use_model_out.lower() == 'y' else False

    # Initialize data
    test_dataset = CodeNamesDataset(code_dir=args.board_dir, game_dir=args.vocab_dir)
    dataloader = DataLoader(test_dataset, batch_size=args.b, shuffle=False)
    vector_db = VectorSearch(test_dataset, prune=True)

    # Initialize model
    # Initialize model
    backbone_name = args.backbone
    if (backbone_name == "all-mpnet-base-v2"):
        model = MORSpyMPNet(vector_db, device, vocab_size=args.sw)
    elif (backbone_name == "all-MiniLM-L6-v2"):
        model = MORSpyMiniLM(vector_db, device, vocab_size=args.sw)
    elif (backbone_name == "integrated"):
        model = MORSpyIntegratedRewards(vector_db, device, vocab_size=args.sw)
    else:
        model = MORSpyMPNet(vector_db, device, vocab_size=args.sw)

    model.load_state_dict(torch.load(args.m))
    model.to(device)
    model.eval()

    test_loop(model, dataloader, device, verbose, use_model_out, pos_split=args.pos_split, neg_split=args.neg_split, neut_split=args.neut_split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-board_dir', type=str, help='Dataset Path', default=BASE_DIR + "data/words_extended.json")
    parser.add_argument('-vocab_dir', type=str, help="", default=BASE_DIR + "data/codewords_full_w_assassin_mini.json")
    parser.add_argument('-m', type=str, help='Model Path', default=BASE_DIR + "model_data/mpnet_dynamic_board/model.pth")
    parser.add_argument('-use_model_out', type=str, help="Whether to use model output or search output, use Y or N", default='N')
    parser.add_argument('-sw', type=int, help='Model search window', default=80)
    parser.add_argument('-b', type=int, help='Batch Size', default=200)
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    parser.add_argument('-pos_split', type=int, default=None)
    parser.add_argument('-neg_split', type=int, default=None)
    parser.add_argument('-neut_split', type=int, default=None)
    parser.add_argument('-v', type=str, help="Verbose [y/N]", default='Y')
    parser.add_argument('-backbone', type=str, help="Model backbone", default='integrated')
    args = parser.parse_args()
    main(args)