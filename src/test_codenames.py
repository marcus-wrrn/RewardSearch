from models.multi_objective_models import MORSpyMaster
from datasets.dataset import CodeNamesDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils.vector_search import VectorSearch
import utils.utilities as utils

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
    # for i in range(num_sents):
    #     print(f"Target Words: {pos_sents[i]}")
    #     print(f"Negative Words: {neg_sents[i]}")
    #     print(f"Neutral Words: {neutral_sents[i]}")
    #     print(f"Assassin Words: {assas_word[i]}")
    #     print(f"\n{words[i][0]}")
    #     print()
    

@torch.no_grad()
def test_loop(model: MORSpyMaster, dataloader, device: torch.device, verbose=False, use_model_out=False):

    total_score = 0
    num_negative = 0
    num_neutral = 0
    num_assassin = 0

    model.eval()
    for i, data in enumerate(dataloader):
        pos_sents, neg_sents, neutral_sents, assasin_word = data[0]
        pos_embs, neg_embs, neut_embs, assas_embs = data[1]

        pos_embs = pos_embs.to(device)
        neg_embs = neg_embs.to(device)
        neut_embs = neut_embs.to(device)
        assas_embs = assas_embs.to(device)

        model_obj = model(pos_embs, neg_embs, neut_embs, assas_embs)
        
        # pos_score, neg_score = calc_cos_score(model_obj.search_out_max, pos_embs, neg_embs)

        # Max search output
        num_correct, neg_sum, neut_sum, assas_sum = utils.calc_codenames_score(model_obj.search_out if not use_model_out else model_obj.model_out, pos_embs, neg_embs, neut_embs, assas_embs, device)

        
        if verbose: 
            print_results(data[0], model_obj.words, num_correct, neg_sum, neut_sum, assas_sum)
            #print(f"Score: {num_correct}/{len(pos_embs[0])}")
        total_score += num_correct
        num_negative += neg_sum
        num_neutral += neut_sum
        num_assassin += assas_sum

    print(f"Average Score: {total_score/len(dataloader)}")
    print(f"Num Neutral: {num_neutral}")
    print(f"Num Negative: {num_negative}")
    print(f"Num Assassin: {num_assassin}")
        
def main(args):
    device = utils.get_device(args.cuda)
    verbose = True if args.v.lower() == 'y' else False
    use_model_out = True if args.use_model_out.lower() == 'y' else False

    # Initialize data
    test_dataset = CodeNamesDataset(code_dir=args.code_dir, game_dir=args.guess_dir)
    dataloader = DataLoader(test_dataset, batch_size=args.b, shuffle=False)
    vector_db = VectorSearch(test_dataset, prune=True)

    # Initialize model
    model = MORSpyMaster(vector_db, device, vocab_size=args.sw)
    model.load_state_dict(torch.load(args.m))
    model.to(device)
    model.eval()

    test_loop(model, dataloader, device, verbose, use_model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-code_dir', type=str, help='Dataset Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json")
    parser.add_argument('-guess_dir', type=str, help="", default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/codewords_full_w_assassin_mini.json")
    parser.add_argument('-m', type=str, help='Model Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/codenames.pth")
    parser.add_argument('-use_model_out', type=str, help="Whether to use model output or search output, use Y or N", default='N')
    parser.add_argument('-sw', type=int, help='Model search window', default=2)
    parser.add_argument('-b', type=int, help='Batch Size', default=200)
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    parser.add_argument('-v', type=str, help="Verbose [y/N]", default='Y')
    args = parser.parse_args()
    main(args)