from models.base_models import OldCodeGiver, SentenceEncoderRaw, CodeGiverRaw, CodeSearchDualNet
from datasets.dataset import CodeGiverDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils.vector_search import VectorSearch
import utils.utilities as utils

def calc_cos_score(anchor, pos_encs, neg_encs):
    pos_score = F.cosine_similarity(anchor, pos_encs, dim=1)
    neg_score = F.cosine_similarity(anchor, neg_encs, dim=1)
    
    return pos_score, neg_score

def get_anchor(word_list, embedding_list, device):
    ...

@torch.no_grad()
def test_loop(model: OldCodeGiver, dataloader, dataset: CodeGiverDataset, device: torch.device, verbose=False):
    # Initialize vector search data struct
    vectorDB = VectorSearch(dataset, prune=True)
    total_score = 0

    for i, data in enumerate(dataloader):
        pos_sents, neg_sents, pos_embs, neg_embs = data
        pos_embs, neg_embs = pos_embs.to(device), neg_embs.to(device)
        logits = model(pos_sents, neg_sents)

        words, embeddings, D = vectorDB.search(logits, num_results=1)
        if verbose:
            for i, word in enumerate(words):
                print(f"{i + 1}: {word[0]}: {D[i][0]}")
        
        # for i, word in enumerate(words):
        #     if word not in pos_sents.split(' '):
        #         out_word = word
        #         anchor = torch.tensor(embeddings[0][i]).unsqueeze(0)
        #         anchor = anchor.to(device)
        #         break
        anchors = torch.tensor(embeddings).to(device)
        pos_score, neg_score = calc_cos_score(anchors, pos_embs, neg_embs)
        # if verbose:
        #     print(f"Output: {out_word}")
        #     print(f"Positive: {pos_sents}\nNegative: {neg_sents}")
            #print(f"Pos Scores: {pos_score}\nNeg Score: {neg_score}")
        pos_score, _ = pos_score.sort(descending=True)
        neg_score, _ = neg_score.sort(descending=False)
        comparison = torch.where(pos_score > neg_score, 1., 0.).to(device)
        score = comparison.sum().item()
        if verbose: 
            print(f"Pos Scores Sorted: {pos_score}\nNeg Scores Sorted: {neg_score}")
            print(f"Score: {score}")
        total_score += score
        i += 1
    print(f"Average Score: {total_score/len(dataset)}")
        
def main(args):
    device = utils.get_device(args.cuda)
    verbose = True if args.v.lower() == 'y' else False
    use_raw = True if args.raw.lower() == 'y' else False
    if use_raw:
        model = CodeGiverRaw(device)
    else:
        model = OldCodeGiver()
    model.load_state_dict(torch.load(args.m))
    model.to(device)
    model.eval()

    test_dataset = CodeGiverDataset(code_dir=args.code_dir, game_dir=args.geuss_dir)
    dataloader = DataLoader(test_dataset, batch_size=200)

    test_loop(model, dataloader, test_dataset, device, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-code_dir', type=str, help='Dataset Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json")
    parser.add_argument('-geuss_dir', type=str, help="", default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/five_word_data_mini.json")
    parser.add_argument('-m', type=str, help='Model Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/test.pth")
    parser.add_argument('-raw', type=str, help="Use the Raw Sentence Encoder, [y/N]", default='N')
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    parser.add_argument('-v', type=str, help="Verbose [y/N]", default='Y')
    args = parser.parse_args()
    main(args)