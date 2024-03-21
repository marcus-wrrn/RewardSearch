from models.multi_objective_models import MORSpyMaster
from datasets.dataset import CodeNamesDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
import argparse
import numpy as np
from utils.vector_search import VectorSearch
import utils.utilities as utils
from utils.logger import TestLogger
from sentence_transformers import SentenceTransformer

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

def get_texts(count: int) -> list[str]:
    texts = []
    while count <= 0:
        text = input()
        if not text or text == '':
            break

        texts.append(text)
        count -= 1
    return texts

def get_embeddings(encoder: SentenceTransformer, texts: list[str]) -> Tensor:
    embs = encoder.encode(texts, batch_size=100)
    return torch.tensor(embs, dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def model_inference(model: MORSpyMaster, encoder: SentenceTransformer, device="cpu", verbose=True):
    
    #print("Input Target string:")
    pos_texts = ["Honda", "Coca-Cola", "Apple", "Walmart", "Hewlett-Packard"]
    pos_embs = get_embeddings(encoder, pos_texts).to(device)


    #print("Input Negative strings")
    neg_texts = ["Potted Plant", "Niagra Falls", "Mt. Everest", "Beach", "Frog", "Waterfall"]
    neg_embs = get_embeddings(encoder, neg_texts).to(device)

    #print("Input Neutral strings")
    neut_texts = ["Boat", "Sailing", "Ship"]
    neut_embs = get_embeddings(encoder, neut_texts).to(device)

    #print("Input Assassin string")
    assas_text = "Train"
    assas_emb = get_embeddings(encoder, assas_text).to(device)

    words, search_index, logits = model(pos_embs, neg_embs, neut_embs, assas_emb)
    search_index = search_index.cpu()[0]

    word = words[0][search_index]
    print(word)
        
def main(args):
    device = utils.get_device(args.cuda)
    verbose = True if args.v.lower() == 'y' else False
    use_model_out = True if args.use_model_out.lower() == 'y' else False

    # Initialize data
    print("Loading Data")
    test_dataset = CodeNamesDataset(code_dir=args.code_dir, game_dir=args.guess_dir)
    vector_db = VectorSearch(test_dataset, prune=False)

    # Initialize model
    print("Loading Model")
    model = MORSpyMaster(vector_db, device, vocab_size=args.sw)
    model.load_state_dict(torch.load(args.m))
    model.to(device)
    model.eval()

    encoder = SentenceTransformer('all-mpnet-base-v2')
    encoder.to(device)

    model_inference(model, encoder, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-code_dir', type=str, help='Dataset Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/Codenames/data/words.json")
    parser.add_argument('-guess_dir', type=str, help="", default="/home/marcuswrrn/Projects/Machine_Learning/NLP/Codenames/data/codewords_full_w_assassin_mini.json")
    parser.add_argument('-m', type=str, help='Model Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/Codenames/model_data/words_neut_2/model.pth")
    parser.add_argument('-use_model_out', type=str, help="Whether to use model output or search output, use Y or N", default='N')
    parser.add_argument('-sw', type=int, help='Model search window', default=80)
    parser.add_argument('-b', type=int, help='Batch Size', default=200)
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    parser.add_argument('-v', type=str, help="Verbose [y/N]", default='Y')
    args = parser.parse_args()
    main(args)