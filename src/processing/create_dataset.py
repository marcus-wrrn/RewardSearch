import random
import json
from sentence_transformers import SentenceTransformer
import torch

class GameBoard:
    def __init__(self, words: list, pos_num=6, neg_num=6, has_assassin=False):
        nwords = len(words)
        assert nwords > pos_num + neg_num + int(has_assassin)

        random.shuffle(words)
        self.pos_words = words[:pos_num]
    
        self.pos_words = words[:pos_num]
        self.neg_words = words[pos_num:pos_num + neg_num:]
    
        if has_assassin:
            self.assassin = words[-1]
            self.neutral = words[pos_num + neg_num:-1]
        else:
            self.assassin = None
            self.neutral = words[pos_num + neg_num:]

    def is_assassin(self, choice: str) -> bool:
        return self.assassin != None and self.assassin == choice

    def remove_word(self, word: str):
        """
        0 -: Word has been removed
        1 -: Word does not exist
        -1 -: Assassin has been called end game (should not get here)
        """
        
        for word_list in [self.pos_words, self.neg_words, self.neutral]:
            if word in word_list:
                word_list.remove(word)
                return 0
        if not self.is_assassin(word): return 1

        return -1
    
    def get_words_string(self, words_list: list):
        return ' '.join(words_list)
    
    def print_board(self):
        print(f"Board:")
        print(f"Positive Words: {self.pos_words}")
        print(f"Negative Words: {self.neg_words}")
        print(f"Neutral Words: {self.neutral}")
        print(f"Assassin: {self.assassin}")
    

class GameManager:
    def __init__(self, wordfile: str, encoder: SentenceTransformer, num_positive=9, num_negative=9, has_assassin=False, nwords=25) -> None:
        self.words = self._get_words(wordfile)
        self.nwords = nwords
        self.num_pos = num_positive
        self.num_neg = num_negative
        self.has_assassin = has_assassin

        self.board = self._create_gameboard()
        self.encoder = encoder

    def _get_words(self, wordfile: str):
        with open(wordfile, 'r') as fp:
            data = json.load(fp)
        words = data['codewords']
        random.shuffle(words)
        return words
    
    def _create_gameboard(self):
        return GameBoard(self.words[:self.nwords], self.num_pos, self.num_neg, self.has_assassin)
    
    def get_sentences(self):
        pos = " ".join(self.board.pos_words)
        neg = " ".join(self.board.neg_words)
        neutral = " ".join(self.board.neutral)
        assassin = self.board.assassin
        return pos, neg, neutral, assassin
    
    def get_encoding(self):
        pos, neg, neutral = self.get_sentences()
        with torch.no_grad():
            pos_emb = self.encoder.encode(pos)
            neg_emb = self.encoder.encode(neg, )
            neutral_emb = self.encoder.encode(neutral)
        return pos_emb, neg_emb, neutral_emb
    
    def shuffle_board(self):
        random.shuffle(self.words)
        self.board = self._create_gameboard()

def create_dataset(game_manager: GameManager, filepath: str, num_datapoints: int, max_words=3, fixed_size=True):
    pos_words = []
    neg_words = []
    neut_words = []
    assassin_words = []
    for i in range(num_datapoints):
        pos, neg, neut, assassin = game_manager.get_sentences()
        pos_words.append(pos)
        neg_words.append(neg)
        neut_words.append(neut)
        assassin_words.append(assassin)
        # Shuffle board to get new words each time
        game_manager.shuffle_board()
    
    data = {
        'positive': pos_words,
        'negative': neg_words,
        'neutral': neut_words,
        'assassin': assassin_words
    }

    with open(filepath, 'w') as file:
        json.dump(data, file)
    print("Dataset created")

def main():
    wordfile = "/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json"
    practice_dataset = "/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/codewords_full_w_assassin_valid.json"


    encoder = SentenceTransformer('all-mpnet-base-v2')
    manager = GameManager(wordfile, encoder, num_positive=9, num_negative=9, has_assassin=True, nwords=25)
    create_dataset(manager, practice_dataset, num_datapoints=100000)

    
if __name__ == "__main__":
    main()