import random
import json
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

class GameBoard:
    def __init__(self, text: list, pos_num=6, neg_num=6, has_assassin=False):
        nwords = len(text)
        assert nwords > pos_num + neg_num + int(has_assassin)

        random.shuffle(text)
        self.pos_texts = text[:pos_num]
    
        self.pos_texts = text[:pos_num]
        self.neg_texts = text[pos_num:pos_num + neg_num:]
    
        if has_assassin:
            self.assassin = text[-1]
            self.neut_texts = text[pos_num + neg_num:-1]
        else:
            self.assassin = None
            self.neut_texts = text[pos_num + neg_num:]

    def is_assassin(self, choice: str) -> bool:
        return self.assassin != None and self.assassin == choice

    def remove_text(self, word: str):
        """
        0 -: Word has been removed
        1 -: Word does not exist
        -1 -: Assassin has been called end game (should not get here)
        """
        
        for word_list in [self.pos_texts, self.neg_texts, self.neut_texts]:
            if word in word_list:
                word_list.remove(word)
                return 0
        if not self.is_assassin(word): return 1

        return -1
    
    def get_texts_string(self, text_list: list):
        return '<SEP>'.join(text_list)
    
    def print_board(self):
        print(f"Board:")
        print(f"Positive Words: {self.pos_texts}")
        print(f"Negative Words: {self.neg_texts}")
        print(f"Neutral Words: {self.neut_texts}")
        print(f"Assassin: {self.assassin}")
    

class GameManager:
    def __init__(self, textfile: str, encoder: SentenceTransformer, num_positive=9, num_negative=9, has_assassin=False, ntexts=25, seperator=" ") -> None:
        self.texts = self._get_texts(textfile)
        self.ntexts = ntexts
        self.num_pos = num_positive
        self.num_neg = num_negative
        self.has_assassin = has_assassin

        self.board = self._create_gameboard()
        self.encoder = encoder

        self.seperator = seperator

    def _get_texts(self, wordfile: str):
        with open(wordfile, 'r') as fp:
            data = json.load(fp)
        words = data['codewords']
        random.shuffle(words)
        return words
    
    def _create_gameboard(self):
        return GameBoard(self.texts[:self.ntexts], self.num_pos, self.num_neg, self.has_assassin)
    
    def get_combined_texts(self):
        pos = f"{self.seperator}".join(self.board.pos_texts)
        neg = f"{self.seperator}".join(self.board.neg_texts)
        neutral = f"{self.seperator}".join(self.board.neut_texts)
        assassin = self.board.assassin
        return pos, neg, neutral, assassin
    
    def get_encoding(self):
        pos, neg, neutral = self.get_combined_texts()
        with torch.no_grad():
            pos_emb = self.encoder.encode(pos)
            neg_emb = self.encoder.encode(neg)
            neutral_emb = self.encoder.encode(neutral)
        return pos_emb, neg_emb, neutral_emb
    
    def shuffle_board(self):
        random.shuffle(self.texts)
        self.board = self._create_gameboard()

class LongTextManager(GameManager):
    def __init__(self, wordfile: str, encoder: SentenceTransformer, num_positive=9, num_negative=9, has_assassin=False, ntexts=25, seperator='<SEP>', limit=None):
        self.limit = limit
        super().__init__(wordfile, encoder, num_positive, num_negative, has_assassin, ntexts, seperator)
        
    def _get_texts(self, wordfile: str):
        data = pd.read_json(wordfile)
        if self.limit:
            data = data[:self.limit]
        data = data['title'].to_list()
        return data

def create_dataset(game_manager: GameManager, filepath: str, num_datapoints: int, max_words=3, fixed_size=True):
    pos_words = []
    neg_words = []
    neut_words = []
    assassin_words = []
    for i in range(num_datapoints):
        print(f"{i+1}/{num_datapoints}")
        pos, neg, neut, assassin = game_manager.get_combined_texts()
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
    textfile = "/home/marcuswrrn/Projects/Machine_Learning/NLP/Codenames/data/news_board.json"
    practice_dataset = "/home/marcuswrrn/Projects/Machine_Learning/NLP/Codenames/data/codetexts_valid.json"


    # encoder = SentenceTransformer('all-mpnet-base-v2')
    manager = LongTextManager(textfile, None, num_positive=9, num_negative=9, has_assassin=True, ntexts=25, seperator="<SEP>")
    create_dataset(manager, practice_dataset, num_datapoints=10000)

    
if __name__ == "__main__":
    main()