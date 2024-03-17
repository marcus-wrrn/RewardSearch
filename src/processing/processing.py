from sentence_transformers import SentenceTransformer
import json
import argparse
"""This script takes a list of words/texts and processes it into a useable format"""



class Processing:
    def __init__(self, encoder: SentenceTransformer | None, board_path: str, vocab_path: str, encode=False) -> None:
        if encode:
            self.codewords = self._get_words(board_path)
            print(f"Processing Board texts")
            self.code_embeddings = self._get_embeddings(encoder, self.codewords)

            self.guesses = self._get_words(vocab_path)
            print(f"Processing Vocab texts")
            self.geuss_embeddings = self._get_embeddings(encoder, self.guesses)
        else:
            with open(board_path, 'r') as file:
                data = json.load(file)
            self.codewords = data['codewords']
            self.code_embeddings = data['embeddings']
            self.geusses = data['guesses']
        

    # Initializers
    def _get_words(self, filepath):
        with open(filepath, 'r') as file:
            lines = [line.replace('\n', '').lower() for line in file.readlines()]
        return lines
    
    def _get_embeddings(self, encoder: SentenceTransformer | None, words: list):
        if type(encoder) == SentenceTransformer:
            return encoder.encode(words)
        
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return encoder.encode(words)
    
    def to_json(self, filepath: str):
        data = {
            'codewords': [word for word in self.codewords],
            'guesses': [word for word in self.guesses],
            'code_embeddings': [[float(x) for x in embedding] for embedding in self.code_embeddings],
            'guess_embeddings': [[float(x) for x in embedding] for embedding in self.geuss_embeddings]
        }

        with open(filepath, 'w') as file:
            json.dump(data, file)
    

def main(args):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    encode_text = True if args.encode.lower() == 'y' else False
    print(f"Model: {encoder._get_name()}")
    proc = Processing(encoder=encoder, board_path=args.board, vocab_path=args.vocab, encode=encode_text)
    proc.to_json(args.out)
    print(f"Saved output to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-board', type=str, help="Filepath containing all texts for the board")
    parser.add_argument('-vocab', type=str, help="Filepath containing all vocab texts")
    parser.add_argument('-out', type=str, help="Filepath to save output")
    parser.add_argument('-encode', type=str, help="Create embeddings from text [Y/n]", default='Y')
    args = parser.parse_args()
    main(args)
    
