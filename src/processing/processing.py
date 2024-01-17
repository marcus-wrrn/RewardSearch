from sentence_transformers import SentenceTransformer
import json

class Processing:
    def __init__(self, encoder: SentenceTransformer | None, filepath="../data/wordlist-eng.txt", download=False) -> None:
        if download:
            self.codewords = self._get_words(filepath)
            self.code_embeddings = self._get_embeddings(encoder, self.codewords)

            self.guesses = self._get_words("/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/webster.txt")
            self.geuss_embeddings = self._get_embeddings(encoder, self.guesses)
        else:
            with open(filepath, 'r') as file:
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
        
        encoder = SentenceTransformer("all-mpnet-base-v2")
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

if __name__ == "__main__":
    proc = Processing(encoder=None, filepath="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/wordlist-eng.txt", download=True)
    proc.to_json("/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words_extended.json")
