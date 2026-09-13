from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


    def generate_embedding(self, text: str):
        if not text:
            raise ValueError("Blank message aa gya chuuu")
        return self.model.encode([text])[0]



def verify_model() -> None:
    model = SemanticSearch()
    print(f'Model loaded: {model.model}')
    print(f"Max sequence length: {model.model.max_seq_length}")


def embed_text(text) -> None:
    model = SemanticSearch()
    embedding =  model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")