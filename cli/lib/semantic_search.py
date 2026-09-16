from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings : None
        self.documents : None
        self.document_map = {}


    def generate_embedding(self, text: str):
        if not text:
            raise ValueError("Blank message aa gya chuuu")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        
        texts = []
        for document in documents:
            self.document_map[document["id"]] = document
            texts.append(f"{document['title']}: {document['description']}")



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