
import pickle
from collections import defaultdict, Counter
import os
import math

from utils import CACHE_DIR, tokenize_text, load_movies

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"
        self.tf_path = CACHE_DIR / "term_frequencies.pkl"
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id : int, text : str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[term])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_documents(self, term) -> list[int]:
        return sorted(self.index.get(term, set()))

    def build(self) -> None:
        movies = load_movies()
        for m in movies: 
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok = True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:

        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.tf_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            pass
        


def search_command(QUERY: str, limit : int) -> list[dict]:
    try :
        idx = InvertedIndex()
        idx.load()
        # -----------
        query_tokens = tokenize_text(QUERY)

        seen, results = set(), []
        for query_token in query_tokens:
            matching_doc_ids = idx.get_documents(query_token)
            for doc_id in matching_doc_ids:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                doc = idx.docmap[doc_id]
                results.append(doc)
                if len(results) >= limit:
                    return results
        return results
    except FileNotFoundError:
        pass


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tokenize_single_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    return tokens[0]


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, tokenize_single_term(term))


def idf_command(term: str) -> None:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(tokenize_single_term(term))