import collections
import json
import math
import os
import pickle

from text import process_text


BM25_K1 = 1.5

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, collections.Counter] = {}

    def __add_document(self, doc_id: int, text: str):
        content_tokens = process_text(text)
        for token in content_tokens:
            self.index[token] = self.index.get(token, set()).union({doc_id})
            self.term_frequencies[doc_id] = self.term_frequencies.get(doc_id, collections.Counter())
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list:
        term = term.lower()
        doc_ids = self.index.get(term, set())
        docs = []
        for i in doc_ids:
            docs.append(self.docmap[i])
        return sorted(docs, key=lambda x: x['id'])

    def get_tf(self, doc_id, term) -> int:
        tokens = process_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single searchable token")
        return self.term_frequencies.get(doc_id, collections.Counter()).get(tokens[0], 0)

    def get_bm25_idf(self, term: str) -> float:
        n = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1) -> float:
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 +1)) / (tf + k1)

    def build(self):
        with open("data/movies.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        movies = data["movies"]
        for m in movies:
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m

    def save(self):
        os.makedirs("cache", exist_ok=True)
        pickle.dump(self.index, open("cache/index.pkl", "wb"))
        pickle.dump(self.docmap, open("cache/docmap.pkl", "wb"))
        pickle.dump(self.term_frequencies, open("cache/term_frequencies.pkl", "wb"))

    def load(self):
        if os.path.exists("cache/index.pkl") and os.path.exists("cache/docmap.pkl"):
            self.index = pickle.load(open("cache/index.pkl", "rb"))
            self.docmap = pickle.load(open("cache/docmap.pkl", "rb"))
            self.term_frequencies= pickle.load(open("cache/term_frequencies.pkl", "rb"))
        else:
            raise FileNotFoundError("Index files not found. Please build the index first.")

def bm25_idf_command(term: str) -> float:
    term = process_text(term)[0]
    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()

    return movies_index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1) -> float:
    term = process_text(term)[0]
    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()

    return movies_index.get_bm25_tf(doc_id, term, k1)