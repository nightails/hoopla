import collections
import json
import math
import os
import pickle

from .text import process_text


BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, collections.Counter] = {}
        self.doc_lengths: dict = {}
        self.doc_lengths_path = "cache/doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str):
        content_tokens = process_text(text)
        self.doc_lengths[doc_id] = len(content_tokens)
        for token in content_tokens:
            self.index[token] = self.index.get(token, set()).union({doc_id})
            self.term_frequencies[doc_id] = self.term_frequencies.get(doc_id, collections.Counter())
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

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

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length == 0:
            return 0.0

        len_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * len_norm)

    def build(self):
        with open("data/movies.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        movies = data["movies"]

        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def bm25(self, doc_id, term) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int) -> list[tuple[dict, float]]:
        query_terms = process_text(query)

        results: list[tuple[dict, float]] = []

        for doc_id, doc in self.docmap.items():
            score = sum(self.bm25(doc_id, term) for term in query_terms)
            results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open("cache/doc_lengths.pkl", "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        required_cache_files = [
            "cache/index.pkl",
            "cache/docmap.pkl",
            "cache/term_frequencies.pkl",
            "cache/doc_lengths.pkl",
        ]

        if not all(os.path.exists(path) for path in required_cache_files):
            raise FileNotFoundError("Index files not found. Please build the index first.")

        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)

        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)

        with open("cache/doc_lengths.pkl", "rb") as file:
            self.doc_lengths = pickle.load(file)
