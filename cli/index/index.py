import json
import os
import pickle

from cli.text import process_text


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str):
        content_tokens = process_text(text)
        for token in content_tokens:
            self.index[token] = self.index.get(token, set()).union({doc_id})

    def get_documents(self, term: str) -> list:
        term = term.lower()
        doc_ids = list(self.index.get(term, set()))
        return sorted(doc_ids)

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