import os, json
import numpy as np

from sentence_transformers import SentenceTransformer

MOVIE_PATH = "data/movies.json"
CACHE_PATH = "cache/movie_embeddings.npy"

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map: dict = {}

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("Invalid string")
        embeddings = self.model.encode(text)
        return embeddings

    def build_embeddings(self, documents):
        self.documents = documents
        docs_list = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            docs_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(docs_list, show_progress_bar = True)
        np.save(CACHE_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(CACHE_PATH):
            self.embeddings = np.load(CACHE_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()

    with open("data/movies.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    docs = data["movies"]

    embeddings = semantic_search.load_or_create_embeddings(docs)
    print(f"Number of docs:   {len(docs)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


