import os, json, re
import numpy as np

from sentence_transformers import SentenceTransformer


MOVIE_PATH = "data/movies.json"
CACHE_PATH = "cache/movie_embeddings.npy"

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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

    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        scores: list[tuple] = []
        query_embedding = self.generate_embedding(query)
        for e, doc in zip(self.embeddings, self.documents) :
            cs = cosine_similarity(query_embedding, e)
            scores.append((cs, doc))
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        results: list[dict] = []
        for s in sorted_scores:
            results.append({"score": s[0], "title": s[1]['title'], "description": s[1]['description']})
        return results[:limit]

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

    with open(MOVIE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    docs = data["movies"]

    embeddings = semantic_search.load_or_create_embeddings(docs)
    print(f"Number of docs:   {len(docs)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def search(query: str, limit=5):
    semantic_search = SemanticSearch()

    with open(MOVIE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    docs = data["movies"]
    semantic_search.load_or_create_embeddings(docs)

    results = semantic_search.search(query, limit)
    for i, result in enumerate(results, start=1):
        print(f"{i}: {result['title']}: (score: {result['score']})")
        print(f" {result['description']}")

def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        if overlap > 0 and i > 0:
            chunk_words = words[i-overlap:i+chunk_size]
        else:
            chunk_words = words[i:i+chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size


    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")

def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []

    n_sentences = len(sentences)
    i = 0
    while i < n_sentences:
        chunk_sentences = sentences[i:i+max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ". join(chunk_sentences))
        i += max_chunk_size - overlap

    return chunks



def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

