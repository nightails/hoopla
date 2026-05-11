import os, json
import numpy as np

from .semantic_search import (
    MOVIE_PATH,
    SemanticSearch,
    semantic_chunk,
    cosine_similarity,
)


CHUNK_CACH_PATH = "cache/chunk_embeddings.npy"
CHUNK_MEATADATA_CACH_PATH = "cache/chunk_metadata.json"

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embedings(self, documents: list[dict]) -> list:
        self.documents = documents
        all_chunks: list[str] = []
        metadata: list[dict] = []

        for i, doc in enumerate(self.documents):
            self.document_map[doc['id']] = doc
            description = doc['description']
            if description == "":
                continue
            doc_chunks = semantic_chunk(description, 4, 1)
            all_chunks.extend(doc_chunks)

            for j, _ in enumerate(doc_chunks):
                metadata.append({
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(doc_chunks),
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        np.save(CHUNK_CACH_PATH, self.chunk_embeddings)
        with open(CHUNK_MEATADATA_CACH_PATH, "w") as f:
            json.dump({
                "chunks": self.chunk_metadata,
                "total_chunks": len(all_chunks),
            }, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(CHUNK_CACH_PATH) and os.path.exists(CHUNK_MEATADATA_CACH_PATH):
            self.chunk_embeddings = np.load(CHUNK_CACH_PATH)
            with open(CHUNK_MEATADATA_CACH_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
                self.chunk_metadata = data['chunks']
            return self.chunk_embeddings
        return self.build_chunk_embedings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("Missing chunk_embeddings or metadata, run load_or_create_chunk_embeddings first.")
        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict] = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chunk_embedding, query_embedding)
            chunk_score = {
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score,
            }
            chunk_scores.append(chunk_score)

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score['movie_idx']
            score = chunk_score['score']
            if not movie_idx in movie_scores:
                movie_scores[movie_idx] = score
            else:
                movie_scores[movie_idx] = max(score, movie_scores[movie_idx])
        movie_scores = list(movie_scores.items())
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        movie_scores = movie_scores[:limit]

        results = []
        for index, score in movie_scores:
            doc = self.documents[index]
            result = {
                "id": doc['id'],
                "title": doc['title'],
                "document": doc['description'][:100],
                "score": round(score, 3),
                "metadata": doc.get('metadata') or {}
            }
            results.append(result)
        return results


def embed_chunks():
    semantic_chunk = ChunkedSemanticSearch()

    with open(MOVIE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    docs = data["movies"]
    embeddings = semantic_chunk.load_or_create_chunk_embeddings(docs)

    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(query: str, limit: int = 10):
    with open(MOVIE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    docs = data["movies"]

    inst = ChunkedSemanticSearch()
    inst.load_or_create_chunk_embeddings(docs)

    results = inst.search_chunks(query, limit)
    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
    

