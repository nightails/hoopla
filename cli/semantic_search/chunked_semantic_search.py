import os, json
import numpy as np

from .semantic_search import (
    MOVIE_PATH,
    SemanticSearch,
    semantic_chunk,
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

def embed_chunks():
    semantic_chunk = ChunkedSemanticSearch()

    with open(MOVIE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    docs = data["movies"]
    embeddings = semantic_chunk.load_or_create_chunk_embeddings(docs)

    print(f"Generated {len(embeddings)} chunked embeddings")

