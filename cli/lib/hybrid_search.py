import os

from .keyword_search import InvertedIndex
from .search_utils import DEFAULT_SEARCH_LIMIT, DOCUMENT_PREVIEW_LENGTH
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = []
        for result in bm25_results:
            bm25_scores.append(result["score"])
        bm25_scores = normalize_scores(bm25_scores)

        semantic_scores = []
        for result in semantic_results:
            semantic_scores.append(result["score"])
        semantic_scores = normalize_scores(semantic_scores)

        docs_scores_map = {}

        if bm25_scores is not None:
            for doc, score in zip(bm25_results, bm25_scores):
                doc_id = doc["id"]
                docs_scores_map[doc_id] = {
                    "title": doc["title"],
                    "document": doc["document"][:DOCUMENT_PREVIEW_LENGTH],
                    "bm25": score,
                    "semantic": 0,
                }

        if semantic_scores is not None:
            for doc, score in zip(semantic_results, semantic_scores):
                doc_id = doc["id"]
                if doc_id not in docs_scores_map:
                    docs_scores_map[doc_id] = {
                        "title": doc["title"],
                        "document": doc["document"][:DOCUMENT_PREVIEW_LENGTH],
                        "bm25": 0,
                        "semantic": score,
                    }
                else:
                    docs_scores_map[doc_id]["semantic"] = score

        for key, value in docs_scores_map.items():
            score = hybrid_score(value["bm25"], value["semantic"], alpha)
            docs_scores_map[key]["hybrid"] = score

        results = list(docs_scores_map.values())
        results.sort(key=lambda x: x["hybrid"], reverse=True)
        return results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        docs_ranks_map = {}
        for i, bm25 in enumerate(bm25_results, start=1):
            bm25_doc_id = bm25["id"]
            if bm25_doc_id not in docs_ranks_map:
                docs_ranks_map[bm25_doc_id] = {
                    "title": bm25["title"],
                    "document": bm25["document"],
                    "bm25": i,
                    "semantic": None,
                }
            else:
                docs_ranks_map[bm25_doc_id]["bm25"] = i

        for i, semantic in enumerate(semantic_results, start=1):
            semantic_doc_id = semantic["id"]
            if semantic_doc_id not in docs_ranks_map:
                docs_ranks_map[semantic_doc_id] = {
                    "title": semantic["title"],
                    "document": semantic["document"],
                    "bm25": None,
                    "semantic": i,
                }
            else:
                docs_ranks_map[semantic_doc_id]["semantic"] = i

        for id, doc in docs_ranks_map.items():
            bm25_rank = doc.get("bm25")
            semantic_rank = doc.get("semantic")

            if bm25_rank is not None and semantic_rank is not None:
                bm25_rrf = rrf_score(bm25_rank, k)
                semantic_rrf = rrf_score(semantic_rank, k)
                docs_ranks_map[id]["rrf"] = bm25_rrf + semantic_rrf
                continue
            if bm25_rank is not None:
                docs_ranks_map[id]["rrf"] = rrf_score(bm25_rank, k)
                continue
            if semantic_rank is not None:
                docs_ranks_map[id]["rrf"] = rrf_score(semantic_rank, k)
                continue

        results = list(docs_ranks_map.values())
        results.sort(key=lambda x: x["rrf"], reverse=True)
        return results[:limit]


def normalize_scores(scores: list[float]):
    if len(scores) == 0:
        return

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)

    norm_scores = []
    for score in scores:
        norm_score = (score - min_score) / (max_score - min_score)
        norm_scores.append(norm_score)
    return norm_scores


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: float, k=60):
    return 1 / (k + rank)
