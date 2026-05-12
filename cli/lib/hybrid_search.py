import os

from .keyword_search import InvertedIndex
from .search_utils import DEFAULT_SEARCH_LIMIT
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
        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)

        bm25_scores = []
        for result in bm25_results:
            bm25_scores.append(result.get('score'))
        bm25_scores = normalize_scores(bm25_scores)

        semantic_scores = []
        for result in semantic_results:
            semantic_scores.append(result.get('score'))
        semantic_scores = normalize_scores(semantic_scores)

        docs_scores_map = {}

        if bm25_scores is not None :
            for doc, score in zip(bm25_results, bm25_scores):
                doc_id = doc.get('id')
                docs_scores_map[doc_id] = {
                    "title": doc.get('title'),
                    "document": doc.get('document'),
                    "bm25": score,
                    "semantic": 0,
                }

        if semantic_scores is not None:
            for doc, score in zip(semantic_results, semantic_scores):
                doc_id = doc.get('id')
                if doc.get('id') not in docs_scores_map:
                    docs_scores_map[doc_id] = {
                        "title": doc.get('title'),
                        "document": doc.get('document'),
                        "bm25": 0,
                        "semantic": score,
                    }
                else:
                    docs_scores_map[doc_id]["semantic"] = score
                
        for key, value in docs_scores_map.items():
            score = hybrid_score(value.get('bm25'), value.get('semantic'), alpha)
            docs_scores_map[key]["hybrid"] = score

        results = list(docs_scores_map.values())
        results.sort(key=lambda x: x['hybrid'], reverse=True)
        return results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

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
