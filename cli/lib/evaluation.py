from .search_utils import (
    load_movies,
    load_golden_dataset
)
from .hybrid_search import HybridSearch

def evaluate_precision(limit: int = 5) -> list[dict]:
    test_cases = load_golden_dataset()
    movies = load_movies()
    search = HybridSearch(movies)

    results = []
    for test_case in test_cases:
        query = test_case["query"]
        documents = search.rrf_search(query, 60, limit)
        retrieved_titles = []
        for d in documents:
            retrieved_titles.append(d["title"])
        relevant_titles = test_case["relevant_docs"]
        precision = len(relevant_titles) / len(retrieved_titles)
        results.append({
            "query": query,
            "precision": precision,
            "retrieved": retrieved_titles,
            "relevant":relevant_titles, 
        })

    return results

