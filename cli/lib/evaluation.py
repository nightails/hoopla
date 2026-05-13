from .search_utils import (
    load_movies,
    load_golden_dataset
)
from .hybrid_search import HybridSearch

def evaluate(limit: int = 5) -> list[dict]:
    test_cases = load_golden_dataset()
    movies = load_movies()
    search = HybridSearch(movies)

    results = []
    for test_case in test_cases:
        query = test_case["query"]
        documents = search.rrf_search(query, 60, limit)
        retrieved_titles = [d["title"] for d in documents]
        relevant_titles = test_case["relevant_docs"]
        results.append({
            "query": query,
            "retrieved": retrieved_titles,
            "relevant":relevant_titles, 
        })

    for i, result in enumerate(results):
        total_retrieved = result["retrieved"]
        total_relevant = result["relevant"]
        relevants = get_relevants(total_retrieved, total_relevant)

        precision = calculate_precision(len(relevants), len(total_retrieved))
        recall = calculate_recall(len(relevants), len(total_relevant))
        
        results[i]["precision"] = precision
        results[i]["recall"] = recall

    return results

def get_relevants(total_retrieved: list, total_relevant: list) -> list:
    relevants = []
    for title in total_retrieved:
        if title in total_relevant:
            relevants.append(title)
    return relevants

def calculate_precision(relevant_retrieved: int , total_retrieved: int) -> float:
    return relevant_retrieved / total_retrieved

def calculate_recall(relevant_retrieved: int, total_relevant: int) -> float:
    return relevant_retrieved / total_relevant

