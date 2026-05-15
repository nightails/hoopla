from .hybrid_search import HybridSearch
from .llm import load_llm, prompt_gemini
from .search_utils import load_movies


def rag_search(query: str) -> dict:
    # load the movies dataset
    movies = load_movies()
    # rrf_search for the top 5
    search = HybridSearch(movies)
    docs = search.rrf_search(query, 60, 5)

    # prompt the LLM for a natural-language answer
    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""

    client = load_llm()
    resp = prompt_gemini(client, prompt)
    resp_text = (resp.text or "").strip()

    return {"results": docs, "response": resp_text}

