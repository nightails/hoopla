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


def summarize_rag_search(query: str, limit: int = 5) -> dict:
    # load the movies dataset
    movies = load_movies()
    # rrf_search for the top 5
    search = HybridSearch(movies)
    docs = search.rrf_search(query, 60, limit)

    # prompt the LLM for a natural-language answer
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {docs}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    client = load_llm()
    resp = prompt_gemini(client, prompt)
    resp_text = (resp.text or "").strip()

    return {"results": docs, "response": resp_text}
