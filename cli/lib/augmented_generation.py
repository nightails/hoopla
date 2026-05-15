from .hybrid_search import HybridSearch
from .llm import load_llm, prompt_gemini
from .search_utils import load_movies


def rag_search(method: str, query: str, limit: int = 5) -> dict:
    # load the movies dataset
    movies = load_movies()
    # rrf_search for the top 5
    search = HybridSearch(movies)
    docs = search.rrf_search(query, 60, limit)

    # prompt the LLM for a natural-language answer
    prompt = ""
    match method:
        case "rag":
            prompt = rag_prompt(query, docs)
        case "summarize":
            prompt = summarize_prompt(query, docs)
        case "citations":
            prompt = citations_prompt(query, docs)

    client = load_llm()
    resp = prompt_gemini(client, prompt)
    resp_text = (resp.text or "").strip()

    return {"results": docs, "response": resp_text}


def rag_prompt(query: str, docs: list) -> str:
    return f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""


def summarize_prompt(query: str, docs: list) -> str:
    return f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {docs}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""


def citations_prompt(query: str, docs: list) -> str:
    return f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Hoopla, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {docs}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""
