import enum
import os, time, json

from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from google import genai


def enhance_query(method: str, query: str):
    client = load_llm()
    match method:
        case "spell":
            message = f"""Fix any spelling errors in the user-provided movie search query below.
            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
            Preserve punctuation and capitalization unless a change is required for a typo fix.
            If there are no spelling errors, or if you're unsure, output the original query unchanged.
            Output only the final query text, nothing else.
            User query: "{query}"
            """
        case "rewrite":
            message = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep the rewritten query concise (under 10 words)
            - It should be a Google-style search query, specific enough to yield relevant results
            - Don't use boolean logic

            Examples:
            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            If you cannot improve the query, output the original unchanged.
            Output only the rewritten query text, nothing else.

            User query: "{query}"
            """
        case "expand":
            message = f"""Expand the user-provided movie search query below with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            Output only the additional terms; they will be appended to the original query.

            Examples:
            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"

            User query: "{query}"
            """
        case _:
            return query

    resp = prompt_gemini(client, message)
    return resp.text


def rerank(method: str, query: str, docs: list[dict], limit: int = 5) -> list[dict]:
    client = load_llm()
    match method:
        case "individual":
            for i, doc in enumerate(docs):
                message = f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("document", "")}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:"""

                resp = prompt_gemini(client, message)
                score_text = (resp.text or "").strip()
                score = int(score_text)
                docs[i]["rerank"] = score

                time.sleep(20)

            docs.sort(key=lambda x: x["rerank"], reverse=True)

        case "batch":
            message = f"""Rank the movies listed below by relevance to the following search query.

            Query: "{query}"

            Movies:
            {docs}

            Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

            For example:
            [75, 12, 34, 2, 1]

            Ranking:"""

            resp = prompt_gemini(client, message)
            resp_text = resp.text.strip("```json ")
            ranked_ids = json.loads(resp_text)

            id_rank_map = {}
            for i, id in enumerate(ranked_ids):
                id = int(id)
                id_rank_map[id] = i

            for i, doc in enumerate(docs):
                rank = id_rank_map.get(i)
                if rank is None:
                    docs[i]["rerank"] = len(docs)
                    continue
                docs[i]["rerank"] = rank

            docs.sort(key=lambda x: x["rerank"])

        case "cross_encoder":
            pairs = []
            for doc in docs:
                pairs.append(
                    [query, f"{doc.get('title', '')} - {doc.get('document', '')}"]
                )
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = cross_encoder.predict(pairs)
            for i, score in enumerate(scores):
                docs[i]["rerank"] = score
            docs.sort(key=lambda x: x["rerank"], reverse=True)

    return docs[:limit]


def evaluate(query: str, current_results: list) -> list:
    formatted_results = []
    for i, r in enumerate(current_results, start=1):
        formatted_results.append(f"{i}. {r['title']}: {r['document']}")

    client = load_llm()
    message = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""

    resp = prompt_gemini(client, message)
    resp_text = resp.text.strip("```json ")
    ranks = json.loads(resp_text)

    results = []
    for rank, result in zip(ranks, current_results):
        results.append(
            {
                "title": result["title"],
                "rank": rank,
            }
        )

    return results


def load_llm() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    return genai.Client(api_key=api_key)


def prompt_gemini(client, message):
    return client.models.generate_content(model="gemma-4-31b-it", contents=message)
