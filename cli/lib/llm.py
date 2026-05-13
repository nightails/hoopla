import os

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
        case _:
            return query

    resp = prompt_gemini(client, message)
    return resp.text


def load_llm():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    return genai.Client(api_key=api_key)

def prompt_gemini(client, message):
    return client.models.generate_content(
        model = 'gemma-4-31b-it',
        contents = message
    )

