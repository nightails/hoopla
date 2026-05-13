import os

from dotenv import load_dotenv
from google import genai

message = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model='gemma-4-31b-it',
        contents=message,
    )

    print(f"{resp.text}")
    print(f"\nPrompt tokens: {resp.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {resp.usage_metadata.candidates_token_count}")

if __name__ == "__main__":
    main()
