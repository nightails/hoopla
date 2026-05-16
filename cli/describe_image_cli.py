import argparse
import mimetypes

from lib.llm import (
    load_llm,
)
from google import genai
from google.genai import types


def main():
    parser = argparse.ArgumentParser(description="Image Description Generation CLI")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the an image file"
    )
    parser.add_argument("--query", type=str, required=True, help="Query to run")

    args = parser.parse_args()
    image_path = args.image
    query = args.query.strip()

    # check for image type
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    # open image binary
    with open(image_path, "rb") as file:
        image_data = file.read()

    client = load_llm()
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            system_prompt,
            types.Part.from_bytes(data=image_data, mime_type=mime),
            query,
        ],
    )

    print(f"Rewritten query: {resp.text.strip()}")
    if resp.usage_metadata is not None:
        print(f"Total tokens:    {resp.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
