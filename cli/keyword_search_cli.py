import argparse
import json

from text import process_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = movie_search(args.query)
            for movie in movies:
                print(movie["title"])
        case _:
            parser.print_help()

def movie_search(keyword) -> list[dict]:
    movies_list: list[dict] = []

    with open("data/movies.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    movies = data["movies"]

    keyword_tokens = process_text(keyword)

    for movie in movies:
        title_tokens = process_text(movie["title"])
        if any(keyword_token in title_token for keyword_token in keyword_tokens for title_token in title_tokens):
            movies_list.append(movie)

    return movies_list[:5]

if __name__ == "__main__":
    main()