import argparse
import math
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from index import InvertedIndex
from text import process_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and save inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get frequency for a term")
    tf_parser.add_argument("movie_id", type=int, help="Movie ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser("idf", help="Get document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to search for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = movie_search(args.query)
            for movie in movies:
                print(f"id: {movie['id']}\ntitle: {movie['title']}\n")
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case "tf":
            movies_index = InvertedIndex()
            try:
                movies_index.load()
            except FileNotFoundError as err:
                print(err)
                exit()
            tf = movies_index.get_tf(args.movie_id, args.term)
            print(f"Term '{args.term}' frequency in movie {args.movie_id}: {tf}")
        case "idf":
            term = process_text(args.term)[0]
            movies_index = InvertedIndex()
            try:
                movies_index.load()
            except FileNotFoundError as err:
                print(err)
                exit()

            total_doc_count = len(movies_index.docmap) + 1
            total_match_doc_count = len(movies_index.get_documents(term)) + 1

            idf = math.log(total_doc_count / total_match_doc_count)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case _:
            parser.print_help()

def movie_search(keyword) -> list[dict]:
    keyword_tokens = process_text(keyword)

    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()

    movies_list: list[dict] = []

    for token in keyword_tokens:
        movies_list.extend(movies_index.get_documents(token))
        if len(movies_list) >= 5:
            break

    return movies_list[:5]

if __name__ == "__main__":
    main()