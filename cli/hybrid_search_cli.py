import argparse

from lib.search_utils import load_movies
from lib.hybrid_search import (
    HybridSearch,
    normalize_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize numbers")
    normalize_parser.add_argument("numbers", type=float, nargs="*", help="Numbers to normalize")

    weight_search_parser = subparser.add_parser("weighted-search", help="Hybrid search using BM25 and Semantic")
    weight_search_parser.add_argument("query", type=str, help="Query to search")
    weight_search_parser.add_argument("--alpha", type=float, nargs="?", default=0.5, help="Optional weight between BM25 and Semantic")
    weight_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Optional display limit of results")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.numbers)
            if scores is None:
                return
            for score in scores:
                print(f"* {score:0.4f}")
        case "weighted-search":
            movies = load_movies()
            search = HybridSearch(movies)
            results = search.weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results, start=1):
                print(f"{i}. {r.get('title')}")
                print(f"  Hybrid Score: {r.get('hybrid'):0.4f}")
                print(f"  BM25: {r.get('bm25'):0.4f}, Semantic: {r.get('semantic'):0.4f}")
                print(f"  {r.get('document')}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
