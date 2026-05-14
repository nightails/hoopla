import argparse

from lib.llm import (
    enhance_query,
    rerank,
    evaluate
)
from lib.search_utils import DOCUMENT_PREVIEW_LENGTH, load_movies
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

    rrf_search_parser = subparser.add_parser("rrf-search", help="Hybrid search using Reciprocal Rank Fusion")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument("-k", type=int, nargs="?", default=60, help="Optional tune value")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Optional display limit of results")
    rrf_search_parser.add_argument(
        "--enhance", 
        type=str, 
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method", 
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Query reranking method",
    )
    rrf_search_parser.add_argument("--evaluate", action="store_true", help="Optional evaluation")

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
        case "rrf-search":
            query = args.query
            print(f"\nLog: original query: {query}")

            movies = load_movies()
            search = HybridSearch(movies)

            if args.enhance:
                query = enhance_query(args.enhance, query)
                print(f"\nLog: query after enhancements: {query}")

                if query != args.query:
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            if args.rerank_method:
                results = search.rrf_search(query, args.k, args.limit*5)
                print("\nLog: RRF Search results:")
                for r in results:
                    print(f"Log:   {r["title"]}")

                results = rerank(args.rerank_method, query, results, args.limit)
                print("\nLog: Results after re-ranking:")
                for r in results:
                    print(f"Log:   {r["title"]}")

                print(f"\nRe-ranking top {args.limit} results using {args.rerank_method} method...")
                print(f"Reciprocal Rrank Fusion Results for '{query}' (k={args.k})")
                for i, r in enumerate(results, start=1):
                    print(f"\n{i}. {r.get('title')}")
                    match args.rerank_method:
                        case "individual":
                            print(f"   Re-rank Score: {r["rerank"]:0.3f}/10")
                        case "batch":
                            print(f"   Re-rank Rank: {r["rerank"]}")
                        case "cross_encoder":
                            print(f"   Cross Encoder Score: {r["rerank"]:0.3f}")
                    print(f"   RRF Score: {r["rrf"]:0.4f}")
                    print(f"   BM25: {r["bm25"]}, Semantic: {r["semantic"]}")
                    print(f"   {r["document"]}..")

            else:
                results = search.rrf_search(query, args.k, args.limit)

                for i, r in enumerate(results, start=1):
                    print(f"\n{i}. {r["title"]}")
                    print(f"   RRF Score: {r["rrf"]:0.4f}")
                    print(f"   BM25: {r["bm25"]}, Semantic: {r["semantic"]}")
                    print(f"   {r["document"]}..")

            if args.evaluate:
                results = evaluate(query, results)
                print("\nLLM evaluate:")
                for i, r in enumerate(results):
                    print(f"{i}. {r["title"]}: {r["rank"]}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
