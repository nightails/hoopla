import argparse

from search_utils import BM25_K1, bm25_idf_command, bm25_tf_command, BM25_B, movie_search, idf_command, tf_command, \
    tfidf_command, build_command, bm25search


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

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a movie")
    tfidf_parser.add_argument("movie_id", type=int, help="Movie ID")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("movie_id", type=int, help="Movie ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("term", type=str, help="Term to search for")
    bm25search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = movie_search(args.query)
            for movie in movies:
                print(f"id: {movie['id']}\ntitle: {movie['title']}\n")
        case "bm25search":
            results = bm25search(args.term, args.limit)
            i = 1
            for doc, score in results:
                print(f"{i}. ({doc['id']}) {doc['title']} - Score: {score:.2f}")
                i += 1
        case "build":
            build_command()
        case "tf":
            tf = tf_command(args.movie_id, args.term)
            print(f"Term '{args.term}' frequency in movie {args.movie_id}: {tf}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.movie_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document {args.movie_id}: {tfidf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.movie_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document {args.movie_id}: {bm25tf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()