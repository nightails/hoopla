import argparse
import math

from text import process_text
from index import bm25_idf_command, BM25_K1, bm25_tf_command, InvertedIndex


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
        case "tfidf":
            movies_index = InvertedIndex()
            try:
                movies_index.load()
            except FileNotFoundError as err:
                print(err)
                exit()

            term = process_text(args.term)[0]
            tf = movies_index.get_tf(args.movie_id, term)
            idf = math.log((len(movies_index.docmap) + 1) / (len(movies_index.get_documents(term)) + 1))
            tfidf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document {args.movie_id}: {tfidf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.movie_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document {args.movie_id}: {bm25tf:.2f}")

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