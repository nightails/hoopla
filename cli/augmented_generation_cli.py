import argparse

from lib.augmented_generation import rag_search, summarize_rag_search


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarize & generate answer)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Optional display limit of results"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            answer = rag_search(query)
            print("\nSearch Results:")
            for doc in answer["results"]:
                print(f" - {doc['title']}")
            print("\n RAG Response:")
            print(f" {answer['response']}")
        case "summarize":
            query = args.query
            limit = args.limit
            answer = summarize_rag_search(query, limit)
            print("\nSearch Results:")
            for doc in answer["results"]:
                print(f" - {doc['title']}")
            print("\n RAG Response:")
            print(f" {answer['response']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
