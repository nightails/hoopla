import argparse

from lib.augmented_generation import rag_search


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

    citations_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + generate answer with citations)"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Optional display limit of results"
    )

    question_parser = subparsers.add_parser(
        "question", help="Perform RAG (search + generate direct answer to the query)"
    )
    question_parser.add_argument("query", type=str, help="Search query for RAG")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Optional display limit of results"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            answer = rag_search("rag", query)
            print_answer(answer)
        case "summarize":
            query = args.query
            limit = args.limit
            answer = rag_search("summarize", query, limit)
            print_answer(answer)
        case "citations":
            query = args.query
            limit = args.limit
            answer = rag_search("citations", query, limit)
            print_answer(answer)
        case "question":
            query = args.query
            limit = args.limit
            answer = rag_search("question", query, limit)
            print_answer(answer)
        case _:
            parser.print_help()


def print_answer(answer: dict):
    print("\nSearch Results:")
    for doc in answer["results"]:
        print(f" - {doc['title']}")
    print("\n RAG Response:")
    print(f" {answer['response']}")


if __name__ == "__main__":
    main()
