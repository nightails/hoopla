#!/usr/bin/env python3

import argparse
from re import search

from semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search,
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify Sentence-Transformers model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embeddings from text")
    embed_text_parser.add_argument("text", type=str, help="Text to be embeddings")

    subparsers.add_parser("verify_embeddings", help="Verify generated embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embeddings from query")
    embed_query_parser.add_argument("query", type=str, help="Query to be embeddings")

    search_parser = subparsers.add_parser("search", help="Semantically search for movie")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit number of results")

    chunk_parser = subparsers.add_parser("chunk", help="Create chunks from text")
    chunk_parser.add_argument("text", type=str, help="Text to be chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Optional chunk size")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size)
        case _:
            parser.print_help()

def chunk_command(text: str, chunk_size: int):
    words = text.split()
    groups = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    chunks = []
    for line in groups:
        chunks.append(" ".join(line))
    
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")
        

if __name__ == "__main__":
    main()
