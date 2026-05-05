import argparse
import json
import string

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

            pass
        case _:
            parser.print_help()

def movie_search(keyword) -> list[dict]:
    movies_list: list[dict] = []

    with open("data/movies.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    movies = data["movies"]
        
    for movie in movies:
        if keyword.lower() in sanitize_title(movie["title"]):
            movies_list.append(movie)
            
    return movies_list[:5]

def sanitize_title(title: str) -> str:
    punc_map = str.maketrans("", "", string.punctuation)
    return title.translate(punc_map).lower()

if __name__ == "__main__":
    main()