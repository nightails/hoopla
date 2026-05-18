import argparse

from lib.search_utils import DOCUMENT_PREVIEW_LENGTH
from lib.multimodal_search import image_search_command


def main():
    parser = argparse.ArgumentParser(description="Multimodel Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search using an image"
    )
    image_search_parser.add_argument("image", type=str, help="Image to query")

    args = parser.parse_args()

    match args.command:
        case "image_search":
            image_path = args.image
            results = image_search_command(image_path)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (similarity: {result['score']:0.3f})")
                print(f"    {result['description'][:DOCUMENT_PREVIEW_LENGTH]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
