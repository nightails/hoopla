import argparse

from lib.multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodel Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify embedding for given image"
    )
    verify_image_parser.add_argument("image", type=str, help="Image to query")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image
            verify_image_embedding(image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
