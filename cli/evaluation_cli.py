import argparse

from lib.evaluation import evaluate_precision

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    results = evaluate_precision(limit)

    if results is not None:
        print(f"k={limit}")
        for r in results:
            print(f"\n- Query: {r["query"]}")
            print(f"  - Precision@{limit}: {r["precision"]:0.4f}")
            print(f"  - Retrieved: {r["retrieved"]}")
            print(f"  - Relevant: {r["relevant"]}")
    else:
        print("no result")


if __name__ == "__main__":
    main()
