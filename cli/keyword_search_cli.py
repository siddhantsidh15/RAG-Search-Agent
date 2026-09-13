import argparse
import string

from utils import tokenize_text,check_term
from inverted_index import build_command, tf_command, search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parse = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a given document ID and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    args = parser.parse_args()

    match args.command:
        case "search":
            QUERY = args.query
            print(f'Searching for: {QUERY}')

            results = search_command(QUERY, 5)
            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")
        case "build":
            build_command()
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()