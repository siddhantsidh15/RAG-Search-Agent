import argparse
import string

from utils import load_movies,load_stop_words,check_term
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            QUERY = args.query.lower().translate(str.maketrans('', '', string.punctuation)).split(' ')
            print(f'Searching for: {QUERY}')
            movies = load_movies()

            # -----------
            stopWords = load_stop_words()

            results = []

            for movie in movies:
                clean_title = movie["clean_title"]
                
                if check_term(QUERY, clean_title, stopWords):
                    results.append(movie)
            for i, movie in enumerate(results, start=1):
                if i>5 : break
                print(f"{i}. {movie['title']}")
        case "build":
            build_command()
        case _:
            parser.print_help()

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


if __name__ == "__main__":
    main()