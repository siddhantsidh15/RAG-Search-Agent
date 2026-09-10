import argparse
import json
import string
import pickle
from pathlib import Path
from nltk.stem import PorterStemmer

CACHE_DIR = Path("cache")
CACHE_DIR_MOVIES = CACHE_DIR / "movies.pkl"
CACHE_DIR_STOP_WORDS = CACHE_DIR / "stop_words.pkl"
CACHE_DIR.mkdir(exist_ok=True)

stemmer = PorterStemmer()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

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
        case _:
            parser.print_help()


def load_stop_words():
    stopWords = []

    path = Path(CACHE_DIR_STOP_WORDS)

    if path.exists():
        with open(CACHE_DIR_STOP_WORDS, "rb") as f :
            stopWords = pickle.load(f)
    else:     
        with open("data/stopwords.txt", "r") as f:
            stopWords = f.read().lower().translate(str.maketrans('', '', string.punctuation)).split()
            with open(CACHE_DIR_STOP_WORDS, "wb") as f :
                pickle.dump(stopWords, f)

    return stopWords


def load_movies():
    movies = []

    path = Path(CACHE_DIR_MOVIES)

    if path.exists():
        with open(CACHE_DIR_MOVIES, "rb") as f :
            movies = pickle.load(f)
    else:
        with open("data/movies.json", "r") as f:
            data  = json.load(f)
            movies = data["movies"]

            for movie in movies : 
                movie["clean_title"] = movie["title"].lower().translate(str.maketrans('', '', string.punctuation)).split()

            with open(CACHE_DIR_MOVIES, "wb") as f :
                pickle.dump(movies, f)
    return movies

def check_term(arr1, arr2, stop_words) -> bool:
    arr1_filtered = [stemmer.stem(word) for word in arr1 if word not in stop_words]
    arr2_filtered = [stemmer.stem(word) for word in arr2 if word not in stop_words]
    if any(q_term in t_term for q_term in arr1_filtered for t_term in arr2_filtered):
        return True
    return False

if __name__ == "__main__":
    main()