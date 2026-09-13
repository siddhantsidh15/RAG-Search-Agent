import pickle
from pathlib import Path
from nltk.stem import PorterStemmer
import string
import json

CACHE_DIR = Path("cache")
CACHE_DIR_MOVIES = CACHE_DIR / "movies.pkl"
CACHE_DIR_STOP_WORDS = CACHE_DIR / "stop_words.pkl"
CACHE_DIR.mkdir(exist_ok=True)

def load_stop_words():
    stopWords = []

    path = Path(CACHE_DIR_STOP_WORDS)

    if path.exists():
        with open(CACHE_DIR_STOP_WORDS, "rb") as f :
            stopWords = pickle.load(f)
    else:     
        with open("data/stopwords.txt", "r") as f:
            stopWords = normalize_text(f.read()).split()
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
                movie["clean_title"] = normalize_text(movie["title"]).split()

            with open(CACHE_DIR_MOVIES, "wb") as f :
                pickle.dump(movies, f)
    return movies

def normalize_text(text: str) -> str :
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def check_term(arr1, arr2, stop_words) -> bool:
    stemmer = PorterStemmer()
    arr1_filtered = [stemmer.stem(word) for word in arr1 if word not in stop_words]
    arr2_filtered = [stemmer.stem(word) for word in arr2 if word not in stop_words]
    if any(q_term in t_term for q_term in arr1_filtered for t_term in arr2_filtered):
        return True
    return False

STOPWORDS = load_stop_words()

def tokenize_text(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = text.split()
    valid_tokens = []

    for token in tokens : 
        if token :
            valid_tokens.append(token)

    # removing stop words
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)

    
    # normalizing the words running, ran, run to run run run
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

    