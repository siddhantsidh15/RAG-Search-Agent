# RAG Search Engine

A lightweight search engine prototype built around an inverted index for movie data.

This repository demonstrates:

- text preprocessing with stopword removal and stemming
- inverted index construction and persistence
- keyword search over a movie dataset
- term frequency (TF), inverse document frequency (IDF), and TF-IDF calculations

## Repository Structure

- `pyproject.toml` - project metadata and dependency declarations
- `README.md` - project documentation
- `cli/keyword_search_cli.py` - command-line interface for build/search/TF/IDF operations
- `cli/helper.py` - text normalization, tokenization, stopword filtering, dataset loading
- `cli/inverted_index.py` - inverted index construction, lookup, persistence, and scoring helpers
- `data/movies.json` - movie dataset used for indexing and searching
- `data/stopwords.txt` - stopwords list used for filtering tokens
- `cache/` - generated pickle cache files after building the index

## Requirements

- Python `>= 3.14`
- `nltk==3.9.1`

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -U pip
python -m pip install nltk==3.9.1
```

## Index Building

Before running search or scoring commands, build the inverted index:

```bash
python cli/keyword_search_cli.py build
```

This will generate cache files in `cache/` to support fast query lookups.

## CLI Usage

### Search by query

```bash
python cli/keyword_search_cli.py search "police action"
```

Output example:

```text
Searching for: police action
1. (1) Kaakha..Kaakha: The Police
```

### Compute term frequency (TF)

```bash
python cli/keyword_search_cli.py tf 1 police
```

### Compute inverse document frequency (IDF)

```bash
python cli/keyword_search_cli.py idf police
```

### Compute TF-IDF

```bash
python cli/keyword_search_cli.py tfidf 1 police
```

## Implementation Details

### Text preprocessing

`cli/helper.py` performs the following operations:

- lowercases text
- removes punctuation
- splits text into tokens
- filters stopwords from `data/stopwords.txt`
- stems words using `nltk.stem.PorterStemmer`

### Inverted index

`cli/inverted_index.py` builds an inverted index that:

- maps each normalized token to a set of document IDs
- stores document metadata in a `docmap`
- tracks token frequencies per document
- saves/loads index data using `pickle`

Search works by tokenizing the query and returning all documents containing any matching tokens.

## Data Format

The movie dataset is located in `data/movies.json` and uses this structure:

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Kaakha..Kaakha: The Police",
      "description": "A badly injured Anbuselvan..."
    }
  ]
}
```

Search results are displayed using the movie `id` and `title` fields.

## Notes

- Run `build` before using `search`, `tf`, `idf`, or `tfidf`.
- If cache files are missing, the CLI will print a load error.
- The search command performs keyword matching on stemmed tokens.

Create a virtual environment at the top level of your project directory:

uv venv

---

Activate the virtual environment:

source .venv/bin/activate

---

In the cli dir we create keyword_search_cli.py

---

load the json using

```
with open('data/movies.json', 'r') as file:
        data = json.load(file)
```

---

break a loop if i>4 in py

for i,val in enumerate(arr):
if i > 4:
break
print(f"{i+1}. {val}")

---

iterate over an array

for item in array:
print(item)

---

remove punctuations from a string

```
    import string

    text = "Hello, world! How's it going?"

    clean = text.translate(str.maketrans('', '', string.punctuation))

    print(clean)

```

---

check if any ele from list a is present in list b

```

a = ["a", "b", "c"]
b = ["e", "c", "f"]

# easiest
if any(x in b for x in a):
    print("Match found")

```

another method

```

if set(a) & set(b):
    print("Match found")

```

---

for substring to match in both arrays

```
a = ["fast"]
b = ["faster"]

if any(x in y for x in a for y in b):
    print("Match found")

```

---

remove common words from array

a = ["a", "b", "c"]
b = ["e", "c", "f"]

# remove common words from two array

common = set(a) & set(b)

a = [x for x in a if x not in common]
b = [x for x in b if x not in common]

print(a) # ['a', 'b']
print(b) # ['e', 'f']

---

Implementing stemming from scratch is a lot of work, so we'll use the nltk.stem library to handle it for us.

Install the nltk library:
uv add nltk==3.9.1

---

🔹 pickle in Python
Used to save Python objects to a file (serialize)
And load them back later
Example
import pickle

data = {"a": 1}

# save

with open("data.pkl", "wb") as f:
pickle.dump(data, f)

# load

with open("data.pkl", "rb") as f:
loaded = pickle.load(f)
