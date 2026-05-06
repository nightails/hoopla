import string

from nltk.stem import PorterStemmer

def process_text(t: str) -> list[str]:
    t = remove_punctuation(t.lower())
    tokens = tokenize_string(t)
    tokens = remove_stop_words(tokens)

    stemmer = PorterStemmer()
    for i in range(len(tokens)):
        tokens[i] = stemmer.stem(tokens[i])

    return tokens

def remove_punctuation(t: str) -> str:
    punc_map = str.maketrans("", "", string.punctuation)
    return t.translate(punc_map).lower()

def tokenize_string(t: str) -> list[str]:
    t = remove_punctuation(t)
    tokens = t.split()
    return tokens

def remove_stop_words(tokens: list[str]) -> list[str]:
    with open("data/stopwords.txt", "r", encoding="utf-8") as f:
        stop_words = f.read().splitlines()
    return [t for t in tokens if t not in stop_words]
