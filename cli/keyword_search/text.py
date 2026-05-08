import string

from nltk.stem import PorterStemmer


def process_text(t: str) -> list[str]:
    t = remove_punctuation(t.lower())
    tokens = tokenize_string(t)
    tokens = remove_stop_words(tokens)

    stemmer = PorterStemmer()

    return [stemmer.stem(token) for token in tokens]


def remove_punctuation(t: str) -> str:
    punc_map = str.maketrans("", "", string.punctuation)
    return t.translate(punc_map).lower()


def tokenize_string(t: str) -> list[str]:
    t = remove_punctuation(t)
    return t.split()


def remove_stop_words(tokens: list[str]) -> list[str]:
    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        stop_words = set(file.read().splitlines())

    return [token for token in tokens if token not in stop_words]
