import math

from index import InvertedIndex, BM25_K1, BM25_B
from text import process_text


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()

def bm25search(term: str, limit=5) -> list[tuple[dict, float]]:
    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()

    return movies_index.bm25_search(term, limit)

def bm25_idf_command(term: str) -> float:
    term = process_text(term)[0]
    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()

    return movies_index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    term = process_text(term)[0]
    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()

    return movies_index.get_bm25_tf(doc_id, term, k1, b)

def tf_command(doc_id: int, term: str) -> float:
    term = process_text(term)[0]
    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()
    return movies_index.get_tf(doc_id, term)

def tfidf_command(doc_id: str, term: str) -> float:
    term = process_text(term)[0]
    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()

    tf = movies_index.get_tf(doc_id, term)
    idf = math.log((len(movies_index.docmap) + 1) / (len(movies_index.get_documents(term)) + 1))
    return tf * idf


def idf_command(term: str) -> float:
    term = process_text(term)[0]
    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()

    total_doc_count = len(movies_index.docmap) + 1
    total_match_doc_count = len(movies_index.get_documents(term)) + 1

    return math.log(total_doc_count / total_match_doc_count)


def movie_search(keyword) -> list[dict]:
    keyword_tokens = process_text(keyword)

    movies_index = InvertedIndex()
    try:
        movies_index.load()
    except FileNotFoundError as err:
        print(err)
        exit()

    movies_list: list[dict] = []

    for token in keyword_tokens:
        movies_list.extend(movies_index.get_documents(token))
        if len(movies_list) >= 5:
            break

    return movies_list[:5]
