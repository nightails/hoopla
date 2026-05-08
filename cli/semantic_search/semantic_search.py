from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

def verify_model():
    semantic_serach = SemanticSearch()
    print(f"Model loaded: {semantic_serach.model}")
    print(f"Max sequence length: {semantic_serach.model.max_seq_length}")
