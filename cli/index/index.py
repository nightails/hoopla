from cli.text import process_text


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, str] = {}

    def __add_document(self, doc_id: int, content: str):
        content_tokens = process_text(content)
        for token in content_tokens:
            self.index[token] = self.index.get(token, set()).union({doc_id})

    def get_documents(self, term: str) -> list:
        term = term.lower()
        doc_ids = list(self.index.get(term, set()))
        return sorted(doc_ids)

    def build(self):
        pass