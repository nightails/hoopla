from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import load_movies


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents: list[dict] = documents
        self.texts: list[str] = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, path: str):
        image = Image.open(path)
        return self.model.encode([image])[0]

    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)
        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append(
                {
                    "score": similarity,
                    "id": self.documents[i]["id"],
                    "title": self.documents[i]["title"],
                    "description": self.documents[i]["description"],
                }
            )
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:5]


def image_search_command(image_path: str):
    movies = load_movies()
    search = MultimodalSearch(movies)
    return search.search_with_image(image_path)
