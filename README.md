# Hoopla: A Learning Journey into RAG

Welcome to **Hoopla**, a comprehensive learning project designed to explore the fundamentals and advanced concepts of **Retrieval-Augmented Generation (RAG)**. 

This project uses a movie streaming service (Hoopla) as a case study to implement various search and generation techniques, from traditional keyword search to modern multimodal RAG pipelines.

---

## 🎓 Learning Topics

This project is structured as a series of learning modules, each covering a critical component of modern search and RAG systems:

### 1. Traditional Keyword Search
*Found in: `cli/keyword_search_cli.py`, `cli/lib/keyword_search.py`*
- **Inverted Index**: Building the foundational data structure for efficient lookups.
- **Text Preprocessing**: Learning about tokenization, stopword removal, and stemming (using Porter Stemmer).
- **Scoring Algorithms**: Implementing and comparing **TF-IDF** and **BM25**.

### 2. Semantic Search & Embeddings
*Found in: `cli/semantic_search_cli.py`, `cli/lib/semantic_search.py`*
- **Dense Retrieval**: Using `SentenceTransformer` (Bi-Encoders) to convert text into vector embeddings.
- **Similarity Metrics**: Implementing **Cosine Similarity** to find relevant documents.
- **Chunking Strategies**: 
    - **Fixed-size chunking**: Splitting text by word count.
    - **Semantic chunking**: Splitting text based on sentence boundaries to preserve context.
- **Multi-vector Search**: Embedding chunks individually and aggregating scores for better retrieval accuracy.

### 3. Hybrid Search
*Found in: `cli/hybrid_search_cli.py`, `cli/lib/hybrid_search.py`*
- **Fusion Techniques**: Combining the strengths of keyword search (precision) and semantic search (recall).
- **Weighted Search**: Using an `alpha` parameter to balance scores.
- **Reciprocal Rank Fusion (RRF)**: A robust, parameter-free way to merge ranked lists from different sources.

### 4. Retrieval-Augmented Generation (RAG)
*Found in: `cli/augmented_generation_cli.py`, `cli/lib/augmented_generation.py`*
- **Retrieval Pipeline**: Combining RRF hybrid search with advanced generation.
- **Prompt Engineering**: Designing templates for different tasks (summarization, citations, conversational).
- **Query Enhancement**: 
    - **Spell Check**: Fixing typos in user queries using LLMs.
    - **Rewriting**: Transforming vague queries into searchable terms.
    - **Expansion**: Adding related terms to increase recall.
- **Generation**: Integrating with LLMs (Google Gemini) to synthesize answers from retrieved documents.

### 5. Advanced Reranking & LLM Evaluation
*Found in: `cli/hybrid_search_cli.py`, `cli/lib/llm.py`*
- **Cross-Encoders**: Using more powerful models to score query-document pairs more accurately than bi-encoders.
- **LLM Reranking**: Utilizing high-reasoning models to evaluate document relevance in **batch** or **individual** modes.
- **LLM-as-a-Judge**: Using an LLM to evaluate the relevance of search results on a 0-3 scale.

### 6. Multimodal Search & Processing
*Found in: `cli/multimodal_search_cli.py`, `cli/describe_image_cli.py`, `cli/lib/multimodal_search.py`*
- **CLIP (Contrastive Language-Image Pre-training)**: Using a shared embedding space for images and text retrieval.
- **Multimodal Query Expansion**: Using Gemini to rewrite a text query based on visual context from an image.
- **Image-to-Text Retrieval**: Finding movies based on visual queries (images).

### 7. Evaluation
*Found in: `cli/evaluation_cli.py`, `cli/lib/evaluation.py`*
- **Search Metrics**: Understanding and calculating **Precision@k**, **Recall@k**, and **F1-Score**.
- **Golden Datasets**: Using a curated set of ground-truth results to measure system performance.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.12+
- `uv` (recommended) or `pip`

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd hoopla

# Install dependencies
uv sync
```

### Environment Configuration
Create a `.env` file in the root directory (this file is ignored by git):
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

---

## 🚀 Usage

The project provides several CLI tools to explore different modules:

### Keyword Search
```bash
python cli/keyword_search_cli.py build
python cli/keyword_search_cli.py bm25_search "scary bear movie"
```

### Semantic Search
```bash
python cli/semantic_search_cli.py embed_movies
python cli/semantic_search_cli.py search "movies about london"
```

### Hybrid & Advanced Search
```bash
# Weighted hybrid search
python cli/hybrid_search_cli.py weighted-search "paddington" --alpha 0.3

# RRF search with query expansion and reranking
python cli/hybrid_search_cli.py rrf-search "adventure films" --enhance expand --rerank-method cross_encoder --evaluate
```

### RAG Search
```bash
python cli/augmented_generation_cli.py rag "Tell me about Paddington bear"
python cli/augmented_generation_cli.py citations "Which movies feature bears?"
```

### Multimodal Search
```bash
# Search for movies using an image
python cli/multimodal_search_cli.py image_search data/paddington.jpeg

# Expand a text query using visual context from an image
python cli/describe_image_cli.py --image data/paddington.jpeg --query "What is this movie about?"
```

### Evaluation
```bash
python cli/evaluation_cli.py --limit 5
```

---

## 📂 Project Structure
- `cli/`: Command-line interface entry points.
- `cli/lib/`: Core implementation logic for search, LLM integration, and RAG.
- `data/`: Datasets including `movies.json` and `golden_dataset.json`.
- `cache/`: Persistent storage for indices and embeddings.

---

## 📝 Notes for Review
- The project demonstrates the full lifecycle of a RAG application.
- Emphasis is placed on **modularity**, allowing for independent testing of search vs. generation.
- Check `cli/lib/llm.py` for LLM-based query enhancement and reranking strategies.
