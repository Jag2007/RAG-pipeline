"""Step 3: turn text into vectors (embeddings).

Texts with similar meaning get vectors that point in similar directions, which
is what lets us find "annual leave" when the user asks about "holidays".

We call a local embedding model through Ollama's HTTP API.
"""
import numpy as np
import requests

from .config import EMBED_MODEL, OLLAMA_URL

BATCH_SIZE = 32


class OllamaEmbedder:
    def __init__(self, model=EMBED_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url

    def _embed(self, texts):
        response = requests.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    # nomic-embed-text is trained with these two prefixes: one for the stored
    # documents and one for the user's question. Using them improves retrieval.
    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            vectors.extend(self._embed([f"search_document: {t}" for t in batch]))
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text):
        return np.array(self._embed([f"search_query: {text}"])[0], dtype=np.float32)
