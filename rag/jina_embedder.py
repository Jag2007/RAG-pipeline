"""Embeddings from Jina AI's API, used when hosting (see config.PROVIDER).

Does the same job as embedder.py, so the rest of the pipeline does not change.
"""
import numpy as np
import requests

from .config import JINA_API_KEY, JINA_EMBED_MODEL

API_URL = "https://api.jina.ai/v1/embeddings"
BATCH_SIZE = 64
DIMENSIONS = 768  # same size as nomic-embed-text


class JinaEmbedder:
    def __init__(self, model=JINA_EMBED_MODEL, api_key=JINA_API_KEY):
        self.model = model
        self.api_key = api_key

    # The task plays the same role as nomic's search_document / search_query prefixes.
    def _embed(self, texts, task):
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "task": task, "dimensions": DIMENSIONS, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed(texts[start:start + BATCH_SIZE], "retrieval.passage"))
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text):
        return np.array(self._embed([text], "retrieval.query")[0], dtype=np.float32)
