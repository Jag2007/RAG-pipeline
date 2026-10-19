"""The same two model jobs as embedder.py and llm.py, done by Google's Gemini API.

Used when the app is hosted somewhere without room for local models (Render's
free plan has 512 MB of memory). Everything else in the pipeline (chunking,
BM25, fusion, the grounding guard, citation checks) stays exactly the same.
"""
import numpy as np
import requests

from .config import GEMINI_API_KEY, GEMINI_CHAT_MODEL, GEMINI_EMBED_MODEL

API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
BATCH_SIZE = 100  # the most texts the API embeds in one request
DIMENSIONS = 768  # same size as nomic-embed-text, smaller to store than the default 3072


def _post(path, body, api_key):
    response = requests.post(
        f"{API_URL}/{path}",
        headers={"x-goog-api-key": api_key},
        json=body,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


class GeminiEmbedder:
    def __init__(self, model=GEMINI_EMBED_MODEL, api_key=GEMINI_API_KEY):
        self.model = model
        self.api_key = api_key

    # The task type plays the same role as nomic's search_document / search_query prefixes.
    def _embed(self, texts, task_type):
        requests_body = [
            {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
             "taskType": task_type, "outputDimensionality": DIMENSIONS}
            for t in texts
        ]
        data = _post(f"{self.model}:batchEmbedContents", {"requests": requests_body}, self.api_key)
        return [e["values"] for e in data["embeddings"]]

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed(texts[start:start + BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text):
        return np.array(self._embed([text], "RETRIEVAL_QUERY")[0], dtype=np.float32)


class GeminiLLM:
    def __init__(self, model=GEMINI_CHAT_MODEL, api_key=GEMINI_API_KEY):
        self.model = model
        self.api_key = api_key

    def chat(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] == "user"]
        data = _post(f"{self.model}:generateContent", {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0},
        }, self.api_key)
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
