"""Stand-ins for the Ollama models so tests run fast and offline."""
import hashlib

import numpy as np

from rag.bm25 import tokenize
from rag.query_rewrite import REWRITE_PROMPT

DIM = 64


class FakeEmbedder:
    """Bag-of-words vectors: texts that share words get similar vectors."""

    def _vector(self, text):
        vector = np.zeros(DIM, dtype=np.float32)
        for word in tokenize(text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1
        return vector

    def embed_documents(self, texts):
        return np.array([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


class FakeLLM:
    """Returns a canned reply and remembers the last messages it was sent.

    Query-rewrite requests get `keywords` back instead of the reply.
    """

    def __init__(self, reply, keywords=""):
        self.reply = reply
        self.keywords = keywords
        self.last_messages = None

    def chat(self, messages):
        if messages[0]["content"] == REWRITE_PROMPT:
            return self.keywords
        self.last_messages = messages
        return self.reply
