"""Step 4: store the chunk vectors and find the ones closest to a question.

This is a tiny vector database written with numpy. All vectors live in one
matrix; searching is a single matrix-vector product. That is fast enough for
tens of thousands of chunks and keeps the idea easy to see.
"""
import json
import os

import numpy as np


def normalize(vectors):
    """Scale vectors to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class VectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = None

    def __len__(self):
        return len(self.chunks)

    def add(self, chunks, vectors=None):
        """Add chunks with their vectors. Vectors can be None in keyword-only mode."""
        if vectors is not None:
            vectors = normalize(np.asarray(vectors, dtype=np.float32))
            self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def similarities(self, query_vector):
        """Cosine similarity between the query and every stored chunk."""
        if not self.chunks:
            return np.array([], dtype=np.float32)
        return self.vectors @ normalize(np.asarray(query_vector, dtype=np.float32))

    def search(self, query_vector, k):
        """The k most similar chunks as [(chunk, similarity), ...], best first."""
        scores = self.similarities(query_vector)
        best = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best]

    def remove_source(self, source):
        keep = [i for i, chunk in enumerate(self.chunks) if chunk["source"] != source]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep and self.vectors is not None else None

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)
        if self.vectors is not None:
            np.save(os.path.join(folder, "vectors.npy"), self.vectors)

    def load(self, folder):
        chunks_path = os.path.join(folder, "chunks.json")
        vectors_path = os.path.join(folder, "vectors.npy")
        if os.path.exists(chunks_path):
            with open(chunks_path) as f:
                self.chunks = json.load(f)
        if os.path.exists(vectors_path):
            self.vectors = np.load(vectors_path)
