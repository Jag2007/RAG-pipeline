"""Step 5: hybrid retrieval, meaning search plus keyword search.

We run both searches and merge their rankings with Reciprocal Rank Fusion
(RRF): a chunk earns 1 / (60 + rank) from each list it appears in. Chunks that
rank well in both lists rise to the top. RRF only uses ranks, so we never have
to compare a cosine similarity with a BM25 score directly.
"""
from collections import defaultdict

import numpy as np

from .bm25 import BM25

RRF_K = 60  # the standard constant from the original RRF paper


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Merge several ranked lists of ids into one, best first."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] += 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self, store, embedder, mode="hybrid"):
        self.store = store
        self.embedder = embedder
        self.mode = mode  # "hybrid", "vector" or "keyword"; the last two exist for evaluation
        self.bm25 = BM25()
        self.refresh()

    def refresh(self):
        """Rebuild the keyword index after chunks are added or removed."""
        self.bm25.fit([chunk["text"] for chunk in self.store.chunks])

    def opening_chunks(self, top_k, per_document=2):
        """Indexes of the first few chunks of every document."""
        seen, order = {}, []
        for i, chunk in enumerate(self.store.chunks):
            seen[chunk["source"]] = seen.get(chunk["source"], 0) + 1
            if seen[chunk["source"]] <= per_document:
                order.append(i)
        return order[:top_k]

    def retrieve(self, question, top_k):
        """Return (results, best_similarity).

        results is a list of {"chunk": ..., "similarity": ...}, best first.
        best_similarity is the highest cosine similarity over all chunks and
        is what the grounding guard checks.

        Without an embedder (keyword-only hosting) only BM25 runs and the
        similarities are None. If no chunk shares a keyword with the question,
        we return the opening chunks of each document instead: that is where a
        document says what it is about, which answers overview questions, and
        the LLM still refuses when they do not contain the answer.
        """
        if len(self.store) == 0:
            return [], 0.0

        if self.embedder is None:
            order = self.bm25.search(question, top_k) or self.opening_chunks(top_k)
            return [{"chunk": self.store.chunks[i], "similarity": None} for i in order], None

        similarities = self.store.similarities(self.embedder.embed_query(question))
        pool = top_k * 4  # look a bit deeper than we need, then fuse
        vector_ranking = list(np.argsort(-similarities)[:pool])
        keyword_ranking = self.bm25.search(question, pool)

        if self.mode == "vector":
            order = vector_ranking
        elif self.mode == "keyword":
            order = keyword_ranking
        else:
            order = reciprocal_rank_fusion([vector_ranking, keyword_ranking])

        results = [
            {"chunk": self.store.chunks[i], "similarity": float(similarities[i])}
            for i in order[:top_k]
        ]
        return results, float(similarities.max())
