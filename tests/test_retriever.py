from fakes import FakeEmbedder

from rag.retriever import HybridRetriever, reciprocal_rank_fusion
from rag.vector_store import VectorStore


def test_rrf_rewards_items_ranked_well_in_both_lists():
    # "b" is second in both lists, so it beats "a" and "c" which each top only one
    assert reciprocal_rank_fusion([["a", "b", "x"], ["c", "b", "y"]])[0] == "b"


def build(texts):
    embedder = FakeEmbedder()
    store = VectorStore()
    chunks = [{"source": "d.pdf", "page": i + 1, "text": t} for i, t in enumerate(texts)]
    store.add(chunks, embedder.embed_documents(texts))
    return HybridRetriever(store, embedder)


def test_retrieve_finds_relevant_chunk():
    retriever = build(["warranty lasts five years", "charge the battery overnight", "clean the chain"])
    results, best = retriever.retrieve("how long is the warranty", top_k=2)
    assert results[0]["chunk"]["text"] == "warranty lasts five years"
    assert 0 < best <= 1


def test_empty_store_returns_nothing():
    assert build([]).retrieve("anything", top_k=3) == ([], 0.0)
