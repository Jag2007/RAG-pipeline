import numpy as np

from rag.vector_store import VectorStore


def make_store():
    store = VectorStore()
    chunks = [{"source": "a.pdf", "page": 1, "text": "x"},
              {"source": "a.pdf", "page": 2, "text": "y"},
              {"source": "b.pdf", "page": 1, "text": "z"}]
    store.add(chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 5]])
    return store


def test_search_returns_most_similar_first():
    results = make_store().search([0.1, 0.9, 0], k=2)
    assert results[0][0]["text"] == "y"
    assert len(results) == 2


def test_similarity_ignores_vector_length():
    # [0, 0, 5] and [0, 0, 1] point the same way, so similarity is 1
    assert make_store().search([0, 0, 1], k=1)[0][1] == np.float32(1.0)


def test_remove_source():
    store = make_store()
    store.remove_source("a.pdf")
    assert [c["source"] for c in store.chunks] == ["b.pdf"]
    assert store.vectors.shape == (1, 3)


def test_save_and_load(tmp_path):
    make_store().save(tmp_path)
    loaded = VectorStore()
    loaded.load(tmp_path)
    assert len(loaded) == 3
    assert loaded.search([1, 0, 0], k=1)[0][0]["text"] == "x"


def test_empty_store():
    assert VectorStore().search([1, 0, 0], k=3) == []
