from rag.bm25 import BM25, tokenize

DOCS = [
    "The battery charges in 4.5 hours with the 4 A charger.",
    "Error E-21 means the speed sensor magnet has moved.",
    "Clean the chain every 300 km.",
]


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert tokenize("What is error E-21?") == ["error", "e-21"]
    assert "4.5" in tokenize("about 4.5 hours")


def test_exact_code_is_found():
    bm25 = BM25()
    bm25.fit(DOCS)
    assert bm25.search("E-21", k=3) == [1]


def test_rare_words_count_more():
    bm25 = BM25()
    bm25.fit(DOCS + ["the battery the battery"])
    assert bm25.idf("chain") > bm25.idf("battery")


def test_no_match_returns_nothing():
    bm25 = BM25()
    bm25.fit(DOCS)
    assert bm25.search("pizza", k=3) == []
