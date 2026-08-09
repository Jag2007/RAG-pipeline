from rag.chunker import chunk_page, chunk_pages, split_sentences

TEXT = " ".join(f"This is sentence number {i}." for i in range(40))


def test_split_sentences():
    assert split_sentences("One. Two!\nThree?") == ["One.", "Two!", "Three?"]


def test_chunks_respect_size():
    chunks = chunk_page(TEXT, size=200, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_neighbouring_chunks_overlap():
    chunks = chunk_page(TEXT, size=200, overlap=60)
    for first, second in zip(chunks, chunks[1:]):
        assert first.split()[-1] in second


def test_no_text_is_lost():
    chunks = chunk_page(TEXT, size=200, overlap=0)
    assert " ".join(chunks) == TEXT


def test_very_long_sentence_is_split():
    chunks = chunk_page("word " * 300, size=100, overlap=0)
    assert all(len(c) <= 100 for c in chunks)


def test_chunks_remember_source_and_page():
    pages = [{"page": 1, "text": "Alpha."}, {"page": 2, "text": "Beta."}]
    assert chunk_pages(pages, "doc.pdf") == [
        {"source": "doc.pdf", "page": 1, "text": "Alpha."},
        {"source": "doc.pdf", "page": 2, "text": "Beta."},
    ]
