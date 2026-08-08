"""Step 2: split each page into overlapping chunks.

A whole page is too much text to match a question against, and a single
sentence is too little context, so we group sentences into chunks of about
CHUNK_SIZE characters. Neighbouring chunks share CHUNK_OVERLAP characters so a
fact that falls on a boundary is not cut in half.

Chunks never cross a page, which keeps every citation exact.
"""
import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    text = re.sub(r"\s+", " ", text)  # line breaks inside a paragraph become spaces
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def split_long(sentence, size):
    """Break a sentence longer than `size` into word-aligned pieces."""
    pieces, current = [], ""
    for word in sentence.split():
        if current and len(current) + len(word) + 1 > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def overlap_tail(text, overlap):
    """The last `overlap` characters of text, starting on a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_page(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one page of text into a list of chunk strings."""
    sentences = []
    for sentence in split_sentences(text):
        sentences.extend(split_long(sentence, size) if len(sentence) > size else [sentence])

    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            chunks.append(current)
            current = overlap_tail(current, overlap)
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages, source, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Turn load_pdf() output into chunk dicts that remember where they came from."""
    chunks = []
    for page in pages:
        for text in chunk_page(page["text"], size, overlap):
            chunks.append({"source": source, "page": page["page"], "text": text})
    return chunks
