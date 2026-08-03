"""Step 1: read a PDF and return its text, page by page.

We keep the page number with the text so answers can cite "page 4".
"""
import re

from pypdf import PdfReader


def clean_text(text):
    """Tidy the raw text that comes out of a PDF."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join words split by a hyphen at a line end
    text = re.sub(r"[ \t]+", " ", text)            # collapse runs of spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)       # at most one blank line
    return text.strip()


def load_pdf(path):
    """Return [{"page": 1, "text": "..."}, ...] for every page that has text."""
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": number, "text": text})
    return pages
