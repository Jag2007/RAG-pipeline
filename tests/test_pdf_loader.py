from conftest import SAMPLE_PDF

from rag.pdf_loader import clean_text, load_pdf


def test_clean_text_joins_hyphenated_words_and_spaces():
    assert clean_text("bat-\ntery   pack") == "battery pack"


def test_clean_text_limits_blank_lines():
    assert clean_text("one\n\n\n\ntwo") == "one\n\ntwo"


def test_load_pdf_keeps_page_numbers():
    pages = load_pdf(SAMPLE_PDF)
    assert [p["page"] for p in pages] == [1, 2, 3, 4, 5, 6]
    assert "ERROR CODES" in pages[3]["text"]
