from rag.citations import support, verify_answer

CHUNKS = [
    {"text": "The frame is covered for 5 years against defects."},
    {"text": "The battery charges fully in 4.5 hours."},
]


def test_support_measures_word_overlap():
    assert support("battery charges 4.5 hours", CHUNKS[1]["text"]) == 1.0
    assert support("pizza delivery", CHUNKS[1]["text"]) == 0.0


def test_correct_citation_is_kept():
    answer, cited, ok = verify_answer("The battery charges in 4.5 hours [2].", CHUNKS)
    assert answer == "The battery charges in 4.5 hours [2]."
    assert cited == [2] and ok


def test_wrong_citation_is_fixed():
    answer, cited, ok = verify_answer("The frame is covered for 5 years [2].", CHUNKS)
    assert answer == "The frame is covered for 5 years [1]."
    assert cited == [1] and ok


def test_missing_citation_is_added():
    answer, cited, _ = verify_answer("The battery charges fully in 4.5 hours.", CHUNKS)
    assert answer.endswith("[2].")
    assert cited == [2]


def test_unsupported_sentence_is_flagged():
    _, cited, ok = verify_answer("The bike costs 90,000 rupees.", CHUNKS)
    assert cited == [] and not ok
