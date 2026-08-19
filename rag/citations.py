"""Step 7: check every sentence of the answer against the sources.

Small local models often state the right fact but forget the [n] citation, or
cite the wrong number. So we do not trust the model's citations blindly:

1. Split the answer into sentences.
2. For each sentence, measure how much of its wording appears in each source
   (the share of the sentence's words that the source also contains).
3. Keep the model's citation if that source really supports the sentence,
   otherwise replace it with the source that supports it best.
4. A sentence that no source supports is marked unsupported.

This is simple word overlap, not another AI model, so it is fast and easy to
reason about.
"""
import re

from .bm25 import tokenize

SUPPORT_THRESHOLD = 0.6  # at least 60% of a sentence's words must be in the source
CITATION = re.compile(r"\[(\d+)\]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def support(sentence, source_text):
    """Share of the sentence's words that also appear in the source (0 to 1)."""
    words = set(tokenize(CITATION.sub("", sentence)))
    if not words:
        return 0.0
    return len(words & set(tokenize(source_text))) / len(words)


def verify_answer(answer, chunks):
    """Return (answer_with_fixed_citations, cited_numbers, all_sentences_supported)."""
    fixed, cited, all_supported = [], set(), True

    for sentence in SENTENCE_END.split(answer.strip()):
        if not sentence:
            continue
        text = CITATION.sub("", sentence).strip()
        if not tokenize(text):  # nothing to check, e.g. a stray "[1]"
            continue

        scores = [support(text, chunk["text"]) for chunk in chunks]
        claimed = [int(n) for n in CITATION.findall(sentence) if 1 <= int(n) <= len(chunks)]
        good = [n for n in claimed if scores[n - 1] >= SUPPORT_THRESHOLD]
        if not good:
            best = max(range(len(chunks)), key=lambda i: scores[i])
            if scores[best] >= SUPPORT_THRESHOLD:
                good = [best + 1]

        if good:
            cited.update(good)
            ending = text[-1] if text[-1] in ".!?" else ""
            body = (text[:-1] if ending else text).rstrip()
            fixed.append(f"{body} {''.join(f'[{n}]' for n in good)}{ending}")
        else:
            all_supported = False
            fixed.append(text)

    return " ".join(fixed), sorted(cited), all_supported
