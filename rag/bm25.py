"""Keyword search with BM25, written from scratch.

Embeddings are good at meaning but weak at exact tokens such as product codes,
names and numbers ("error E-404", "clause 7.2"). BM25 is the classic search
engine formula that scores a chunk higher when it contains the question's
words, especially rare words, without letting long chunks win just by length.

score(doc) = sum over query words of
    idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
"""
import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in is it its "
    "of on or that the their there this to was what when where which who why will "
    "with you your".split()
)


UNICODE_DASHES = re.compile(r"[\u2010-\u2015\u2212]")


def tokenize(text):
    text = UNICODE_DASHES.sub("-", text.lower())  # "two‑month" written with a non-ASCII hyphen
    return [t for t in TOKEN.findall(text) if t not in STOPWORDS]


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # how quickly repeated words stop adding score
        self.b = b    # how much long documents are penalised
        self.fit([])

    def fit(self, texts):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.docs) if self.docs else 0
        self.doc_freq = Counter(word for doc in self.docs for word in doc)

    def idf(self, word):
        """Rare words matter more than common ones."""
        n, df = len(self.docs), self.doc_freq.get(word, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query):
        words = tokenize(query)
        result = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for word in words:
                tf = doc.get(word, 0)
                if tf:
                    norm = 1 - self.b + self.b * length / self.avg_length
                    score += self.idf(word) * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            result.append(score)
        return result

    def search(self, query, k):
        """Indexes of the k best-matching documents, skipping ones with no match."""
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:k] if scores[i] > 0]
