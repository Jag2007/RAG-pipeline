"""Query rewriting for keyword-only search (used when there are no embeddings).

Keyword search only finds words that are literally in the document. A user
asking "what is sop" gets nothing if the PDF only says "Statement of Purpose",
and "what is this pdf about" has no useful keywords at all.

So before searching, we ask the LLM to turn the question into search keywords:
the original words plus synonyms and spelled-out abbreviations. The answer is
still checked against the document afterwards, so a bad rewrite can only make
search worse, never put made-up facts in the answer.
"""

REWRITE_PROMPT = """You turn a user's question about a document into keywords for a keyword search engine.
Include the important words of the question, synonyms, spelled-out forms of abbreviations
(for example SOP -> statement of purpose), and words the document itself would likely use.
For questions about what the document is about, add words like introduction, purpose, subject, summary.
Reply with only the keywords on one line, nothing else."""


def rewrite_query(llm, question):
    """The question plus extra search keywords from the LLM."""
    keywords = llm.chat([
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": question},
    ])
    return f"{question} {keywords}"
