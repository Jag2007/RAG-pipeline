"""Step 6: build the prompt that keeps the model grounded.

The retrieved chunks are numbered [1], [2], ... and the model is told to answer
only from them and cite those numbers. That makes every claim checkable.
"""

NOT_FOUND = "I could not find this in the document."

SYSTEM_PROMPT = f"""You answer questions about a document using ONLY the numbered sources you are given.
Rules:
1. Use only facts written in the sources. Never use outside knowledge.
2. End every sentence with the number of the source it came from, like [2].
3. If the sources do not answer the question, reply with exactly: {NOT_FOUND}
4. Answer in one to three short sentences.

Example:
Sources: [1] The warranty lasts 2 years. [2] Shipping is free over $50.
Question: How long is the warranty?
Answer: The warranty lasts 2 years [1]."""


def build_messages(question, chunks):
    """Chat messages for the model: the rules, then the sources and the question."""
    sources = "\n\n".join(
        f"[{n}] ({chunk['source']}, page {chunk['page']})\n{chunk['text']}"
        for n, chunk in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}\nAnswer:"},
    ]
