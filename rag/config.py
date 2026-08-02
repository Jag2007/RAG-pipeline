"""Every setting the pipeline uses, in one place.

Each value can be overridden with an environment variable of the same name,
so you can try a different model without touching the code.
"""
import os

# Ollama runs the models locally, so no API key is needed.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:3b")

# Chunking: roughly a paragraph per chunk, with some overlap so a fact that
# sits on a chunk boundary still appears whole in at least one chunk.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))  # characters

# Retrieval: how many chunks the LLM gets to read.
TOP_K = int(os.getenv("TOP_K", 5))

# Grounding guard: if even the best chunk is less similar than this, the
# question is not covered by the document and we refuse instead of guessing.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", 0.52))

# Where uploaded PDFs and the saved index live.
DATA_DIR = os.getenv("DATA_DIR", "data")
