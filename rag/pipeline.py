"""The whole RAG pipeline in one class.

Indexing (once per PDF):  PDF -> pages -> chunks -> embeddings -> vector store
Answering (per question): question -> hybrid retrieval -> grounding guard
                          -> prompt with numbered sources -> LLM -> citation verification
"""
import os

from .chunker import chunk_pages
from .citations import verify_answer
from .config import DATA_DIR, MIN_SIMILARITY, PROVIDER, TOP_K
from .embedder import OllamaEmbedder
from .groq_llm import GroqLLM
from .llm import OllamaLLM
from .pdf_loader import load_pdf
from .prompt import NOT_FOUND, build_messages
from .retriever import HybridRetriever
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(self, embedder=None, llm=None, data_dir=DATA_DIR,
                 top_k=TOP_K, min_similarity=MIN_SIMILARITY, provider=PROVIDER):
        # embedder and llm can be swapped for fakes in tests
        if provider == "groq":
            self.embedder = None  # Groq has no embedding model: keyword search only
            self.llm = llm or GroqLLM()
        else:
            self.embedder = embedder or OllamaEmbedder()
            self.llm = llm or OllamaLLM()
        self.top_k = top_k
        self.min_similarity = min_similarity
        # vectors from different embedding models cannot be mixed, so each
        # provider keeps its own saved index
        self.index_dir = os.path.join(data_dir, "index", provider) if data_dir else None

        self.store = VectorStore()
        if self.index_dir:
            self.store.load(self.index_dir)
        self.retriever = HybridRetriever(self.store, self.embedder)

    # ---------- indexing ----------
    def add_pdf(self, path, source=None):
        source = source or os.path.basename(path)
        pages = load_pdf(path)
        if not pages:
            raise ValueError("No text found in this PDF. Is it a scanned image?")

        self.store.remove_source(source)  # re-uploading a file replaces it
        chunks = chunk_pages(pages, source)
        vectors = self.embedder.embed_documents([c["text"] for c in chunks]) if self.embedder else None
        self.store.add(chunks, vectors)
        self._changed()
        return {"source": source, "pages": len(pages), "chunks": len(chunks)}

    def remove(self, source):
        self.store.remove_source(source)
        self._changed()

    def documents(self):
        counts = {}
        for chunk in self.store.chunks:
            counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
        return [{"source": s, "chunks": n} for s, n in counts.items()]

    def _changed(self):
        self.retriever.refresh()
        if self.index_dir:
            self.store.save(self.index_dir)

    # ---------- answering ----------
    def ask(self, question):
        results, best_similarity = self.retriever.retrieve(question, self.top_k)

        # Grounding guard: nothing relevant in the document, so do not ask the
        # LLM at all. With embeddings the best chunk must be similar enough;
        # in keyword-only mode at least one chunk must share a keyword.
        too_far = best_similarity is not None and best_similarity < self.min_similarity
        if not results or too_far:
            return {"answer": NOT_FOUND, "sources": [], "cited": [], "grounded": False}

        chunks = [r["chunk"] for r in results]
        answer = self.llm.chat(build_messages(question, chunks))
        if NOT_FOUND.lower().rstrip(".") in answer.lower():
            return {"answer": NOT_FOUND, "sources": [], "cited": [], "grounded": False}

        answer, cited, all_supported = verify_answer(answer, chunks)
        sources = [
            {"n": n, "source": r["chunk"]["source"], "page": r["chunk"]["page"],
             "text": r["chunk"]["text"],
             "similarity": round(r["similarity"], 3) if r["similarity"] is not None else None}
            for n, r in enumerate(results, start=1)
        ]
        # Grounded = every sentence of the answer is backed by a source.
        grounded = bool(cited) and all_supported
        return {"answer": answer, "sources": sources, "cited": cited, "grounded": grounded}
