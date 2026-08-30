"""Measure how well the pipeline finds and uses the right part of a PDF.

    python eval/evaluate.py              # retrieval + answer evaluation
    python eval/evaluate.py --retrieval  # retrieval only (fast, no chat model)

Retrieval: for each question we know a phrase ("evidence") that must be in the
right chunk. We check where that chunk lands in the ranking for vector-only,
keyword-only (BM25) and hybrid search:
  hit@k = share of questions whose evidence is in the top k chunks
  MRR   = average of 1 / rank of the first correct chunk (1.0 = always first)

Answers: for answerable questions we check the answer cites a source that
contains the evidence. For questions the PDF cannot answer we check the
pipeline refuses instead of inventing something.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline  # noqa: E402
from rag.prompt import NOT_FOUND  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH = 10  # how deep to look when computing ranks


def first_hit_rank(results, evidence):
    for rank, result in enumerate(results, start=1):
        if evidence.lower() in result["chunk"]["text"].lower():
            return rank
    return None


def evaluate_retrieval(pipeline, questions):
    report = {}
    for mode in ("vector", "keyword", "hybrid"):
        retriever = HybridRetriever(pipeline.store, pipeline.embedder, mode=mode)
        ranks = [first_hit_rank(retriever.retrieve(q["question"], DEPTH)[0], q["evidence"]) for q in questions]
        report[mode] = {
            "hit@1": sum(r == 1 for r in ranks if r) / len(ranks),
            "hit@3": sum(r <= 3 for r in ranks if r) / len(ranks),
            "hit@5": sum(r <= 5 for r in ranks if r) / len(ranks),
            "mrr": sum(1 / r for r in ranks if r) / len(ranks),
        }
    return report


def evaluate_answers(pipeline, answerable, unanswerable):
    correct_citation, latencies = 0, []
    for q in answerable:
        start = time.time()
        result = pipeline.ask(q["question"])
        latencies.append(time.time() - start)
        cited_texts = [s["text"].lower() for s in result["sources"] if s["n"] in result["cited"]]
        ok = any(q["evidence"].lower() in text for text in cited_texts)
        correct_citation += ok
        print(f"  {'ok ' if ok else 'MISS'} {q['question']}\n       -> {result['answer'][:110]}")

    refused = 0
    for question in unanswerable:
        result = pipeline.ask(question)
        did_refuse = NOT_FOUND in result["answer"] or not result["grounded"]
        refused += did_refuse
        print(f"  {'ok ' if did_refuse else 'MISS'} {question}\n       -> {result['answer'][:110]}")

    return {
        "cited_correct_source": correct_citation / len(answerable),
        "refused_out_of_scope": refused / len(unanswerable),
        "avg_seconds_per_answer": sum(latencies) / len(latencies),
    }


def main():
    with open(os.path.join(HERE, "questions.json")) as f:
        data = json.load(f)

    pipeline = RAGPipeline(data_dir=None)
    info = pipeline.add_pdf(os.path.join(HERE, data["pdf"]))
    print(f"Indexed {info['source']}: {info['pages']} pages, {info['chunks']} chunks\n")

    print("Retrieval")
    print(f"  {'mode':<8} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6}")
    for mode, m in evaluate_retrieval(pipeline, data["answerable"]).items():
        print(f"  {mode:<8} {m['hit@1']:>6.0%} {m['hit@3']:>6.0%} {m['hit@5']:>6.0%} {m['mrr']:>6.2f}")

    if "--retrieval" in sys.argv:
        return

    print("\nAnswers")
    m = evaluate_answers(pipeline, data["answerable"], data["unanswerable"])
    print(f"\n  answers citing the correct source:  {m['cited_correct_source']:.0%}")
    print(f"  out-of-scope questions refused:     {m['refused_out_of_scope']:.0%}")
    print(f"  average time per answer:            {m['avg_seconds_per_answer']:.1f}s")


if __name__ == "__main__":
    main()
