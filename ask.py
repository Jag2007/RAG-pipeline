"""Ask a PDF questions from the terminal.

    python ask.py path/to/file.pdf "What is the refund policy?"
    python ask.py path/to/file.pdf            # interactive mode
"""
import sys

from rag.pipeline import RAGPipeline


def show(result):
    print("\n" + result["answer"] + "\n")
    for source in result["sources"]:
        mark = "*" if source["n"] in result["cited"] else " "
        similarity = "" if source["similarity"] is None else f"  (similarity {source['similarity']})"
        print(f" {mark}[{source['n']}] {source['source']} p.{source['page']}{similarity}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pipeline = RAGPipeline(data_dir=None)  # in memory only, nothing saved
    info = pipeline.add_pdf(sys.argv[1])
    print(f"Indexed {info['source']}: {info['pages']} pages, {info['chunks']} chunks")

    if len(sys.argv) > 2:
        show(pipeline.ask(" ".join(sys.argv[2:])))
        return
    while True:
        question = input("Question (blank to quit): ").strip()
        if not question:
            break
        show(pipeline.ask(question))


if __name__ == "__main__":
    main()
