"""Web server: upload PDFs and ask questions in the browser.

Run:  python app.py   then open http://localhost:5000
"""
import os

import requests
from flask import Flask, jsonify, request, send_from_directory

from rag.config import (CHAT_MODEL, DATA_DIR, EMBED_MODEL, GEMINI_API_KEY, GEMINI_CHAT_MODEL,
                        GEMINI_EMBED_MODEL, OLLAMA_URL, PROVIDER)
from rag.pipeline import RAGPipeline

UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload limit

pipeline = RAGPipeline()


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    if PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            return jsonify({"ok": False, "error": "GEMINI_API_KEY is not set."})
        return jsonify({"ok": True, "embed_model": GEMINI_EMBED_MODEL, "chat_model": GEMINI_CHAT_MODEL})
    try:
        installed = [m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()["models"]]
    except requests.RequestException:
        return jsonify({"ok": False, "error": "Ollama is not running. Start it with: ollama serve"})
    missing = [m for m in (EMBED_MODEL, CHAT_MODEL) if not any(name.startswith(m) for name in installed)]
    if missing:
        return jsonify({"ok": False, "error": "Missing models. Run: " + "; ".join(f"ollama pull {m}" for m in missing)})
    return jsonify({"ok": True, "embed_model": EMBED_MODEL, "chat_model": CHAT_MODEL})


@app.get("/api/documents")
def documents():
    return jsonify(pipeline.documents())


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please choose a PDF file."}), 400

    name = os.path.basename(file.filename)
    path = os.path.join(UPLOAD_DIR, name)
    file.save(path)
    try:
        info = pipeline.add_pdf(path, name)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except requests.RequestException:
        return jsonify({"error": "Could not reach the embedding model."}), 503
    return jsonify(info)


@app.delete("/api/documents/<path:source>")
def delete_document(source):
    pipeline.remove(source)
    return jsonify({"ok": True})


@app.post("/api/ask")
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Please type a question."}), 400
    if not pipeline.documents():
        return jsonify({"error": "Upload a PDF first."}), 400
    try:
        return jsonify(pipeline.ask(question))
    except requests.RequestException:
        return jsonify({"error": "Could not reach the answer model."}), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)
