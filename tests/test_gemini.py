"""Check the Gemini clients send the right requests, without calling the real API."""
from rag import gemini


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def capture(monkeypatch, reply):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(reply(json))

    monkeypatch.setattr(gemini.requests, "post", fake_post)
    return calls


def test_embed_documents_batches_and_sets_task_type(monkeypatch):
    calls = capture(monkeypatch, lambda body: {"embeddings": [{"values": [1.0, 0.0]} for _ in body["requests"]]})
    vectors = gemini.GeminiEmbedder(api_key="k").embed_documents(["t"] * 150)

    assert vectors.shape == (150, 2)
    assert len(calls) == 2  # 100 + 50
    assert calls[0]["headers"] == {"x-goog-api-key": "k"}
    assert calls[0]["json"]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"


def test_embed_query_uses_query_task_type(monkeypatch):
    calls = capture(monkeypatch, lambda body: {"embeddings": [{"values": [0.5, 0.5]}]})
    gemini.GeminiEmbedder(api_key="k").embed_query("question")
    assert calls[0]["json"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_chat_sends_system_prompt_separately(monkeypatch):
    calls = capture(monkeypatch, lambda body: {"candidates": [{"content": {"parts": [{"text": " Hi [1]. "}]}}]})
    answer = gemini.GeminiLLM(api_key="k").chat([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ])

    assert answer == "Hi [1]."
    body = calls[0]["json"]
    assert body["systemInstruction"]["parts"][0]["text"] == "rules"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "question"}]}]
    assert body["generationConfig"]["temperature"] == 0
