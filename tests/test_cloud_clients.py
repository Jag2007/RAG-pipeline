"""Check the Jina and Groq clients send the right requests, without calling the real APIs."""
from rag import groq_llm, jina_embedder


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def capture(monkeypatch, module, reply):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(reply(json))

    monkeypatch.setattr(module.requests, "post", fake_post)
    return calls


def embeddings_reply(body):
    return {"data": [{"embedding": [1.0, 0.0]} for _ in body["input"]]}


def test_jina_batches_documents(monkeypatch):
    calls = capture(monkeypatch, jina_embedder, embeddings_reply)
    vectors = jina_embedder.JinaEmbedder(api_key="k").embed_documents(["t"] * 100)

    assert vectors.shape == (100, 2)
    assert len(calls) == 2  # 64 + 36
    assert calls[0]["headers"] == {"Authorization": "Bearer k"}
    assert calls[0]["json"]["task"] == "retrieval.passage"


def test_jina_query_uses_query_task(monkeypatch):
    calls = capture(monkeypatch, jina_embedder, embeddings_reply)
    jina_embedder.JinaEmbedder(api_key="k").embed_query("question")
    assert calls[0]["json"]["task"] == "retrieval.query"


def test_groq_sends_messages_unchanged(monkeypatch):
    calls = capture(monkeypatch, groq_llm, lambda body: {"choices": [{"message": {"content": " Hi [1]. "}}]})
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]
    answer = groq_llm.GroqLLM(api_key="k").chat(messages)

    assert answer == "Hi [1]."
    assert calls[0]["json"]["messages"] == messages
    assert calls[0]["json"]["temperature"] == 0
