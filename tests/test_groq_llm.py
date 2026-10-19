"""Check the Groq client sends the right request, without calling the real API."""
from rag import groq_llm


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


def test_groq_sends_messages_unchanged(monkeypatch):
    calls = capture(monkeypatch, groq_llm, lambda body: {"choices": [{"message": {"content": " Hi [1]. "}}]})
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]
    answer = groq_llm.GroqLLM(api_key="k").chat(messages)

    assert answer == "Hi [1]."
    assert calls[0]["json"]["messages"] == messages
    assert calls[0]["json"]["temperature"] == 0
