"""Check the Groq client sends the right request, without calling the real API."""
from rag import groq_llm


class FakeResponse:
    def __init__(self, data, status_code=200, headers=None):
        self.data = data
        self.status_code = status_code
        self.headers = headers or {}

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


def test_gpt_oss_models_use_low_reasoning_effort(monkeypatch):
    calls = capture(monkeypatch, groq_llm, lambda body: {"choices": [{"message": {"content": "ok"}}]})
    groq_llm.GroqLLM(model="openai/gpt-oss-120b", api_key="k").chat([{"role": "user", "content": "q"}])
    assert calls[0]["json"]["reasoning_effort"] == "low"


def test_rate_limit_is_retried(monkeypatch):
    replies = [FakeResponse({}, 429, {"retry-after": "0"}),
               FakeResponse({"choices": [{"message": {"content": "ok"}}]})]
    monkeypatch.setattr(groq_llm.requests, "post", lambda *args, **kwargs: replies.pop(0))
    monkeypatch.setattr(groq_llm.time, "sleep", lambda seconds: None)
    assert groq_llm.GroqLLM(api_key="k").chat([{"role": "user", "content": "q"}]) == "ok"
