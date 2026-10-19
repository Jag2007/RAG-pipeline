"""Answers from Groq's API, used when hosting (see config.PROVIDER).

Does the same job as llm.py. Groq uses the OpenAI chat format, so our
system/user messages are sent as they are.
"""
import requests

from .config import GROQ_API_KEY, GROQ_CHAT_MODEL

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    def __init__(self, model=GROQ_CHAT_MODEL, api_key=GROQ_API_KEY):
        self.model = model
        self.api_key = api_key

    def chat(self, messages):
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": 0},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
