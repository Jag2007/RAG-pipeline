"""Answers from Groq's API, used when hosting (see config.PROVIDER).

Does the same job as llm.py. Groq uses the OpenAI chat format, so our
system/user messages are sent as they are.
"""
import time

import requests

from .config import GROQ_API_KEY, GROQ_CHAT_MODEL

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_ATTEMPTS = 3
MAX_WAIT_SECONDS = 30


class GroqLLM:
    def __init__(self, model=GROQ_CHAT_MODEL, api_key=GROQ_API_KEY):
        self.model = model
        self.api_key = api_key

    def chat(self, messages):
        body = {"model": self.model, "messages": messages, "temperature": 0}
        if self.model.startswith("openai/gpt-oss"):
            # these models think before answering; "low" keeps answers fast
            # and inside the free tier's tokens-per-minute limit
            body["reasoning_effort"] = "low"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
                timeout=120,
            )
            # 429 = free-tier rate limit: wait as long as Groq asks, then retry
            if response.status_code == 429 and attempt < MAX_ATTEMPTS:
                wait = float(response.headers.get("retry-after", 5))
                time.sleep(min(wait, MAX_WAIT_SECONDS))
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
