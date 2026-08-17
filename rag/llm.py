"""Talk to the local chat model through Ollama."""
import requests

from .config import CHAT_MODEL, OLLAMA_URL


class OllamaLLM:
    def __init__(self, model=CHAT_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url

    def chat(self, messages):
        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                # temperature 0 = always pick the most likely word, so the
                # model sticks to the sources instead of getting creative
                "options": {"temperature": 0},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
