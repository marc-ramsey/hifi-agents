"""OpenAI-compatible chat client for the loop driver.

Uses httpx so there is no hard dependency on the openai SDK. An optional
``client`` may be injected for tests (any object exposing ``post``).
"""

import httpx

from .config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig, *, client=None):
        self._cfg = config
        self._client = client  # optional httpx.Client-like (tests)

    def chat(self, messages: list[dict]) -> str:
        url = f"{self._cfg.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        payload = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": self._cfg.temperature,
        }

        client = self._client or httpx.Client(timeout=120)
        try:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        finally:
            if self._client is None:
                client.close()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected LLM response: {data!r}") from exc
