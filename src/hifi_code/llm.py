"""OpenAI-compatible chat client for the loop driver.

Uses httpx so there is no hard dependency on the openai SDK. An optional
``client`` may be injected for tests (any object exposing an async ``post``).
"""

import httpx

from .config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig, *, client=None):
        self._cfg = config
        self._client = client  # optional httpx.AsyncClient-like (tests)

    async def chat(self, messages: list[dict]) -> str:
        url = f"{self._cfg.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        payload = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": self._cfg.temperature,
        }

        client = self._client or httpx.AsyncClient(timeout=120)
        owned = self._client is None
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            try:
                message = data["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"unexpected LLM response: {data!r}") from exc
        finally:
            if owned:
                await client.aclose()
        # Reasoning models often leave ``content`` empty and put the answer in
        # ``reasoning_content``; fall back so the loop isn't starved of output.
        content = message.get("content") or message.get("reasoning_content") or ""
        return content
