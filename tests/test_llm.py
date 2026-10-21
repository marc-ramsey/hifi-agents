"""Tests for the OpenAI-compatible LLM client."""

from hifi_code.config import LLMConfig
from hifi_code.llm import LLMClient


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    async def post(self, url, headers=None, json=None):
        return _FakeResp(self._payload)

    async def aclose(self):
        pass


async def _chat(payload):
    llm = LLMClient(LLMConfig(), client=_FakeClient(payload))
    try:
        return await llm.chat([{"role": "user", "content": "hi"}])
    finally:
        await llm._client.aclose()


async def test_chat_returns_content():
    payload = {"choices": [{"message": {"content": '{"action": "finish"}'}}]}
    assert await _chat(payload) == '{"action": "finish"}'


async def test_chat_falls_back_to_reasoning_content():
    # Reasoning models often leave ``content`` empty and put the answer in
    # ``reasoning_content``; the client must not be starved of output.
    payload = {
        "choices": [{
            "message": {
                "content": "",
                "reasoning_content": '{"action": "edit"}',
            },
        }],
    }
    assert await _chat(payload) == '{"action": "edit"}'


async def test_chat_raises_on_malformed_payload():
    payload = {"choices": []}
    raised = False
    try:
        await _chat(payload)
    except RuntimeError as exc:
        raised = "unexpected LLM response" in str(exc)
    assert raised
