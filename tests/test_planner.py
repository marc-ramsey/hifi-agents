"""Tests for the coding loop (planner) with fakes for LLM / ast-bro / verify."""

import json

import pytest

from hifi_code.config import Settings
from hifi_code.editor import Editor
from hifi_code.planner import Planner
from hifi_code.verify import Verifier


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


class FakeAstBro:
    async def digest(self, paths=None, **kwargs):
        return {"digest": "seed"}

    async def map(self, paths=None, **kwargs):
        return {"files": []}

    async def show(self, path=None, **kwargs):
        return {"symbols": [{"name": "f"}]}

    async def call(self, name, arguments=None):
        return {"tool": name, "args": arguments}


@pytest.fixture
def planner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(max_steps=40, sandbox_enabled=True, denylist=[])
    llm = FakeLLM([])
    astbro = FakeAstBro()
    editor = Editor()
    verifier = Verifier()
    p = Planner(settings, llm, astbro, editor, verifier)
    return p, llm


async def test_seed_returns_digest(planner):
    p, _ = planner
    await p.seed("build X", ".")
    assert p.state["map"] == {"digest": "seed"}
    assert p.goal == "build X"


async def test_finish_action(planner):
    p, llm = planner
    llm.responses = [json.dumps({"action": "finish", "reason": "done"})]
    await p.seed("g", ".")
    result = await p.tick()
    assert result["action"] == "finish"
    assert p.step == 1
    assert llm.calls == 1


async def test_navigate_dispatch(planner):
    p, llm = planner
    llm.responses = [json.dumps({"action": "navigate", "tool": "map", "args": {}})]
    await p.seed("g", ".")
    result = await p.tick()
    assert result == {"files": []}


async def test_edit_applies_then_verify(planner, tmp_path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n")
    p, llm = planner
    decision = json.dumps({
        "action": "edit",
        "edits": [{"path": str(target), "old": "x = 1", "new": "x = 2"}],
    })
    llm.responses = [decision, json.dumps({"action": "finish", "reason": "done"})]
    await p.seed("g", ".")

    r1 = await p.tick()
    assert r1["edits"][0]["status"] == "applied"
    assert target.read_text() == "x = 2\n"

    r2 = await p.tick()
    assert r2["action"] == "finish"
    assert r2["ok"] is True


async def test_edit_blocked_by_denylist(planner, tmp_path):
    target = tmp_path / "secret.py"
    target.write_text("x = 1\n")
    p, llm = planner
    p.settings.denylist = [str(tmp_path / "secret.py")]
    decision = json.dumps({
        "action": "edit",
        "edits": [{"path": str(target), "old": "x = 1", "new": "x = 2"}],
    })
    llm.responses = [decision, json.dumps({"action": "finish", "reason": "done"})]
    await p.seed("g", ".")

    r1 = await p.tick()
    assert r1["edits"][0]["status"] == "blocked"
    assert target.read_text() == "x = 1\n"  # untouched


async def test_step_budget_stops_without_llm(planner):
    p, llm = planner
    p.settings.max_steps = 1
    llm.responses = [json.dumps({"action": "finish"})]
    await p.seed("g", ".")
    await p.tick()  # step 1 -> consumes the only response
    result = await p.tick()  # step 2 -> budget hit, no llm call
    assert result["action"] == "finish"
    assert "budget" in result["reason"]
    assert llm.calls == 1
