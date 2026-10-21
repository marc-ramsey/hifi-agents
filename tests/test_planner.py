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

    async def list_tools(self):
        return [
            {"name": "map", "description": ""},
            {"name": "show", "description": ""},
            {"name": "call", "description": ""},
        ]

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


async def test_navigate_unknown_tool_errors_gracefully(planner):
    p, llm = planner
    # "read" is a loop action, not an ast-bro tool; the model must not crash.
    llm.responses = [json.dumps({"action": "navigate", "tool": "read", "args": {}})]
    await p.seed("g", ".")
    result = await p.tick()
    assert result["status"] == "error"
    assert "unknown ast-bro tool" in result["reason"]
    assert "valid tools" in result["hint"]
    assert "map" in result["hint"]


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


class _ShowCapture:
    def __init__(self):
        self.calls = []

    async def show(self, path=None, symbols=None, **kwargs):
        self.calls.append({"path": path, "symbols": symbols})
        return {"symbols": [{"name": s} for s in (symbols or [])]}


async def test_read_uses_explicit_symbols_normalized(planner):
    p, llm = planner
    show = _ShowCapture()
    p.astbro.show = show.show
    llm.responses = [
        json.dumps({"action": "read", "path": "x.py", "symbols": ["class Foo", "bar()"]}),
    ]
    await p.seed("g", ".")
    await p.tick()
    assert show.calls[0]["symbols"] == ["Foo", "bar"]


async def test_read_without_symbols_falls_back_to_map(planner):
    p, llm = planner
    map_text = (
        "# legend: name() = callable\n"
        "/\n"
        "  planner.py ([small], 142 lines, 1 types, 10 methods)\n"
        "    class Planner  L21-141\n"
        "      __init__()  [async] seed()  [async] tick()\n"
    )
    show = _ShowCapture()
    p.astbro.show = show.show
    llm.responses = [json.dumps({"action": "read", "path": "planner.py"})]
    await p.seed("g", ".")
    p.state["map"] = {"raw": map_text}
    await p.tick()
    assert show.calls[0]["symbols"] == ["Planner", "__init__", "seed", "tick"]


async def test_read_without_path_or_symbols_errors(planner):
    p, llm = planner
    show = _ShowCapture()
    p.astbro.show = show.show
    llm.responses = [json.dumps({"action": "read", "path": "empty.py"})]
    await p.seed("g", ".")
    p.state["map"] = {"raw": "# legend: name() = callable\n/\n  empty.py ([tiny], 3 lines)\n"}
    result = await p.tick()
    assert result["status"] == "error"
    assert "symbol" in result["reason"]
    assert show.calls == []


async def test_llm_turn_failure_returns_terminal_error(planner):
    class RaisingLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            raise RuntimeError("502 Bad Gateway")

    p, _ = planner
    p.llm = RaisingLLM()
    await p.seed("g", ".")
    result = await p.tick()
    assert result["action"] == "finish"
    assert result["status"] == "error"
    assert result["ok"] is False
    assert "502" in result["reason"]


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


class _CaptureMessagesLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []
        self.calls = 0

    def chat(self, messages):
        self.messages.append(messages)
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


def test_history_messages_formats_role_keys():
    history = [
        {"step": 1, "action": "read",
         "decision": {"action": "read", "path": "x.py"}, "result": {"raw": "abc"}},
    ]
    out = Planner._history_messages(history)
    assert out == [
        {"role": "assistant", "content": json.dumps({"action": "read", "path": "x.py"})}
    ]


async def test_history_sent_as_valid_assistant_messages(planner):
    p, llm = planner
    # finish after two ticks so two history entries accumulate
    llm.responses = [
        json.dumps({"action": "finish", "reason": "done"}),
        json.dumps({"action": "finish", "reason": "done"}),
    ]
    cap = _CaptureMessagesLLM(llm.responses)
    p.llm = cap
    await p.seed("g", ".")
    await p.tick()
    await p.tick()
    # every message across all calls must carry a role key, or the backend 502s
    for messages in cap.messages:
        for m in messages:
            assert "role" in m
    # the accumulated history entries appear as assistant messages
    assistant = [m for messages in cap.messages for m in messages if m["role"] == "assistant"]
    assert len(assistant) == 1


async def test_stalled_after_repeated_rejected_edits(planner, tmp_path):
    """Re-submitting an edit that can't apply stalls the loop (doesn't spin)."""
    target = tmp_path / "a.py"
    target.write_text("x = 1\n")
    p, llm = planner
    p.settings.max_stalls = 3
    # old text is never present, so every edit is rejected (no progress).
    bad_edit = json.dumps({
        "action": "edit",
        "edits": [{"path": str(target), "old": "x = 999", "new": "x = 2"}],
    })
    llm.responses = [bad_edit]  # repeated every tick
    await p.seed("g", ".")
    await p.tick()  # 1: rejected, stall=1
    await p.tick()  # 2: rejected, stall=2
    result = await p.tick()  # 3: rejected, stall=3 -> stalled
    assert result["action"] == "finish"
    assert result["status"] == "stalled"
    assert "no progress" in result["reason"]
    assert target.read_text() == "x = 1\n"  # never changed


async def test_stalled_after_repeated_reads(planner):
    """Re-reading the same symbol stalls the loop instead of spinning."""
    p, llm = planner
    p.settings.max_stalls = 3
    llm.responses = [
        json.dumps({"action": "read", "path": "x.py", "symbols": ["f"]}),
    ]
    await p.seed("g", ".")
    await p.tick()  # 1: read, stall=1
    await p.tick()  # 2: read, stall=2
    r3 = await p.tick()  # 3: read, stall=3 -> stalled
    assert r3["action"] == "finish"
    assert r3["status"] == "stalled"


async def test_applied_edit_resets_stall_counter(planner, tmp_path):
    """A real edit resets the stall counter, so exploration before it isn't
    penalised."""
    target = tmp_path / "a.py"
    target.write_text("x = 1\n")
    p, llm = planner
    p.settings.max_stalls = 3
    decisions = [
        json.dumps({"action": "read", "path": "x.py", "symbols": ["f"]}),
        json.dumps({"action": "read", "path": "x.py", "symbols": ["f"]}),
        json.dumps({
            "action": "edit",
            "edits": [{"path": str(target), "old": "x = 1", "new": "x = 2"}],
        }),
        json.dumps({"action": "read", "path": "x.py", "symbols": ["f"]}),
        json.dumps({"action": "read", "path": "x.py", "symbols": ["f"]}),
    ]
    llm.responses = decisions
    await p.seed("g", ".")
    await p.tick()  # read, stall=1
    await p.tick()  # read, stall=2
    r3 = await p.tick()  # edit applied -> stall reset to 0
    assert r3["edits"][0]["status"] == "applied"
    await p.tick()  # read, stall=1
    r5 = await p.tick()  # read, stall=2
    # Without the reset, five reads would have stalled at 3. The applied edit
    # cleared the counter, so we are still not stalled.
    assert r5.get("status") != "stalled"


def test_parse_bare_keys_and_values():
    """Bare (unquoted) keys/values must be quoted so json.loads accepts them,
    with every separator preserved."""
    assert Planner._parse("{action: navigate, path: a/b.py}") == {
        "action": "navigate",
        "path": "a/b.py",
    }
    # literal true/null survive; they must not be quoted.
    assert Planner._parse('{path: x, flag: true, n: null}') == {
        "path": "x",
        "flag": True,
        "n": None,
    }
    # bare numbers stay unquoted; bare strings are quoted.
    assert Planner._parse('{count: 5, ratio: -3.14, label: x}') == {
        "count": 5,
        "ratio": -3.14,
        "label": "x",
    }
