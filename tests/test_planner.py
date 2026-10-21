"""Tests for the coding loop (planner) with fakes for LLM / ast-bro / verify."""

import asyncio
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

    async def chat(self, messages):
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
    llm.responses = [json.dumps({"action": "navigate", "tool": "map", "args": {"paths": ["a.py"]}})]
    await p.seed("g", ".")
    result = await p.tick()
    assert result == {"files": []}


async def test_navigate_map_without_paths_returns_cached_map(planner):
    p, llm = planner
    # A read tool called with no usable input is served the map already
    # fetched at seed time instead of crashing on a null argument.
    llm.responses = [json.dumps({"action": "navigate", "tool": "map", "args": {}})]
    await p.seed("g", ".")
    result = await p.tick()
    assert result == {"digest": "seed"}


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


async def test_navigate_show_without_symbols_errors_gracefully(planner):
    """A navigate:show with only a path (no symbols) must not crash ast-bro;
    it degrades to the read hint instead of "missing field symbols"."""
    p, llm = planner
    llm.responses = [
        json.dumps({"action": "navigate", "args": {
            "tool": "show", "path": "x.py"}}),
    ]
    await p.seed("g", ".")
    result = await p.tick()
    assert result["status"] == "error"
    assert "symbol" in result["reason"]


async def test_navigate_show_with_symbols_calls_show(planner):
    """navigate:show with explicit symbols is served like read."""
    p, llm = planner
    show = _ShowCapture()
    p.astbro.show = show.show
    llm.responses = [
        json.dumps({"action": "navigate", "args": {
            "tool": "show", "path": "x.py", "symbols": ["Foo"]}}),
    ]
    await p.seed("g", ".")
    await p.tick()
    assert show.calls[0]["symbols"] == ["Foo"]


async def test_navigate_show_falls_back_to_map(planner):
    """navigate:show with only a path derives symbols from the map, like read."""
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
    llm.responses = [
        json.dumps({"action": "navigate", "args": {
            "tool": "show", "path": "planner.py"}}),
    ]
    await p.seed("g", ".")
    p.state["map"] = {"raw": map_text}
    await p.tick()
    assert show.calls[0]["symbols"] == ["Planner", "__init__", "seed", "tick"]


async def test_verify_failure_increments_stall_and_stops(planner):
    """A failing verify increments the stall counter and halts the loop.

    Regression: the verify branch used to neither reset nor increment the
    stall counter, so a loop stuck in verify→verify→... could burn the full
    max_steps instead of stalling at max_stalls.
    """
    p, llm = planner
    async def fake_verify(cfg=None):
        return {"ok": False, "reason": "still failing"}
    p.verifier.verify = fake_verify
    llm.responses = [json.dumps({"action": "verify"})]
    await p.seed("g", ".")

    r = await p.tick()
    assert p.state["stall"] == 1
    assert r["ok"] is False

    for _ in range(p.settings.max_stalls - 1):
        r = await p.tick()
    assert p.state["stall"] == p.settings.max_stalls
    assert r["action"] == "finish" and r["ok"] is False
    assert r["status"] == "stalled"


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


class _ShowSource:
    """ast-bro ``show`` stub returning exact symbol source, like the real server."""

    def __init__(self, sources: dict):
        self.sources = sources  # {path: {symbol: source}}

    async def show(self, path=None, symbols=None, **kwargs):
        files = []
        for path_key, syms in self.sources.items():
            matches = [
                {"qualified_name": sym, "kind": "function",
                 "start_line": 1, "end_line": 2, "source": src}
                for sym, src in syms.items()
            ]
            if matches:
                files.append({"path": path_key, "language": "python",
                              "matches": matches})
        return {"files": files}


async def test_edit_symbol_scope_fetches_exact_source(planner, tmp_path):
    """Symbol-scoped edit: harness reads the exact source via show, so the model
    never reproduces the file body. It only names the symbol and the substring."""
    target = tmp_path / "demo.py"
    target.write_text('def greet(name):\n    return "hello " + name\n')
    source = 'def greet(name):\n    return "hello " + name'
    p, llm = planner
    p.astbro.show = _ShowSource({str(target): {"greet": source}}).show
    decision = json.dumps({
        "action": "edit",
        "edits": [{"path": str(target), "symbol": "greet",
                   "from": "hello ", "to": "hi "}],
    })
    llm.responses = [decision, json.dumps({"action": "finish", "reason": "done"})]
    await p.seed("g", ".")
    r1 = await p.tick()
    assert r1["edits"][0]["status"] == "applied"
    assert target.read_text() == 'def greet(name):\n    return "hi " + name\n'


async def test_edit_symbol_scope_missing_from_rejected(planner, tmp_path):
    target = tmp_path / "demo.py"
    target.write_text("x = 1\n")
    p, llm = planner
    p.astbro.show = _ShowSource({}).show
    decision = json.dumps({
        "action": "edit",
        "edits": [{"path": str(target), "symbol": "f", "to": "2"}],
    })
    llm.responses = [decision]
    await p.seed("g", ".")
    r1 = await p.tick()
    assert r1["edits"][0]["status"] == "rejected"
    assert "from" in r1["edits"][0]["reason"]


async def test_edit_symbol_scope_from_not_in_source_rejected(planner, tmp_path):
    target = tmp_path / "demo.py"
    target.write_text('def greet(name):\n    return "hello " + name\n')
    p, llm = planner
    p.astbro.show = _ShowSource({
        str(target): {"greet": 'def greet(name):\n    return "hello " + name'},
    }).show
    decision = json.dumps({
        "action": "edit",
        "edits": [{"path": str(target), "symbol": "greet",
                   "from": "hello world", "to": "hi there"}],
    })
    llm.responses = [decision]
    await p.seed("g", ".")
    r1 = await p.tick()
    assert r1["edits"][0]["status"] == "rejected"
    assert "not found in greet" in r1["edits"][0]["reason"]
    assert target.read_text() == 'def greet(name):\n    return "hello " + name\n'  # untouched


class _ShowRaw:
    """ast-bro ``show`` stub returning the real ``raw`` outline format."""

    def __init__(self, raw: str):
        self.raw = raw

    async def show(self, path=None, symbols=None, **kwargs):
        return {"raw": self.raw}


async def test_edit_symbol_scope_parses_real_raw_show(planner, tmp_path):
    """Symbol-scoped edit against the real ast-bro ``raw`` outline: the harness
    parses the ``# path:start-end Name (kind)`` header to recover exact source,
    so the model still only names the symbol and the substring."""
    target = tmp_path / "demo.py"
    target.write_text('def greet(name):\n    return "hello " + name\n')
    raw = (
        f"# {target}:1-2 greet (function)\n"
        'def greet(name):\n    return "hello " + name'
    )
    p, llm = planner
    p.astbro.show = _ShowRaw(raw).show
    decision = json.dumps({
        "action": "edit",
        "edits": [{"path": str(target), "symbol": "greet",
                   "from": "hello ", "to": "hi "}],
    })
    llm.responses = [decision, json.dumps({"action": "finish", "reason": "done"})]
    await p.seed("g", ".")
    r1 = await p.tick()
    assert r1["edits"][0]["status"] == "applied"
    assert target.read_text() == 'def greet(name):\n    return "hi " + name\n'


async def test_edit_symbol_scope_parses_method_raw_show(planner, tmp_path):
    """A method's ``raw`` entry carries a ``# in: class X`` annotation line and a
    fully-qualified header name; the symbol is matched by its trailing component."""
    target = tmp_path / "demo.py"
    target.write_text("class C:\n    def greet(self):\n        return 'hi'\n")
    raw = (
        f"# {target}:1-3 C.greet (method)\n"
        "# in: class C\n"
        "    def greet(self):\n        return 'hi'"
    )
    p, llm = planner
    p.astbro.show = _ShowRaw(raw).show
    decision = json.dumps({
        "action": "edit",
        "edits": [{"path": str(target), "symbol": "greet",
                   "from": "'hi'", "to": "'hey'"}],
    })
    llm.responses = [decision, json.dumps({"action": "finish", "reason": "done"})]
    await p.seed("g", ".")
    r1 = await p.tick()
    assert r1["edits"][0]["status"] == "applied"
    assert target.read_text() == "class C:\n    def greet(self):\n        return 'hey'\n"


async def test_edit_symbol_scope_ambiguous_substring_rejected(planner, tmp_path):
    """A substring that appears twice in the symbol is rejected, not silently
    replaced everywhere (the classic symbol-scoped edit bug)."""
    target = tmp_path / "demo.py"
    target.write_text("def f():\n    return x\n    return x\n")
    source = "def f():\n    return x\n    return x"
    p, llm = planner
    p.astbro.show = _ShowSource({str(target): {"f": source}}).show
    decision = json.dumps({
        "action": "edit",
        "edits": [{"path": str(target), "symbol": "f",
                   "from": "x", "to": "y"}],
    })
    llm.responses = [decision]
    await p.seed("g", ".")
    r1 = await p.tick()
    assert r1["edits"][0]["status"] == "rejected"
    assert "not unique" in r1["edits"][0]["reason"]
    assert target.read_text() == "def f():\n    return x\n    return x\n"  # untouched


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

        async def chat(self, messages):
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

    async def chat(self, messages):
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
    # The harness result (the source the model read) is what gets surfaced, with
    # the action label preserved so the model knows which turn it belongs to.
    assert out == [
        {"role": "assistant",
         "content": json.dumps({"raw": "abc", "action": "read"})}
    ]


def test_history_messages_falls_back_to_decision_without_result():
    history = [
        {"step": 1, "action": "read",
         "decision": {"action": "read", "path": "x.py"}},
    ]
    out = Planner._history_messages(history)
    assert out == [
        {"role": "assistant",
         "content": json.dumps({"action": "read", "path": "x.py"})}
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


def test_parse_extracts_first_balanced_object():
    """Trailing prose and a second object must not be swallowed into the first.

    A naive find/rfind would grab from the first ``{`` to the last ``}``,
    merging them; the balanced scan stops at the first close.
    """
    text = 'here is {action: navigate, tool: map} then {"action": "finish"}'
    assert Planner._parse(text) == {"action": "navigate", "tool": "map"}
    # a brace inside a string value must not close the object early.
    assert Planner._parse('{reason: "a}b", n: 1}') == {"reason": "a}b", "n": 1}


async def test_invalid_action_rejected_not_silent_success(planner):
    """A hallucinated action is rejected (ok=False) -- never a silent success.
    The rejection is retryable: the loop reports ``invalid`` with a hint so the
    model can self-correct, and the stall detector bounds a repeat loop."""
    p, llm = planner
    llm.responses = [
        json.dumps({"action": "frobnicate"}),
        json.dumps({"action": "finish", "reason": "done"}),
    ]
    await p.seed("g", ".")
    result = await p.tick()
    assert result["status"] == "invalid"
    assert result["ok"] is False
    assert "unknown action" in result["reason"]
    # retryable: a valid action on the next tick completes the loop.
    r2 = await p.tick()
    assert r2["action"] == "finish" and r2["ok"] is True


class _FakeSubAstBro:
    """ast-bro stub for nested sub-planners, created per subgoal by _make_astbro."""

    def __init__(self):
        self.digest_calls = 0
        self.closed = False

    async def digest(self, paths=None, **kwargs):
        self.digest_calls += 1
        return {"digest": "seed"}

    async def list_tools(self):
        return []

    async def close(self):
        self.closed = True


async def test_plan_without_subgoals_errors(planner):
    p, llm = planner
    llm.responses = [json.dumps({"action": "plan", "subgoals": []})]
    await p.seed("g", ".")
    result = await p.tick()
    assert result["action"] == "finish"
    assert result["status"] == "error"
    assert result["ok"] is False
    assert "no subgoals" in result["reason"]


async def test_plan_rejects_overlapping_paths(planner):
    """Two subgoals claiming the same file violate the disjoint-path invariant;
    the plan fails loudly instead of racing two concurrent writers."""
    p, llm = planner
    llm.responses = [
        json.dumps({
            "action": "plan",
            "subgoals": [
                {"goal": "a", "path": "a.py"},
                {"goal": "b", "path": "a.py"},
            ],
        }),
    ]
    await p.seed("g", ".")
    result = await p.tick()
    assert result["action"] == "finish"
    assert result["status"] == "error"
    assert "disjoint" in result["reason"]
    assert "a.py" in result["reason"]


async def test_plan_drives_each_subgoal_to_completion(planner):
    """Each subgoal is driven to its own terminal finish in its own nested
    Planner + ast-bro session, and the terminal results are concatenated."""
    p, llm = planner
    llm.responses = [
        json.dumps({"action": "plan", "subgoals": [
            {"goal": "a", "path": "a.py"},
            {"goal": "b", "path": "b.py"},
            {"goal": "c", "path": "c.py"},
        ]}),
        json.dumps({"action": "finish", "reason": "done"}),
    ]
    created = []
    p._make_astbro = lambda: (created.append(_FakeSubAstBro()) or created[-1])
    await p.seed("g", ".")
    result = await p.tick()
    assert len(created) == 3  # one fresh session per subgoal
    assert all(f.closed for f in created)  # each session closed after driving
    subgoals = result["subgoals"]
    assert len(subgoals) == 3
    assert all(r["action"] == "finish" for r in subgoals)


async def test_plan_runs_subgoals_concurrently(planner):
    """Subgoals actually run concurrently, not one-after-another: with a shared
    release gate, both subgoals start their work before either is unblocked."""
    p, llm = planner
    llm.responses = [
        json.dumps({"action": "plan", "subgoals": [
            {"goal": "a", "path": "a.py"},
            {"goal": "b", "path": "b.py"},
        ]}),
        json.dumps({"action": "finish", "reason": "done"}),
    ]

    started = []
    release = asyncio.Event()

    class _SlowAstBro(_FakeSubAstBro):
        async def digest(self, paths=None, **kwargs):
            started.append(1)
            await release.wait()
            return {"digest": "seed"}

    p._make_astbro = _SlowAstBro
    await p.seed("g", ".")

    task = asyncio.create_task(p.tick())
    await asyncio.sleep(0.05)  # let both subgoals reach their release gate
    assert len(started) == 2, "both subgoals should start before either is released"
    release.set()
    result = await task
    assert len(result["subgoals"]) == 2


async def test_invalid_action_rejected_not_dispatched(planner):
    """A decision with an unknown action is rejected before dispatch, with a
    specific reason and hint; the model can then retry with a valid one."""
    p, llm = planner
    llm.responses = [
        json.dumps({"action": "run"}),
        json.dumps({"action": "finish", "reason": "done"}),
    ]
    await p.seed("g", ".")
    r1 = await p.tick()
    assert r1["status"] == "invalid"
    assert "unknown action" in r1["reason"]
    assert "navigate" in r1["hint"]
    r2 = await p.tick()
    assert r2["action"] == "finish" and r2["ok"] is True


async def test_missing_required_field_rejected(planner):
    p, llm = planner
    llm.responses = [json.dumps({"action": "navigate"})]  # no tool
    await p.seed("g", ".")
    r1 = await p.tick()
    assert r1["status"] == "invalid"
    assert "missing required field 'tool'" in r1["reason"]


async def test_invalid_decision_reaches_model_in_history(planner):
    """A rejected decision is recorded in history as an assistant message, so
    the model sees the rejection on the next tick and can self-correct."""
    p, _ = planner
    cap = _CaptureMessagesLLM([
        json.dumps({"action": "bogus"}),
        json.dumps({"action": "finish", "reason": "done"}),
    ])
    p.llm = cap
    await p.seed("g", ".")
    r1 = await p.tick()
    assert r1["status"] == "invalid"
    r2 = await p.tick()
    assert r2["action"] == "finish"
    tick2_messages = cap.messages[1]
    assistant_msgs = [m for m in tick2_messages if m["role"] == "assistant"]
    assert any("unknown action" in m["content"] for m in assistant_msgs)


async def test_repeated_invalid_decisions_stall(planner):
    """Re-submitting an invalid decision is stagnation and eventually stalls
    the loop rather than spinning to the step budget."""
    p, llm = planner
    p.settings.max_stalls = 3
    llm.responses = [json.dumps({"action": "bogus"})]  # repeated every tick
    await p.seed("g", ".")
    for _ in range(3):
        r = await p.tick()
    assert r["action"] == "finish"
    assert r["status"] == "stalled"
