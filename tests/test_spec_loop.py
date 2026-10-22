"""Tests for the outer loop: draft a spec with the user, resolve it, hand it off.

Mirrors test_planner.py: the loop logic is separated from I/O behind SpecDriver,
so a scripted driver drives it here while the real input()-based driver (CliDriver)
is exercised by hand. The seam test feeds a resolved spec's goal into a *real*
inner Planner to prove the two loops connect.

Note on the schema: ``events``/``state``/``trigger``/``data_model``/``transitions``/
``modules``/``seams``/``inputs``/``acceptance`` are required with no default, so an
empty instance never resolves -- the loop must fill them (or the user does) before
validation passes. ``side_effects``/``outputs`` are required *with* a default (``[]``)
and are materialized by the loop.
"""

import json

from hifi_code.config import Settings
from hifi_code.editor import Editor
from hifi_code.planner import Planner
from hifi_code.spec_data import DATA_EVENT_SPEC
from hifi_code.spec_loop import CliDriver, DraftOutcome, SpecLoop
from hifi_code.verify import Verifier


class ScriptedDriver:
    """A scripted SpecDriver. Approves defaults by default; records I/O."""

    def __init__(self, outcomes, goal_ok=True):
        self.outcomes = list(outcomes)
        self.prompt = None
        self.goal = None
        self.goal_ok = goal_ok
        self.errors_seen: list[list[str]] = []

    def present_prompt(self, text):
        self.prompt = text

    def present_draft(self, draft):
        self.draft = draft
        return self.outcomes.pop(0)

    def present_errors(self, errors):
        self.errors_seen.append(errors)

    def present_goal(self, goal):
        self.goal = goal
        return self.goal_ok


def _resolve(instance=None):
    return DraftOutcome(instance=instance or {}, resolve=True)


def _edit(instance):
    return DraftOutcome(instance=instance)  # resolve=False -> edit branch


def _full_edit():
    """Fill every required-without-default field so a spec can resolve."""
    return DraftOutcome(instance={
        "events": ["OrderPlaced"],
        "state": ["order: pending -> paid"],
        "trigger": "HTTP request arrives",
        "data_model": ["order: id: int, total: decimal, status: str"],
        "transitions": ["OrderPlaced -> order: paid"],
        "modules": ["domain/", "adapters/", "main.py"],
        "seams": ["Store protocol"],
        "inputs": ["order payload"],
        "acceptance": ["order persisted and email sent"],
    })


async def _run(driver):
    return await SpecLoop(DATA_EVENT_SPEC, driver).run()


async def test_resolves_defaults_and_builds_goal():
    # Required-with-default gaps (side_effects/outputs) are materialized from [].
    driver = ScriptedDriver([_full_edit(), _resolve()])
    resolved, goal = await _run(driver)
    assert resolved["side_effects"] == []
    assert resolved["outputs"] == []
    assert "data/event-driven application" in goal
    assert "transitions" in goal


async def test_driver_sees_prompt_once_and_goal():
    driver = ScriptedDriver([_full_edit(), _resolve()])
    _, goal = await _run(driver)
    assert "Spec schema" in driver.prompt
    assert driver.goal == goal


async def test_abort_returns_none():
    resolved, goal = await _run(ScriptedDriver([DraftOutcome(instance={}, abort=True)]))
    assert resolved is None
    assert goal == ""


async def test_resolve_retries_when_required_missing():
    # First resolve fails (events/state/... required with no default); errors
    # surfaced, then an edit fills them, then resolve succeeds.
    driver = ScriptedDriver([_resolve(), _full_edit(), _resolve()])
    resolved, goal = await _run(driver)
    assert resolved["events"] == ["OrderPlaced"]
    assert len(driver.errors_seen) == 1
    assert driver.goal == goal


async def test_declining_goal_aborts():
    driver = ScriptedDriver([_full_edit(), _resolve()], goal_ok=False)
    resolved, _goal = await _run(driver)
    assert resolved is None


async def test_cli_driver_normalizes_list_values():
    # CliDriver accumulates edits across drafts and splits list fields on commas.
    answers = iter([
        "edit", "events", "OrderPlaced, EmailSent",
        "edit", "state", "order: pending -> paid",
        "edit", "trigger", "HTTP request arrives",
        "edit", "data_model", "order: id: int, total: decimal, status: str",
        "edit", "transitions", "OrderPlaced -> order: paid",
        "edit", "modules", "domain/, adapters/, main.py",
        "edit", "seams", "Store protocol",
        "edit", "inputs", "order payload",
        "edit", "acceptance", "order persisted",
        "resolve", "y",
    ])
    driver = CliDriver(DATA_EVENT_SPEC, in_=lambda _="": next(answers))
    resolved, goal = await _run(driver)
    assert resolved["events"] == ["OrderPlaced", "EmailSent"]
    assert "OrderPlaced" in goal


# --- The seam: a resolved spec's goal drives a real inner Planner ----------


class _FakeAstBro:
    async def digest(self, paths=None, **kwargs):
        return {"digest": "seed"}

    async def list_tools(self):
        return [{"name": "map"}, {"name": "show"}, {"name": "call"}]

    async def map(self, paths=None, **kwargs):
        return {"files": []}

    async def show(self, path=None, **kwargs):
        return {"symbols": []}

    async def call(self, name, arguments=None):
        return {"tool": name}


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def chat(self, messages):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


async def test_goal_drives_inner_planner():
    # The seam: to_goal() output seeds a real inner loop that runs to finish.
    _, goal = await _run(ScriptedDriver([_full_edit(), _resolve()]))

    settings = Settings(max_steps=5, sandbox_enabled=False, denylist=[])
    llm = _FakeLLM([json.dumps({"action": "finish", "reason": "done"})])
    astbro = _FakeAstBro()
    planner = Planner(settings, llm, astbro, Editor(), Verifier())
    await planner.seed(goal, ".")
    assert planner.goal == goal
    result = await planner.tick()
    assert result["action"] == "finish"
    assert planner.step == 1
