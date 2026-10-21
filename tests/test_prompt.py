"""Tests for the system prompt builder.

The prompt is generated from ``.agents/skills/hifi_code/SKILL.md``. These tests
are a drift guard: if the playbook changes, the prompt must change with it, and
if the prompt stops reading SKILL.md the whole thing breaks.
"""

from pathlib import Path

from hifi_code.prompt import _strip_frontmatter, system_prompt

_SKILL_PATH = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "skills"
    / "hifi_code"
    / "SKILL.md"
)


def _skill_body() -> str:
    return _strip_frontmatter(_SKILL_PATH.read_text(encoding="utf-8")).strip()


def test_prompt_sources_from_skill_md():
    # The whole SKILL.md body must appear verbatim in the prompt. This is the
    # drift guard: editing the playbook changes the prompt automatically.
    prompt = system_prompt("Add feature X", digest="DIGEST", tools="map\ndigest")
    assert _skill_body() in prompt


def test_prompt_strips_skill_frontmatter():
    # The YAML frontmatter (name/description) is skill metadata, not prompt.
    prompt = system_prompt("Add feature X")
    assert "name: hifi_code" not in prompt


def test_prompt_injects_goal_digest_and_tools():
    prompt = system_prompt("Add feature X", digest="DIGEST", tools="map\ndigest")
    assert "Add feature X" in prompt
    assert "DIGEST" in prompt
    assert "Available ast-bro tools:\nmap\ndigest" in prompt


def test_prompt_carries_plan_and_create_file_guidance():
    # Regression lock: the plan (parallel disjoint subgoals) and CREATE-file
    # (empty old) guidance added to SKILL.md must reach the model.
    prompt = system_prompt("Add feature X")
    assert "plan" in prompt
    assert "disjoint" in prompt
    assert "empty `old`" in prompt
