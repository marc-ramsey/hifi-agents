"""System prompt for the loop driver.

The loop's playbook lives in ``.agents/skills/hifi_code/SKILL.md``. This module
loads it and injects the per-run context (goal, ast-bro digest, tool list).
Keeping SKILL.md as the single source of truth means the prompt can't drift
from the documented workflow.
"""

from pathlib import Path

_SKILL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / ".agents"
    / "skills"
    / "hifi_code"
    / "SKILL.md"
)


def _strip_frontmatter(text: str) -> str:
    """Drop the leading ``---\\n...\\n---`` YAML block if present."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 1 :].lstrip("\n")
    return text


def _read_skill() -> str:
    try:
        return _SKILL_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def system_prompt(goal: str, digest: str = "", tools: str = "") -> str:
    skill = _strip_frontmatter(_read_skill()).strip()
    body = (
        "You are hifi_code, an autonomous coding loop driven by ast-bro.\n"
        f"Goal: {goal}\n\n"
        + skill
        + f"\n\n{digest}"
    )
    if tools:
        body += f"\n\nAvailable ast-bro tools:\n{tools}"
    return body + "\n"
