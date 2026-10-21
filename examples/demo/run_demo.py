"""Demo: drive the hifi_code coding loop against a small example file.

The loop is an autonomous coding agent: it seeds a code map from ``ast-bro``,
then repeatedly asks the LLM for the next action (navigate / read / edit /
verify / finish). Edits go through a gated editor that replaces one exact block
in one file; the loop verifies after every change.

This demo rewrites ``greeting.py`` to a known starting state, points the loop at
it with a one-line goal, and prints each loop step plus the resulting file.

Run from the repo root::

    HIFI_CODE_LLM_MODEL=code-Ornith-1.5-35B python examples/demo/run_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(DEMO_DIR, "greeting.py")
DB_PATH = os.path.join(DEMO_DIR, "demo.db")

GOAL = (
    "In examples/demo/greeting.py, replace the trailing comment "
    "'# TODO: wire greet into add' with '# Greeting helpers.'"
)

EXAMPLE_SRC = '''def greet(name):
    return "Hello, " + name


def add(a, b):
    return a + b


# TODO: wire greet into add
'''


def _reset_example() -> None:
    """Write the example to a known starting state so the demo is repeatable."""
    with open(EXAMPLE, "w", encoding="utf-8") as fh:
        fh.write(EXAMPLE_SRC)


async def main() -> None:
    sys.path.insert(0, _REPO_ROOT)
    os.chdir(_REPO_ROOT)
    _reset_example()

    from hifi_code.config import Settings, VerifyConfig
    from hifi_code.transports.stdio import run

    # Verify only the example file so the demo is self-contained and is not
    # confused by repo-wide test/lint state.
    settings = Settings(
        db_path=DB_PATH,
        sandbox_enabled=True,
        denylist=[],
        verify=VerifyConfig(
            run_tests=True,
            run_lint=False,
            test_cmd=["python", "-m", "py_compile", EXAMPLE],
        ),
    )
    print(f"goal: {GOAL}\n")

    await run(settings, GOAL, root="examples/demo")

    print("\n--- examples/demo/greeting.py after loop ---")
    content = await asyncio.to_thread(Path(EXAMPLE).read_text, encoding="utf-8")
    print(content)

    # Leave the working tree clean: the demo intentionally mutates the fixture,
    # so reset it here (after printing the result) instead of expecting the
    # caller to revert it before committing.
    _reset_example()


if __name__ == "__main__":
    asyncio.run(main())
