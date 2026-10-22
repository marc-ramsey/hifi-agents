"""Gated end-to-end test of the real coding loop against the live model + ast-bro.

The mechanical loop (dispatch, edit application, symbol scoping, verify, stall,
budget) is covered by ``tests/test_planner.py`` with fakes. This test exercises
the *real* model (``code-Ornith-1.5-35B`` by default) and the real ast-bro MCP
server end-to-end:

    seed -> navigate -> read -> edit -> verify -> finish

then loads the resulting file and asserts ``transform(3) == 7`` at runtime.

It is skipped unless the LLM endpoint and ast-bro are both reachable, so the
suite stays green in offline / CI environments while still validating
model-driven behavior wherever the endpoints exist.
"""

import importlib.util
import os
import shutil

import httpx
import pytest

from hifi_code.astbro import AstBro
from hifi_code.config import Settings, VerifyConfig
from hifi_code.editor import Editor
from hifi_code.llm import LLMClient
from hifi_code.planner import Planner
from hifi_code.verify import Verifier

OUT = "transform"
HELPER = "def helper(x):\n    return x * 2\n"


def _model_available(base_url: str, model: str, timeout: float = 3.0) -> bool:
    """True if ``model`` is listed (and loaded) at the endpoint's /v1/models."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{base_url}/models")
            resp.raise_for_status()
            ids = [m["id"] for m in resp.json().get("data", [])]
        return model in ids
    except httpx.HTTPError:
        return False


def _astbro_available() -> bool:
    return shutil.which("ast-bro") is not None


def _requires_live_model(reason: str):
    settings = Settings()
    model = os.environ.get("HIFI_CODE_LLM_MODEL") or "code-Ornith-1.5-35B"
    return pytest.mark.skipif(
        not _model_available(settings.llm.base_url, model)
        or not _astbro_available(),
        reason=reason,
    )


@_requires_live_model(
    "LLM endpoint and ast-bro are not reachable; skipping real e2e loop"
)
async def test_real_loop_transform(tmp_path, monkeypatch):
    # The model emits relative paths that the editor resolves against CWD, so
    # chdir into tmp_path and seed with "." so every path lands inside tmp_path.
    monkeypatch.chdir(tmp_path)
    scratch = tmp_path / f"{OUT}.py"
    scratch.write_text(HELPER)

    goal = (
        f"Add a function {OUT}(x) to {OUT}.py that returns helper(x) + 1. "
        f"helper is already defined in {OUT}.py. Then verify."
    )
    settings = Settings(
        max_steps=20,
        sandbox_enabled=False,
        verify=VerifyConfig(run_tests=False, run_lint=False),
    )
    # The loop defaults to ``gpt-4o-mini``; this test targets the code model the
    # endpoint actually serves. Honor an explicit override, else pin it here.
    settings.llm.model = os.environ.get("HIFI_CODE_LLM_MODEL") or "code-Ornith-1.5-35B"

    llm = LLMClient(settings.llm)
    astbro = AstBro(cmd=settings.server_cmd)
    editor = Editor()
    verifier = Verifier()
    planner = Planner(settings, llm, astbro, editor, verifier)

    await planner.seed(goal, ".")

    terminal = None
    history = []
    try:
        for _ in range(settings.max_steps):
            terminal = await planner.tick()
            history.append(terminal)
            if terminal.get("action") == "finish":
                break
    finally:
        await astbro.close()

    # The loop must finish cleanly (not error/stall) ...
    assert terminal is not None, "loop never produced a tick"
    assert terminal.get("action") == "finish"
    assert terminal.get("ok") is True
    assert terminal.get("status") != "stalled"

    # ... and the produced file must be correct at runtime: transform(3) == 7.
    spec = importlib.util.spec_from_file_location(f"{OUT}_e2e", scratch)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert getattr(mod, OUT)(3) == 7
