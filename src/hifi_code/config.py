"""All hifi_code settings and environment overrides.

Mirrors search_agent.config: Pydantic v2 models with sub-configs wired via
``Field(default_factory=...)`` and env-var defaults.
"""

import os

from pydantic import BaseModel, Field


def _default_server_cmd() -> list[str]:
    env = os.environ.get("HIFI_CODE_SERVER")
    if env:
        return env.split()
    return ["ast-bro", "mcp"]


def _default_llm_base_url() -> str:
    return os.environ.get("HIFI_CODE_LLM_BASE_URL") or "http://localhost:8080/v1"


def _default_llm_model() -> str:
    return os.environ.get("HIFI_CODE_LLM_MODEL") or "gpt-4o-mini"


class LLMConfig(BaseModel):
    base_url: str = Field(default_factory=_default_llm_base_url)
    api_key: str = ""
    model: str = Field(default_factory=_default_llm_model)
    temperature: float = 0.0


class VerifyConfig(BaseModel):
    run_tests: bool = True
    run_lint: bool = True
    test_cmd: list[str] = Field(default_factory=lambda: ["pytest", "-q"])
    lint_cmd: list[str] = Field(default_factory=lambda: ["ruff", "check", "."])


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    server_cmd: list[str] = Field(default_factory=_default_server_cmd)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    max_steps: int = 40
    # Finish with a "stalled" reason after this many consecutive ticks that make
    # no meaningful progress (no applied edit, no passing verify). Prevents the
    # loop from spinning forever on repeated reads or re-submitted edits.
    max_stalls: int = 12
    working_dir: str = "."
    sandbox_enabled: bool = True
    denylist: list[str] = Field(default_factory=list)
    # When True the editor records proposed edits but never writes to disk.
    propose_only: bool = False
