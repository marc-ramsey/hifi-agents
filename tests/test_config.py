"""Tests for Settings / config defaults and env-var overrides."""

from hifi_code.config import Settings


def test_defaults():
    s = Settings()
    assert s.llm.base_url == "http://localhost:8080/v1"
    assert s.llm.model == "gpt-4o-mini"
    assert s.llm.temperature == 0.0
    assert s.server_cmd == ["ast-bro", "mcp"]
    assert s.max_steps == 40
    assert s.max_stalls == 12
    assert s.verify.run_tests is True
    assert s.verify.run_lint is True
    assert s.denylist == []
    assert s.sandbox_enabled is True


def test_llm_base_url_from_env(monkeypatch):
    monkeypatch.setenv("HIFI_CODE_LLM_BASE_URL", "http://example.test/v1")
    assert Settings().llm.base_url == "http://example.test/v1"


def test_server_cmd_from_env(monkeypatch):
    monkeypatch.setenv("HIFI_CODE_SERVER", "custom server --flag")
    assert Settings().server_cmd == ["custom", "server", "--flag"]


def test_partial_override_keeps_subconfig_defaults():
    s = Settings(llm={"model": "my-model"})
    assert s.llm.model == "my-model"
    assert s.llm.base_url == "http://localhost:8080/v1"
