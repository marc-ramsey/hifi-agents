"""Tests for the CLI entry point (search_agent/__main__.main).

Exercises argument handling and the stdio result/error paths without any
network by monkeypatching the engines that main() constructs.
"""

import json

import pytest

import search_agent.__main__ as mod


class _FakeEngine:
    """Stand-in for RetrievalEngine/IngestionEngine with no side effects."""

    def __init__(self, results=None):
        self._results = results or []
        self.closed = False

    async def retrieve(self, query, mode="preference", top_k=None):
        return self._results

    async def close(self):
        self.closed = True


@pytest.fixture
def patch_engines(monkeypatch):
    """Replace RetrievalEngine + IngestionEngine with fakes that return results."""

    def _install(results=None):
        fake = _FakeEngine(results)
        monkeypatch.setattr(mod, "RetrievalEngine", lambda *a, **k: fake)
        monkeypatch.setattr(mod, "IngestionEngine", lambda *a, **k: fake)
        return fake

    return _install


def test_no_query_prints_help_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(mod.sys, "argv", ["search-agent"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_stdio_prints_results_as_json_lines(monkeypatch, capsys, patch_engines):
    fake = patch_engines(results=[{"url": "https://a.com", "score": 0.9}])
    monkeypatch.setattr(mod.sys, "argv", ["search-agent", "hello", "--mode", "parallel"])
    mod.main()  # success path returns normally (no sys.exit)

    line = json.loads(capsys.readouterr().out.strip())
    assert line["url"] == "https://a.com"
    assert fake.closed is True  # close() runs in finally


def test_stdio_no_results_exits_with_error(monkeypatch, capsys, patch_engines):
    patch_engines(results=[])
    monkeypatch.setattr(mod.sys, "argv", ["search-agent", "hello"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out.strip())["error"] == "No results found"
