"""Tests for the hifi_code CLI entry point (hifi_code/__main__.main).

Mirrors tests/test_main.py for hifi_search. The primary purpose of this module
is a regression guard: importing the entry point here means a syntax error or a
broken import in ``__main__.py`` (e.g. the unindented import block that once
made ``--outer`` unrunnable) fails the suite instead of crashing the CLI only
when invoked from the shell.
"""

import pytest

import hifi_code.__main__ as mod


def test_entry_point_imports():
    """Importing the CLI module must succeed; a broken import is caught here."""
    assert callable(mod.main)


def test_no_goal_prints_help_and_exits(monkeypatch, capsys):
    """With no goal and no --outer/--transport, main prints help and exits 1."""
    monkeypatch.setattr(mod.sys, "argv", ["hifi_code"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1
    assert "usage" in capsys.readouterr().out.lower()
