"""Tests for the surgical editor and its propose-diff gate."""

from hifi_code.editor import Edit, Editor


def test_apply_success(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n")
    editor = Editor()
    res = editor.apply(Edit(str(f), "x = 1", "x = 42"))
    assert res["status"] == "applied"
    assert f.read_text() == "x = 42\ny = 2\n"


def test_apply_noop_when_old_equals_new(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    editor = Editor()
    res = editor.apply(Edit(str(f), "x = 1", "x = 1"))
    assert res["status"] == "noop"
    assert f.read_text() == "x = 1\n"  # unchanged


def test_apply_rejected_when_old_missing(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    editor = Editor()
    res = editor.apply(Edit(str(f), "x = 999", "x = 42"))
    assert res["status"] == "rejected"
    assert "not found" in res["reason"]
    assert f.read_text() == "x = 1\n"  # untouched


def test_apply_rejected_when_old_not_unique(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("f()\nf()\n")
    editor = Editor()
    res = editor.apply(Edit(str(f), "f()", "g()"))
    assert res["status"] == "rejected"
    assert "not unique" in res["reason"]
    assert f.read_text() == "f()\nf()\n"  # untouched


def test_propose_only_never_writes(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    editor = Editor(propose_only=True)
    res = editor.apply(Edit(str(f), "x = 1", "x = 2"))
    assert res["status"] == "proposed"
    assert f.read_text() == "x = 1\n"
    assert len(editor.diff()) == 1


def test_diff_accumulates(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("a\nb\n")
    editor = Editor()
    editor.apply(Edit(str(f), "a", "A"))
    editor.apply(Edit(str(f), "b", "B"))
    diffs = editor.diff()
    assert len(diffs) == 2
    assert diffs[0]["new"] == "A"
