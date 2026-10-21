"""Tests for the surgical editor and its propose-diff gate."""

from hifi_code.editor import Edit, Editor


async def test_apply_success(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n")
    editor = Editor()
    res = await editor.apply(Edit(str(f), "x = 1", "x = 42"))
    assert res["status"] == "applied"
    assert f.read_text() == "x = 42\ny = 2\n"


async def test_apply_noop_when_old_equals_new(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    editor = Editor()
    res = await editor.apply(Edit(str(f), "x = 1", "x = 1"))
    assert res["status"] == "noop"
    assert f.read_text() == "x = 1\n"  # unchanged


async def test_apply_rejected_when_old_missing(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    editor = Editor()
    res = await editor.apply(Edit(str(f), "x = 999", "x = 42"))
    assert res["status"] == "rejected"
    assert "not found" in res["reason"]
    assert f.read_text() == "x = 1\n"  # untouched


async def test_apply_rejected_when_old_not_unique(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("f()\nf()\n")
    editor = Editor()
    res = await editor.apply(Edit(str(f), "f()", "g()"))
    assert res["status"] == "rejected"
    assert "not unique" in res["reason"]
    assert f.read_text() == "f()\nf()\n"  # untouched


async def test_propose_only_never_writes(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    editor = Editor(propose_only=True)
    res = await editor.apply(Edit(str(f), "x = 1", "x = 2"))
    assert res["status"] == "proposed"
    assert f.read_text() == "x = 1\n"
    assert len(editor.diff()) == 1


async def test_diff_accumulates(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("a\nb\n")
    editor = Editor()
    await editor.apply(Edit(str(f), "a", "A"))
    await editor.apply(Edit(str(f), "b", "B"))
    diffs = editor.diff()
    assert len(diffs) == 2
    assert diffs[0]["new"] == "A"


async def test_apply_rejects_breaking_syntax(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n")
    editor = Editor()
    # 'return 1' is unique; 'return 1 +' leaves a trailing operator -> SyntaxError.
    res = await editor.apply(Edit(str(f), "return 1", "return 1 +"))
    assert res["status"] == "rejected"
    assert "invalid Python" in res["reason"]
    assert f.read_text() == "def f():\n    return 1\n"  # untouched, not written


async def test_apply_accepts_edit_that_keeps_valid_syntax(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n")
    editor = Editor()
    res = await editor.apply(Edit(str(f), "return 1", "return 2"))
    assert res["status"] == "applied"
    assert f.read_text() == "def f():\n    return 2\n"


async def test_apply_ignores_non_python_files(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("return 1\n")
    editor = Editor()
    res = await editor.apply(Edit(str(f), "return 1", "return 1 +"))
    assert res["status"] == "applied"  # .py check is skipped for non-Python


async def test_apply_region_replaces_within_region_only(tmp_path):
    """A region-scoped edit changes only the occurrence inside the region; the
    same substring elsewhere in the file is left untouched."""
    f = tmp_path / "a.py"
    f.write_text("x = 'x'\ndef f():\n    y = 'x'\n")
    editor = Editor()
    region = "def f():\n    y = 'x'"
    res = await editor.apply(Edit(str(f), "'x'", "'z'"), region=region)
    assert res["status"] == "applied"
    assert f.read_text() == "x = 'x'\ndef f():\n    y = 'z'\n"


async def test_apply_region_rejects_ambiguous_substring(tmp_path):
    """The core bug: a substring appearing twice inside the region is rejected
    rather than silently replaced everywhere, and the file is left untouched."""
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return x\n    return x\n")
    editor = Editor()
    region = "def f():\n    return x\n    return x"
    res = await editor.apply(Edit(str(f), "x", "y"), region=region)
    assert res["status"] == "rejected"
    assert "not unique" in res["reason"]
    assert f.read_text() == "def f():\n    return x\n    return x\n"


async def test_apply_region_rejected_when_region_missing(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    editor = Editor()
    res = await editor.apply(Edit(str(f), "x", "y"), region="def missing():\n    pass")
    assert res["status"] == "rejected"
    assert "not found" in res["reason"]
    assert f.read_text() == "x = 1\n"


async def test_apply_region_rejected_when_region_not_unique(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def dup():\n    return 1\ndef dup():\n    return 1\n")
    editor = Editor()
    region = "def dup():\n    return 1"
    res = await editor.apply(Edit(str(f), "1", "2"), region=region)
    assert res["status"] == "rejected"
    assert "not unique" in res["reason"]
    assert f.read_text() == "def dup():\n    return 1\ndef dup():\n    return 1\n"
