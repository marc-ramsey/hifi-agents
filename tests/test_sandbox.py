"""Sandbox denylist + working-dir boundary enforcement."""

from hifi_code.sandbox import Sandbox


def test_relative_path_inside_working_dir_is_allowed(tmp_path, monkeypatch):
    """Edit paths resolve against working_dir, so a legitimate relative file
    inside it is allowed even when CWD != working_dir.

    Old code resolved edit paths against CWD (base=None), so a relative path
    landed outside working_dir and was denied by the containment boundary.
    """
    monkeypatch.chdir(tmp_path)  # CWD != working_dir
    wd = "project"
    (tmp_path / wd).mkdir()
    (tmp_path / wd / "ok.py").write_text("x = 1\n")
    sb = Sandbox(working_dir=wd, denylist=[])
    assert sb.check_edit("ok.py") is True


def test_relative_denylist_blocks_file_in_working_dir(tmp_path, monkeypatch):
    """A relative denylist entry is resolved against working_dir and blocks the
    same-named edit path (both use the working_dir base)."""
    monkeypatch.chdir(tmp_path)
    wd = "project"
    (tmp_path / wd).mkdir()
    (tmp_path / wd / "secret.py").write_text("x = 1\n")
    sb = Sandbox(working_dir=wd, denylist=["secret.py"])
    assert sb.check_edit("secret.py") is False
    assert sb.is_denied(str(tmp_path / wd / "secret.py")) is True
    # a sibling inside working_dir is allowed
    (tmp_path / wd / "ok.py").write_text("x = 1\n")
    assert sb.check_edit("ok.py") is True


def test_absolute_denylist_and_absent_file(tmp_path):
    """Absolute denylist entries block the named file; files outside the
    working_dir are blocked by the containment boundary."""
    wd = tmp_path / "project"
    wd.mkdir()
    denied = wd / "secret.py"
    denied.write_text("x = 1\n")
    sb = Sandbox(working_dir=str(wd), denylist=[str(denied)])
    assert sb.check_edit(str(denied)) is False
    # a path outside working_dir is denied by containment
    assert sb.check_edit(str(tmp_path / "escape.py")) is False
    # a sibling inside working_dir is allowed
    ok = wd / "ok.py"
    ok.write_text("x = 1\n")
    assert sb.check_edit(str(ok)) is True


def test_disabled_sandbox_allows_everything(tmp_path):
    sb = Sandbox(working_dir=str(tmp_path), denylist=[str(tmp_path / "secret.py")],
                 enabled=False)
    assert sb.check_edit(str(tmp_path / "secret.py")) is True
