"""Tests for the Verifier with an injected fake runner."""

from hifi_code.config import VerifyConfig
from hifi_code.verify import Verifier


class FakeProc:
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_verify_ok():
    def runner(cmd):
        return FakeProc(0, "3 passed in 0.1s")

    v = Verifier(runner=runner)
    res = v.verify(VerifyConfig(run_tests=True, run_lint=False))
    assert res["ok"] is True
    assert len(res["checks"]) == 1
    assert res["checks"][0]["summary"] == "3 passed in 0.1s"


def test_verify_fail_on_tests():
    def runner(cmd):
        return FakeProc(1, "1 failed")

    v = Verifier(runner=runner)
    res = v.verify(VerifyConfig(run_tests=True, run_lint=False))
    assert res["ok"] is False
    assert res["checks"][0]["ok"] is False


def test_verify_pytest_no_tests_is_ok():
    # pytest exit code 5 means "no tests were collected" — not a failure.
    def runner(cmd):
        return FakeProc(5, "no tests ran in 0.00s")

    v = Verifier(runner=runner)
    res = v.verify(VerifyConfig(run_tests=True, run_lint=False))
    assert res["ok"] is True
    assert res["checks"][0]["ok"] is True


def test_verify_composes_tests_and_lint():
    calls = []

    def runner(cmd):
        calls.append(cmd)
        # tests pass, lint fails
        return FakeProc(1 if cmd[0] == "ruff" else 0, "")

    v = Verifier(runner=runner)
    res = v.verify(VerifyConfig(run_tests=True, run_lint=True))
    assert res["ok"] is False
    assert len(res["checks"]) == 2
    assert calls == [["pytest", "-q"], ["ruff", "check", "."]]
