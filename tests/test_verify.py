"""Tests for the Verifier with an injected fake runner."""

from hifi_code.config import VerifyConfig
from hifi_code.verify import Verifier


class FakeProc:
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def test_verify_ok():
    async def runner(cmd):
        return FakeProc(0, "3 passed in 0.1s")

    v = Verifier(runner=runner)
    res = await v.verify(VerifyConfig(run_tests=True, run_lint=False))
    assert res["ok"] is True
    assert len(res["checks"]) == 1
    assert res["checks"][0]["summary"] == "3 passed in 0.1s"


async def test_verify_fail_on_tests():
    async def runner(cmd):
        return FakeProc(1, "1 failed")

    v = Verifier(runner=runner)
    res = await v.verify(VerifyConfig(run_tests=True, run_lint=False))
    assert res["ok"] is False
    assert res["checks"][0]["ok"] is False


async def test_verify_pytest_no_tests_is_ok():
    # pytest exit code 5 means "no tests were collected" — not a failure.
    async def runner(cmd):
        return FakeProc(5, "no tests ran in 0.00s")

    v = Verifier(runner=runner)
    res = await v.verify(VerifyConfig(run_tests=True, run_lint=False))
    assert res["ok"] is True
    assert res["checks"][0]["ok"] is True


async def test_verify_composes_tests_and_lint():
    calls = []

    async def runner(cmd):
        calls.append(cmd)
        # tests pass, lint fails
        return FakeProc(1 if cmd[0] == "ruff" else 0, "")

    v = Verifier(runner=runner)
    res = await v.verify(VerifyConfig(run_tests=True, run_lint=True))
    assert res["ok"] is False
    assert len(res["checks"]) == 2
    assert calls == [["pytest", "-q"], ["ruff", "check", "."]]
