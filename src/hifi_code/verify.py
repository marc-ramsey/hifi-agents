"""Verification checks (tests / lint / re-map).

A command runner is injectable so tests can supply a fake; the default runs
subprocesses. ``verify`` composes the configured checks and reports overall ok.
"""

import subprocess

from .config import VerifyConfig


def _default_runner(cmd: list[str]):
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


class VerifyResult:
    def __init__(self, name, ok, summary, returncode=0, output=""):
        self.name = name
        self.ok = ok
        self.summary = summary
        self.returncode = returncode
        self.output = output

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok,
                "summary": self.summary, "returncode": self.returncode}


class Verifier:
    def __init__(self, *, runner=None):
        self._runner = runner or _default_runner

    def run(self, cmd: list[str]) -> VerifyResult:
        proc = self._runner(cmd)
        output = proc.stdout or proc.stderr or ""
        lines = [line for line in output.splitlines() if line.strip()]
        return VerifyResult(
            name=cmd[0],
            ok=proc.returncode == 0,
            summary="\n".join(lines[-5:]),
            returncode=proc.returncode,
            output=output,
        )

    def verify(self, cfg: VerifyConfig) -> dict:
        checks = []
        if cfg.run_tests:
            checks.append(self.run(cfg.test_cmd))
        if cfg.run_lint:
            checks.append(self.run(cfg.lint_cmd))
        return {
            "ok": all(check.ok for check in checks),
            "checks": [check.to_dict() for check in checks],
        }
