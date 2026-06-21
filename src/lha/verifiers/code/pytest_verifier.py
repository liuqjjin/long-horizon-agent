"""Code verifier: run pytest and report structured pass/fail."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ...tools.shell import run
from ..base import Verifier, VerifyContext
from ..verdict import Check


class PytestVerifier(Verifier):
    name = "pytest"
    family = "code"

    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        # Absolute path: pytest runs with cwd=workdir, so a relative report path
        # would resolve nested under the workdir and be unreadable here.
        report = Path(ctx.workdir).resolve() / ".lha_pytest.json"
        report.unlink(missing_ok=True)
        res = run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--json-report",
                f"--json-report-file={report}",
                "-q",
            ],
            cwd=ctx.workdir,
            timeout=self.timeout,
        )

        data: dict = {}
        if report.exists():
            try:
                data = json.loads(report.read_text())
            except json.JSONDecodeError:
                data = {}

        summary = data.get("summary", {})
        passed_n = int(summary.get("passed", 0))
        failed_n = int(summary.get("failed", 0))
        error_n = int(summary.get("error", 0))
        collected = int(summary.get("total", passed_n + failed_n + error_n))
        failing = [
            t.get("nodeid")
            for t in data.get("tests", [])
            if t.get("outcome") not in ("passed", "skipped", "xfailed")
        ]

        ok = res.returncode == 0 and failed_n == 0 and error_n == 0 and collected > 0
        return Check(
            name=self.name,
            family=self.family,
            passed=ok,
            score=float(passed_n),
            detail={
                "summary": f"{passed_n} passed, {failed_n} failed, {error_n} error",
                "failing": failing[:10],
                "returncode": res.returncode,
            },
            duration_s=res.duration_s,
        )
