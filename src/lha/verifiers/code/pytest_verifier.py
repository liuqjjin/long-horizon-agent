"""Code verifier: run pytest and report structured pass/fail."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..base import Verifier, VerifyContext
from ..verdict import Check


class PytestVerifier(Verifier):
    name = "pytest"
    family = "code"

    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        # The report path is relative so it works identically on the host and
        # inside a container (cwd is the workdir either way); it is read back
        # from the host side of the workdir mount.
        report = Path(ctx.workdir).resolve() / ".lha_pytest.json"
        report.unlink(missing_ok=True)
        # Clear bytecode caches first: a repair often rewrites a file to the same
        # byte-size within the same mtime-second, and Python validates .pyc on
        # (mtime, size) — a stale cache would make the verifier test the OLD code and
        # the repair loop never converge. Recompiling from source keeps verify honest.
        for cache in Path(ctx.workdir).rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        res = ctx.exec.run(
            [
                ctx.exec.python(),
                "-m",
                "pytest",
                "--json-report",
                "--json-report-file=.lha_pytest.json",
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
        skipped_n = int(summary.get("skipped", 0))
        collected = int(summary.get("total", passed_n + failed_n + error_n))
        bad = [
            t
            for t in data.get("tests", [])
            if t.get("outcome") not in ("passed", "skipped", "xfailed")
        ]
        failing = [t.get("nodeid") for t in bad]
        # Surface the actual assertion message of the first few failures. This is the
        # repair loop's fuel: it flows into the Verdict's failures -> the next
        # Implementer prompt, so the model fixes the real defect instead of guessing
        # from "tests failed". Compact by design (ACI): a short reason, not a dump.
        messages = [m for t in bad if (m := _failure_message(t))]
        count_line = f"{passed_n} passed, {failed_n} failed, {error_n} error"
        detail_summary = count_line
        if collected == 0:
            detail_summary += " (no tests collected)"
        elif passed_n == 0 and skipped_n == collected:
            detail_summary += " (all tests skipped)"
        elif messages:
            detail_summary += " — " + "; ".join(messages[:3])

        ok = (
            res.returncode == 0
            and failed_n == 0
            and error_n == 0
            and collected > 0
            and passed_n > 0
        )
        return Check(
            name=self.name,
            family=self.family,
            passed=ok,
            score=float(passed_n),
            detail={
                "summary": detail_summary,
                "failing": failing[:10],
                "messages": messages[:10],
                "returncode": res.returncode,
            },
            duration_s=res.duration_s,
        )


def _failure_message(test: dict, limit: int = 160) -> str | None:
    """A compact one-line reason for a failing test, from the json-report entry."""
    nodeid = str(test.get("nodeid", "?")).split("::")[-1]
    longrepr = ""
    for phase in ("call", "setup", "teardown"):
        rep = test.get(phase) or {}
        crash = rep.get("crash") or {}
        longrepr = crash.get("message") or rep.get("longrepr") or ""
        if longrepr:
            break
    if not longrepr:
        return None
    # The most informative line of an assertion repr is usually the last "E   ..." line.
    line = ""
    for raw in str(longrepr).splitlines():
        s = raw.strip()
        if s.startswith("E "):
            line = s[2:].strip()
        elif s and not line:
            line = s
    line = line or str(longrepr).strip().splitlines()[-1]
    if len(line) > limit:
        line = line[:limit] + "…"
    return f"{nodeid}: {line}"
