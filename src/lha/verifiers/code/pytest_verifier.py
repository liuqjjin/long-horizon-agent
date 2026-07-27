"""Code verifier: run pytest and report structured pass/fail."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Iterable

from ...pytest_evidence import (
    PytestEvidenceOutcome,
    clear_pytest_caches,
    collect_inventory,
    run_with_evidence,
)
from ..base import Verifier, VerifyContext
from ..verdict import Check

_MAX_DIAGNOSTIC_BYTES = 2 * 1024 * 1024


class PytestVerifier(Verifier):
    name = "pytest"
    family = "code"

    def __init__(self, timeout: float = 300.0, *, isolated_interpreter: bool = False):
        self.timeout = timeout
        self.isolated_interpreter = isolated_interpreter

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        workdir = Path(ctx.workdir).resolve()
        inventory = collect_inventory(
            workdir,
            ctx.exec,
            timeout=self.timeout,
            autoload_plugins=True,
        )
        duration_s = inventory.driver.duration_s
        if not inventory.protocol_valid:
            return self._check(
                passed=False,
                score=0,
                summary="pytest inventory lacked a complete control-plane receipt",
                failing=[],
                messages=[],
                returncode=inventory.driver.returncode,
                outcome=PytestEvidenceOutcome.INFRA_ERROR,
                report_status="not-run",
                collected=0,
                receipt_sha256="",
                duration_s=duration_s,
            )

        evidence = run_with_evidence(
            workdir,
            ctx.exec,
            expected_nodeids=inventory.expected_nodeids,
            timeout=self.timeout,
            autoload_plugins=True,
        )
        duration_s += evidence.duration_s
        receipt = evidence.receipt or {}
        reports = receipt.get("reports", [])
        if not isinstance(reports, list):
            reports = []
        failing = _ordered_nodeids(
            report
            for report in reports
            if isinstance(report, dict) and report.get("outcome") == "failed"
        )
        skipped = _ordered_nodeids(
            report
            for report in reports
            if isinstance(report, dict) and report.get("outcome") == "skipped"
        )
        collection_failures = receipt.get("collection_failures", 0)
        if type(collection_failures) is not int or collection_failures < 0:
            collection_failures = 0

        messages: list[str] = []
        report_status = "not-required"
        if evidence.outcome is PytestEvidenceOutcome.TEST_FAIL:
            # The legacy json-report remains useful as repair feedback, but it
            # cannot change the receipt-based outcome. Candidate-writable JSON
            # is therefore never accepted as proof that tests passed.
            report_status, messages, diagnostic_duration = self._failure_diagnostics(
                workdir,
                ctx,
                allowed_nodeids=set(failing) | set(inventory.expected_nodeids),
            )
            duration_s += diagnostic_duration

        passed_n = evidence.passed_tests
        failed_n = len(failing)
        error_n = collection_failures
        collected = len(inventory.expected_nodeids)
        count_line = f"{passed_n} passed, {failed_n} failed, {error_n} error"
        detail_summary = count_line
        if inventory.collection_failed:
            detail_summary += " (collection failed)"
        elif collected == 0:
            detail_summary += " (no tests collected)"
        elif passed_n == 0 and len(skipped) == collected:
            detail_summary += " (all tests skipped)"
        elif messages:
            detail_summary += " — " + "; ".join(messages[:3])

        return self._check(
            passed=evidence.outcome is PytestEvidenceOutcome.PASS,
            score=passed_n,
            summary=detail_summary,
            failing=failing,
            messages=messages,
            returncode=evidence.returncode,
            outcome=evidence.outcome,
            report_status=report_status,
            collected=collected,
            receipt_sha256=evidence.receipt_sha256,
            duration_s=duration_s,
        )

    def _failure_diagnostics(
        self,
        workdir: Path,
        ctx: VerifyContext,
        *,
        allowed_nodeids: set[str],
    ) -> tuple[str, list[str], float]:
        """Read json-report only for bounded repair feedback after a proven failure."""
        report = workdir / ".lha_pytest.json"
        _unlink_diagnostic(report)
        clear_pytest_caches(workdir)
        command = [ctx.exec.python()]
        if self.isolated_interpreter:
            command.append("-I")
        command.extend(
            [
                "-m",
                "pytest",
                "--json-report",
                "--json-report-file=.lha_pytest.json",
                "--continue-on-collection-errors",
                "-q",
            ]
        )
        result = ctx.exec.run(command, cwd=workdir, timeout=self.timeout)
        data: dict[str, Any] = {}
        status = "missing"
        try:
            metadata = report.lstat()
            if (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_size <= _MAX_DIAGNOSTIC_BYTES
            ):
                raw = json.loads(report.read_bytes())
                if isinstance(raw, dict):
                    data = raw
                    status = "valid"
                else:
                    status = "invalid"
            else:
                status = "invalid"
        except FileNotFoundError:
            pass
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            status = "invalid"
        finally:
            _unlink_diagnostic(report)
        tests = data.get("tests", [])
        if not isinstance(tests, list) or not all(isinstance(test, dict) for test in tests):
            tests = []
            if status == "valid":
                status = "invalid"
        bad = [
            test
            for test in tests
            if test.get("outcome") not in ("passed", "skipped", "xfailed")
            and test.get("nodeid") in allowed_nodeids
        ]
        messages = [message for test in bad if (message := _failure_message(test))]
        return status, messages[:10], result.duration_s

    def _check(
        self,
        *,
        passed: bool,
        score: int,
        summary: str,
        failing: list[str],
        messages: list[str],
        returncode: int,
        outcome: PytestEvidenceOutcome,
        report_status: str,
        collected: int,
        receipt_sha256: str,
        duration_s: float,
    ) -> Check:
        return Check(
            name=self.name,
            family=self.family,
            passed=passed,
            score=float(score),
            detail={
                "summary": summary,
                "failing": failing[:10],
                "messages": messages[:10],
                "returncode": returncode,
                "outcome": outcome.value,
                "report_status": report_status,
                "collected": collected,
                "receipt_sha256": receipt_sha256,
            },
            duration_s=duration_s,
        )


def _ordered_nodeids(reports: Iterable[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    nodeids: list[str] = []
    for report in reports:
        nodeid = report.get("nodeid")
        if isinstance(nodeid, str) and nodeid and nodeid not in seen:
            nodeids.append(nodeid)
            seen.add(nodeid)
    return nodeids


def _unlink_diagnostic(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Diagnostic JSON is never pass evidence. An unwritable or non-file
        # path therefore suppresses feedback without changing the proven result.
        pass


def _failure_message(test: dict, limit: int = 160) -> str | None:
    """A compact one-line reason for a failing test, from the json-report entry."""
    nodeid = str(test.get("nodeid", "?")).split("::")[-1]
    longrepr = ""
    for phase in ("call", "setup", "teardown"):
        raw_rep = test.get(phase)
        rep = raw_rep if isinstance(raw_rep, dict) else {}
        raw_crash = rep.get("crash")
        crash = raw_crash if isinstance(raw_crash, dict) else {}
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
