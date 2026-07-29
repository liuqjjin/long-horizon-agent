"""Code verifier: run pytest and report structured pass/fail."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Iterable

from ...oracle_inventory import (
    OracleInventoryError,
    collect_pytest_inventory_disposable,
    validate_pytest_oracle_inventory,
)
from ...pytest_evidence import (
    PytestEvidenceOutcome,
    clear_pytest_caches,
    run_with_evidence,
)
from ..base import Verifier, VerifyContext
from ..verdict import Check, process_cleanup_failure_detail
from .oracle_snapshot import OracleSnapshot, OracleSnapshotError, capture_oracle_snapshot

_MAX_DIAGNOSTIC_BYTES = 2 * 1024 * 1024


class PytestVerifier(Verifier):
    name = "pytest"
    family = "code"

    def __init__(self, timeout: float = 300.0, *, isolated_interpreter: bool = False):
        self.timeout = timeout
        self.isolated_interpreter = isolated_interpreter

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        workdir = Path(ctx.workdir)
        try:
            oracle = capture_oracle_snapshot(workdir)
        except OracleSnapshotError as error:
            return self._oracle_failure(str(error))

        baseline = ctx.pytest_oracle_inventory
        if baseline is None and ctx.attempt_id is not None:
            return self._oracle_failure(
                "persisted pytest oracle inventory is missing",
                oracle=oracle,
            )
        duration_s = 0.0
        expected_nodeids: tuple[str, ...] = ()
        if baseline is not None:
            try:
                validate_pytest_oracle_inventory(
                    workdir,
                    baseline,
                    allowed_changes=ctx.allowed_oracle_changes,
                )
            except OracleInventoryError as error:
                return self._oracle_failure(
                    f"persisted pytest oracle inventory is invalid ({error})",
                    oracle=oracle,
                )
            if not baseline.nodeids:
                return self._oracle_failure(
                    "persisted pytest oracle inventory contains no tests",
                    oracle=oracle,
                )
            expected_nodeids = baseline.nodeids

        # An explicit task override may intentionally replace a test file and
        # therefore its node IDs. The baseline still protects every unlisted
        # oracle path; only this audited exception recollects after the patch.
        if baseline is None or ctx.allowed_oracle_changes:
            try:
                inventory = collect_pytest_inventory_disposable(
                    workdir,
                    ctx.exec,
                    timeout=self.timeout,
                )
            except OracleInventoryError as error:
                summary = str(error)
                summary = summary.replace(
                    "pytest baseline collection changed repository files",
                    "pytest collection changed protected files",
                    1,
                )
                return self._oracle_failure(summary, oracle=oracle)
            duration_s = inventory.driver.duration_s
            if inventory.driver.cleanup_unconfirmed:
                return self._cleanup_failure(
                    summary="pytest collection process cleanup could not be confirmed",
                    detail=inventory.driver.detail,
                    duration_s=duration_s,
                    oracle=oracle,
                )
            if not inventory.ready:
                collection_failed = (
                    inventory.protocol_valid and inventory.collection_failed
                )
                return self._check(
                    passed=False,
                    score=0,
                    summary=(
                        "pytest collection failed; test execution was not started"
                        if collection_failed
                        else "pytest inventory was not ready for test execution"
                    ),
                    failing=[],
                    messages=[],
                    returncode=inventory.driver.returncode,
                    outcome=(
                        PytestEvidenceOutcome.TEST_FAIL
                        if collection_failed
                        else PytestEvidenceOutcome.INFRA_ERROR
                    ),
                    report_status="not-run",
                    collected=len(inventory.expected_nodeids),
                    receipt_sha256=inventory.driver.receipt_sha256,
                    oracle_snapshot_sha256=oracle.sha256,
                    duration_s=duration_s,
                )
            expected_nodeids = inventory.expected_nodeids

        if not expected_nodeids:
            return self._oracle_failure(
                "pytest inventory contains no executable tests",
                oracle=oracle,
            )
        evidence = run_with_evidence(
            workdir,
            ctx.exec,
            expected_nodeids=expected_nodeids,
            timeout=self.timeout,
            autoload_plugins=False,
        )
        duration_s += evidence.duration_s
        if evidence.cleanup_unconfirmed:
            return self._cleanup_failure(
                summary="pytest execution process cleanup could not be confirmed",
                detail=evidence.detail,
                duration_s=duration_s,
                receipt_sha256=evidence.receipt_sha256,
                oracle=oracle,
            )
        integrity_failure = self._oracle_change(oracle, workdir)
        if integrity_failure is not None:
            return self._oracle_failure(
                f"pytest execution changed protected files ({integrity_failure})",
                returncode=evidence.returncode,
                duration_s=duration_s,
                receipt_sha256=evidence.receipt_sha256,
                oracle=oracle,
            )
        if baseline is not None:
            try:
                validate_pytest_oracle_inventory(
                    workdir,
                    baseline,
                    allowed_changes=ctx.allowed_oracle_changes,
                )
            except OracleInventoryError as error:
                return self._oracle_failure(
                    f"pytest execution changed persisted oracle files ({error})",
                    returncode=evidence.returncode,
                    duration_s=duration_s,
                    receipt_sha256=evidence.receipt_sha256,
                    oracle=oracle,
                )
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
            (
                report_status,
                messages,
                diagnostic_duration,
                cleanup_detail,
            ) = self._failure_diagnostics(
                workdir,
                ctx,
                allowed_nodeids=set(failing) | set(expected_nodeids),
            )
            duration_s += diagnostic_duration
            if cleanup_detail is not None:
                return self._cleanup_failure(
                    summary=(
                        "pytest diagnostic process cleanup could not be confirmed"
                    ),
                    detail=cleanup_detail,
                    duration_s=duration_s,
                    receipt_sha256=evidence.receipt_sha256,
                    oracle=oracle,
                )

        passed_n = evidence.passed_tests
        failed_n = len(failing)
        error_n = collection_failures
        collected = len(expected_nodeids)
        count_line = f"{passed_n} passed, {failed_n} failed, {error_n} error"
        detail_summary = count_line
        if collection_failures:
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
            oracle_snapshot_sha256=oracle.sha256,
            duration_s=duration_s,
            baseline_inventory_sha256=baseline.sha256 if baseline is not None else "",
        )

    @staticmethod
    def _oracle_change(oracle: OracleSnapshot, workdir: Path) -> str | None:
        try:
            current = capture_oracle_snapshot(workdir)
        except OracleSnapshotError as error:
            return str(error)
        return oracle.difference(current)

    def _oracle_failure(
        self,
        summary: str,
        *,
        returncode: int = 126,
        duration_s: float = 0.0,
        receipt_sha256: str = "",
        oracle: OracleSnapshot | None = None,
    ) -> Check:
        return self._check(
            passed=False,
            score=0,
            summary=summary,
            failing=[],
            messages=[],
            returncode=returncode,
            outcome=PytestEvidenceOutcome.INFRA_ERROR,
            report_status="not-run",
            collected=0,
            receipt_sha256=receipt_sha256,
            oracle_snapshot_sha256=oracle.sha256 if oracle is not None else "",
            duration_s=duration_s,
        )

    def _cleanup_failure(
        self,
        *,
        summary: str,
        detail: str,
        duration_s: float,
        receipt_sha256: str = "",
        oracle: OracleSnapshot | None = None,
    ) -> Check:
        return self._check(
            passed=False,
            score=0,
            summary=summary,
            failing=[],
            messages=[],
            returncode=126,
            outcome=PytestEvidenceOutcome.INFRA_ERROR,
            report_status="not-run",
            collected=0,
            receipt_sha256=receipt_sha256,
            oracle_snapshot_sha256=oracle.sha256 if oracle is not None else "",
            duration_s=duration_s,
            cleanup_detail=detail,
        )

    def _failure_diagnostics(
        self,
        workdir: Path,
        ctx: VerifyContext,
        *,
        allowed_nodeids: set[str],
    ) -> tuple[str, list[str], float, str | None]:
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
        if result.cleanup_unconfirmed:
            return (
                "not-run",
                [],
                result.duration_s,
                result.cleanup_detail or result.stderr[-500:],
            )
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
        return status, messages[:10], result.duration_s, None

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
        oracle_snapshot_sha256: str,
        duration_s: float,
        baseline_inventory_sha256: str = "",
        cleanup_detail: str | None = None,
    ) -> Check:
        detail: dict[str, Any] = {
            "summary": summary,
            "failing": failing[:10],
            "messages": messages[:10],
            "returncode": returncode,
            "outcome": outcome.value,
            "report_status": report_status,
            "collected": collected,
            "receipt_sha256": receipt_sha256,
            "oracle_snapshot_sha256": oracle_snapshot_sha256,
            "baseline_inventory_sha256": baseline_inventory_sha256,
        }
        if cleanup_detail is not None:
            detail.update(
                process_cleanup_failure_detail(
                    returncode=returncode,
                    cleanup_unconfirmed=True,
                    detail=cleanup_detail,
                )
            )
        return Check(
            name=self.name,
            family=self.family,
            passed=passed,
            score=float(score),
            detail=detail,
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
