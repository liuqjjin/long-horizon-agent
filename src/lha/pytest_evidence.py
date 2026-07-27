"""Nonce-bound Pytest execution evidence shared by gates and scorers.

The receipt proves that the trusted driver regained control after ``pytest.main``
returned and that the executed node IDs match a separately collected inventory.
It prevents candidate output or a candidate-writable json-report from being
accepted as success. It is not a privilege boundary against arbitrary Python
running with the same UID; untrusted repositories still require containment.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .sandbox import ExecutionBackend

EVIDENCE_SCHEMA = 1
RECEIPT_MARKER = "LHA_SCORER_RECEIPT"
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_NODEIDS = 100_000
MAX_REPORTS = 300_000
MAX_NODEID_CHARS = 4_096

_DRIVER = r"""
import hashlib as _hashlib
import json as _json
import os as _os
import sys as _sys

_cfg = _json.loads(_sys.stdin.read())
if not _cfg["autoload_plugins"]:
    _os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest as _pytest

_nonce = _cfg["nonce"]
_mode = _cfg["mode"]
_report_name = _cfg["report_name"]
# ``-I`` protects interpreter startup. Add the repository only after the
# trusted Pytest runner and driver dependencies are retained above.
_sys.path.insert(0, _os.getcwd())
_open = _os.open
_write = _os.write
_close = _os.close
_sha256 = _hashlib.sha256
_dumps = _json.dumps


class _Recorder:
    def __init__(self):
        self.collected = []
        self.collection_failures = 0
        self.reports = []

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_failures += 1

    def pytest_runtest_logreport(self, report):
        self.reports.append(
            {
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "wasxfail": bool(getattr(report, "wasxfail", False)),
            }
        )


_recorder = _Recorder()
_args = [
    "-q",
    "--tb=no",
    "--continue-on-collection-errors",
    "--disable-warnings",
    "--color=no",
    "-p",
    "no:cacheprovider",
]
if _mode == "collect":
    _args.insert(0, "--collect-only")
_exit_code = int(_pytest.main(_args, plugins=[_recorder]))
_receipt = {
    "schema_version": 1,
    "nonce": _nonce,
    "mode": _mode,
    "pytest_exit_code": _exit_code,
    "collected": _recorder.collected,
    "collection_failures": _recorder.collection_failures,
    "reports": _recorder.reports,
}
_payload = _dumps(
    _receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode()
_digest = _sha256(_payload).hexdigest()
_fd = _open(_report_name, _os.O_WRONLY | _os.O_CREAT | _os.O_EXCL, 0o600)
try:
    _write(_fd, _payload)
finally:
    _close(_fd)
_write(1, ("\nLHA_SCORER_RECEIPT " + _nonce + " " + _digest + "\n").encode())
raise SystemExit(_exit_code)
""".strip()


class PytestEvidenceOutcome(str, Enum):
    PASS = "PASS"
    TEST_FAIL = "TEST_FAIL"
    INFRA_ERROR = "INFRA_ERROR"


@dataclass(frozen=True)
class DriverResult:
    returncode: int
    receipt: dict[str, Any] | None
    receipt_sha256: str
    detail: str
    duration_s: float


@dataclass(frozen=True)
class InventoryResult:
    expected_nodeids: tuple[str, ...]
    protocol_valid: bool
    collection_failed: bool
    driver: DriverResult

    @property
    def ready(self) -> bool:
        return self.protocol_valid and not self.collection_failed


@dataclass(frozen=True)
class PytestEvidenceResult:
    outcome: PytestEvidenceOutcome
    returncode: int
    detail: str
    expected_nodeids: tuple[str, ...]
    passed_tests: int
    receipt: dict[str, Any] | None
    receipt_sha256: str
    duration_s: float


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    """Serialize scorer receipts and evidence without changing schema-v1 bytes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def valid_driver_receipt(
    receipt: Any,
    *,
    nonce: str,
    mode: str,
) -> bool:
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "schema_version",
            "nonce",
            "mode",
            "pytest_exit_code",
            "collected",
            "collection_failures",
            "reports",
        }
        or receipt.get("schema_version") != EVIDENCE_SCHEMA
        or receipt.get("nonce") != nonce
        or len(nonce) != 48
        or any(character not in "0123456789abcdef" for character in nonce)
        or receipt.get("mode") != mode
        or type(receipt.get("pytest_exit_code")) is not int
        or type(receipt.get("collection_failures")) is not int
        or receipt["collection_failures"] < 0
    ):
        return False
    collected = receipt.get("collected")
    reports = receipt.get("reports")
    if (
        not isinstance(collected, list)
        or len(collected) > MAX_NODEIDS
        or not all(
            isinstance(nodeid, str)
            and 0 < len(nodeid) <= MAX_NODEID_CHARS
            for nodeid in collected
        )
        or len(collected) != len(set(collected))
        or not isinstance(reports, list)
        or len(reports) > MAX_REPORTS
    ):
        return False
    for report in reports:
        if (
            not isinstance(report, dict)
            or set(report) != {"nodeid", "when", "outcome", "wasxfail"}
            or not isinstance(report.get("nodeid"), str)
            or not report["nodeid"]
            or len(report["nodeid"]) > MAX_NODEID_CHARS
            or report.get("when") not in {"setup", "call", "teardown"}
            or report.get("outcome") not in {"passed", "failed", "skipped"}
            or not isinstance(report.get("wasxfail"), bool)
        ):
            return False
    return True


def clear_pytest_caches(workdir: Path) -> None:
    """Remove timestamp-based bytecode before each independent Pytest phase."""
    for cache in workdir.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def run_driver(
    workdir: Path,
    backend: ExecutionBackend,
    *,
    mode: str,
    timeout: float = 300.0,
    autoload_plugins: bool = False,
) -> DriverResult:
    if mode not in {"collect", "run"}:
        raise ValueError(f"unsupported Pytest evidence mode: {mode!r}")
    nonce = secrets.token_hex(24)
    report_name = f".lha-scorer-{secrets.token_hex(16)}.json"
    report_path = workdir / report_name
    report_path.unlink(missing_ok=True)
    result = backend.run(
        [backend.python(), "-I", "-c", _DRIVER],
        cwd=workdir,
        timeout=timeout,
        input=json.dumps(
            {
                "nonce": nonce,
                "mode": mode,
                "report_name": report_name,
                "autoload_plugins": autoload_plugins,
            },
            sort_keys=True,
        ),
    )
    try:
        if report_path.is_symlink():
            payload = b""
        else:
            with report_path.open("rb") as stream:
                payload = stream.read(MAX_RECEIPT_BYTES + 1)
            if len(payload) > MAX_RECEIPT_BYTES:
                payload = b""
    except OSError:
        payload = b""
    finally:
        report_path.unlink(missing_ok=True)
    digest = hashlib.sha256(payload).hexdigest() if payload else ""
    marker = f"{RECEIPT_MARKER} {nonce} {digest}"
    marker_count = sum(line.strip() == marker for line in result.stdout.splitlines())
    try:
        receipt = json.loads(payload) if payload else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        receipt = None
    valid = (
        bool(payload)
        and marker_count == 1
        and valid_driver_receipt(receipt, nonce=nonce, mode=mode)
        and isinstance(receipt, dict)
        and result.returncode == receipt["pytest_exit_code"]
    )
    if not valid:
        return DriverResult(
            result.returncode,
            None,
            "",
            "missing or inconsistent control-plane receipt",
            result.duration_s,
        )
    return DriverResult(
        result.returncode,
        receipt,
        digest,
        "valid control-plane receipt",
        result.duration_s,
    )


def collect_inventory(
    workdir: Path,
    backend: ExecutionBackend,
    *,
    timeout: float = 300.0,
    autoload_plugins: bool = False,
) -> InventoryResult:
    """Collect test node IDs before the execution phase.

    A normal collection error is retained separately from a protocol error so
    the main verifier can report broken candidate imports as ``TEST_FAIL``.
    A pristine benchmark scorer may instead reject any non-ready inventory as
    infrastructure failure.
    """
    clear_pytest_caches(workdir)
    driver = run_driver(
        workdir,
        backend,
        mode="collect",
        timeout=timeout,
        autoload_plugins=autoload_plugins,
    )
    receipt = driver.receipt
    if receipt is None or receipt["reports"]:
        return InventoryResult((), False, False, driver)
    expected = tuple(receipt["collected"])
    collection_failures = receipt["collection_failures"]
    if driver.returncode == 0 and collection_failures == 0 and expected:
        return InventoryResult(expected, True, False, driver)
    if driver.returncode in {1, 2} and collection_failures > 0:
        return InventoryResult(expected, True, True, driver)
    return InventoryResult(expected, False, False, driver)


def classify_receipt(
    *,
    process_returncode: int,
    receipt: dict[str, Any],
    expected_nodeids: tuple[str, ...],
) -> tuple[PytestEvidenceOutcome, int]:
    """Cross-check process exit, the run receipt, and the collected inventory."""
    if process_returncode != receipt["pytest_exit_code"] or receipt["mode"] != "run":
        return PytestEvidenceOutcome.INFRA_ERROR, 0
    collected = tuple(receipt["collected"])
    collection_failures = receipt["collection_failures"]
    reports = receipt["reports"]
    if process_returncode in {1, 2} and collection_failures > 0:
        # A candidate syntax/import error is a wrong artifact. A receipt is still
        # mandatory, so early process exit cannot be misclassified this way.
        return PytestEvidenceOutcome.TEST_FAIL, 0
    if (
        not expected_nodeids
        or len(expected_nodeids) > MAX_NODEIDS
        or len(expected_nodeids) != len(set(expected_nodeids))
        or any(
            not isinstance(nodeid, str)
            or not nodeid
            or len(nodeid) > MAX_NODEID_CHARS
            for nodeid in expected_nodeids
        )
    ):
        return PytestEvidenceOutcome.INFRA_ERROR, 0
    if collected != expected_nodeids or collection_failures:
        return PytestEvidenceOutcome.INFRA_ERROR, 0
    report_nodeids = {report["nodeid"] for report in reports}
    if not report_nodeids.issubset(set(expected_nodeids)):
        return PytestEvidenceOutcome.INFRA_ERROR, 0
    call_reports = [report for report in reports if report["when"] == "call"]
    passed = sum(
        report["outcome"] == "passed" and not report["wasxfail"]
        for report in call_reports
    )
    failed = any(report["outcome"] == "failed" for report in reports)
    calls_by_nodeid: dict[str, list[dict[str, Any]]] = {}
    for report in call_reports:
        calls_by_nodeid.setdefault(report["nodeid"], []).append(report)
    complete_pass = len(call_reports) == len(expected_nodeids) and all(
        len(calls_by_nodeid.get(nodeid, [])) == 1
        and calls_by_nodeid[nodeid][0]["outcome"] == "passed"
        and calls_by_nodeid[nodeid][0]["wasxfail"] is False
        for nodeid in expected_nodeids
    )
    if process_returncode == 0 and complete_pass and not failed:
        return PytestEvidenceOutcome.PASS, passed
    if process_returncode == 1 and failed:
        return PytestEvidenceOutcome.TEST_FAIL, passed
    return PytestEvidenceOutcome.INFRA_ERROR, passed


def run_with_evidence(
    workdir: Path,
    backend: ExecutionBackend,
    *,
    expected_nodeids: tuple[str, ...],
    timeout: float = 300.0,
    autoload_plugins: bool = False,
) -> PytestEvidenceResult:
    clear_pytest_caches(workdir)
    driver = run_driver(
        workdir,
        backend,
        mode="run",
        timeout=timeout,
        autoload_plugins=autoload_plugins,
    )
    if driver.receipt is None:
        return PytestEvidenceResult(
            PytestEvidenceOutcome.INFRA_ERROR,
            driver.returncode,
            driver.detail,
            expected_nodeids,
            0,
            None,
            "",
            driver.duration_s,
        )
    outcome, passed = classify_receipt(
        process_returncode=driver.returncode,
        receipt=driver.receipt,
        expected_nodeids=expected_nodeids,
    )
    return PytestEvidenceResult(
        outcome,
        driver.returncode,
        driver.detail,
        expected_nodeids,
        passed,
        driver.receipt,
        driver.receipt_sha256,
        driver.duration_s,
    )


def validate_evidence(
    evidence: Any,
) -> tuple[PytestEvidenceOutcome, int, int]:
    """Validate a persisted schema-v1 evidence envelope and reclassify it."""
    if (
        not isinstance(evidence, dict)
        or set(evidence)
        != {
            "schema_version",
            "expected_nodeids",
            "process_returncode",
            "receipt",
            "receipt_sha256",
            "classification",
        }
        or evidence.get("schema_version") != EVIDENCE_SCHEMA
        or type(evidence.get("process_returncode")) is not int
        or not isinstance(evidence.get("expected_nodeids"), list)
        or len(evidence["expected_nodeids"]) > MAX_NODEIDS
        or not all(
            isinstance(nodeid, str)
            and 0 < len(nodeid) <= MAX_NODEID_CHARS
            for nodeid in evidence["expected_nodeids"]
        )
        or len(evidence["expected_nodeids"]) != len(set(evidence["expected_nodeids"]))
        or not isinstance(evidence.get("receipt"), dict)
        or not isinstance(evidence.get("receipt_sha256"), str)
        or len(evidence["receipt_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in evidence["receipt_sha256"]
        )
    ):
        raise ValueError("invalid scorer evidence envelope")
    receipt = evidence["receipt"]
    nonce = receipt.get("nonce") if isinstance(receipt, dict) else None
    if (
        not isinstance(nonce, str)
        or not valid_driver_receipt(receipt, nonce=nonce, mode="run")
        or hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
        != evidence["receipt_sha256"]
    ):
        raise ValueError("invalid scorer receipt")
    expected = tuple(evidence["expected_nodeids"])
    outcome, passed = classify_receipt(
        process_returncode=evidence["process_returncode"],
        receipt=receipt,
        expected_nodeids=expected,
    )
    if evidence.get("classification") != outcome.value:
        raise ValueError("stale scorer evidence classification")
    return outcome, len(expected), passed
