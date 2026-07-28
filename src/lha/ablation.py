"""Verification ablation: measure whether the verifier loop makes a real LLM more reliable.

For each bug-fix task, draw one first attempt from the LLM and score that same attempt
under three conditions (a paired design, so only the verification varies):

  - ``trust``  — apply the first attempt and accept it; no gate, no repair.
  - ``gate``   — apply it, run the internal test gate, refuse (revert, FAIL) on failure.
  - ``verify`` — gate, then repair (re-prompt with the failing-test feedback) up to a budget.

Prediction and truth are produced by different mechanisms:

  - The **internal gate** runs pytest in the agent's working copy. Its outcome is the
    condition's *claim* (a prediction).
  - The **final scorer** freezes the effective source diff (SHA-256 recorded), applies
    it to a fresh copy of the canonical repo — canonical tests restored regardless of
    what happened in the working copy — and runs pytest there through a separate
    execution backend. Its outcome is *truth*.

Every condition is scored by the final scorer, including attempts the gate refused.
``artifact_correct`` records that verdict; ``true_success`` additionally requires
the condition to deliver the artifact. Gate precision/recall/FPR/FNR use artifact
correctness, while end-to-end success uses delivered correctness.

Integrity properties:
  - Leak-free: the implementer is a single-shot completion with file tools denied
    (``no_tools``) and sees only non-test source, so it cannot read the oracle.
  - Protected paths are excluded from the frozen patch; this is an input-policy
    guarantee, not containment against arbitrary same-UID code at runtime.
  - Transient backend errors are retried and, if they persist, recorded as ``ERROR``.
    An ERROR backed by a complete failed-call receipt is sealed as a terminal cell;
    it is excluded from rates but counted and reported, never silently dropped.
  - Cached cells carry a provenance fingerprint over task/corpus bytes, the complete
    ``lha`` source tree, model/CLI settings, scorer identity, runtime versions, and
    repair configuration; any change recomputes.

A weaker implementer ``model`` calibrates difficulty, so first-attempt success lands
in a range where the gate has errors to catch. This runner isolates the gate
mechanism (single-step fix, no planning/context retrieval); it is not the full
harness loop.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from . import __version__
from .ablation_attempts import (
    FORMAL_ABLATION_ATTEMPTS_PATH,
    MAX_FORMAL_ABLATION_ATTEMPTS_BYTES,
    FormalAblationProtocol,
    FormalCodexClientConfig,
    FormalGitCredentialHelper,
    RegisteredAttempt,
    formal_ablation_protocol_sha256,
    formal_ablation_witness_commit_bytes,
    formal_ablation_witness_commit_oid,
    formal_ablation_witness_message,
    formal_attempt_lock,
    formal_codex_client_config_from_runtime,
    formal_codex_client_sha256,
    make_formal_codex_client,
    parse_formal_ablation_attempt_registry,
    validate_formal_witness_remote_url,
)
from .agents.implementer import Implementer
from .artifacts import Patch, Step
from .clock import now
from .config import Config
from .durable_io import atomic_replace_bytes, atomic_replace_text, durable_mkdir_chain
from .live_context.models import ContextBundle, Freshness
from .llm.base import LLMClient
from .pytest_evidence import (
    EVIDENCE_SCHEMA as PYTEST_EVIDENCE_SCHEMA,
)
from .pytest_evidence import (
    canonical_json_bytes,
    classify_receipt,
    collect_inventory,
    run_with_evidence,
    validate_evidence,
)
from .sandbox import DockerBackend, ExecutionBackend, TrustedLocalBackend, make_backend
from .sandbox.base import process_group_cleanup_supported, terminate_process_group
from .sandbox.docker import resolve_docker_executable
from .tasks.spec import TaskSpec
from .tools import policy
from .tools.patch import apply_patch
from .tools.shell import run, sanitized_absolute_path, trusted_executable
from .verifiers import VerifyContext
from .verifiers.code.pytest_verifier import PytestVerifier

logger = logging.getLogger(__name__)

CONDITIONS = [
    ("trust", "apply the first attempt and accept it; no gate, no repair"),
    ("gate", "apply it, run the internal test gate, refuse on failure"),
    ("verify", "gate plus repair loop"),
]

_MAX_REPAIRS = 3
_LLM_RETRIES = 3
_CACHE_SCHEMA = 8
_REPORT_SCHEMA = 4
_FROZEN_ARTIFACT_SCHEMA = 1
_INPUT_SNAPSHOT_SCHEMA = 1
_SCORER_EVIDENCE_SCHEMA = 2
_LLM_CALL_RECEIPT_SCHEMA = 2
_CELL_ATTEMPT_SCHEMA = 1
_FORMAL_CORPUS_MANIFEST_SCHEMA = 1
_FORMAL_OUTPUT_LOCK_NAME = ".formal-ablation.lock"
_FORMAL_RUN_HEADER_NAME = "formal_run.json"
_FORMAL_RUN_HEADER_SCHEMA = 1
_FORMAL_TASK_COUNT = 17
_FORMAL_REPETITIONS = 12
_FORMAL_CORPUS_MANIFEST_PATH = Path("benchmarks/formal_ablation_manifest.json")
_FORMAL_CONTROL_FILES = ("pyproject.toml", "uv.lock", ".python-version")
_BOOTSTRAP_N = 10_000
_READ_CHUNK_BYTES = 64 * 1024
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_SCORER_EVIDENCE_BYTES = 12 * 1024 * 1024
_MAX_LLM_CALL_RECEIPT_BYTES = 512 * 1024
_MAX_CELL_ATTEMPT_BYTES = 4 * 1024
_MAX_FORMAL_RUN_HEADER_BYTES = 4 * 1024
_MAX_FORMAL_MANIFEST_BYTES = 512 * 1024
_MAX_CONTROL_EXECUTABLE_BYTES = 256 * 1024 * 1024
_MAX_CACHE_BYTES = 8 * 1024 * 1024
_MAX_REPORT_BYTES = 32 * 1024 * 1024
_MAX_TASK_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_TEXT_BYTES = 8 * 1024 * 1024
_MAX_FORMAL_HEAD_BYTES = 128 * 1024 * 1024
_MAX_FORMAL_HEAD_FILES = 10_000
_MAX_GH_CONFIG_BYTES = 1024 * 1024
_DOCKER_IMAGE_PROBE_SCHEMA = 1
_DOCKER_IMAGE_PROBE_MARKER = "LHA_DOCKER_IMAGE_PROBE "
_DOCKER_IMAGE_PROBE_SCRIPT = r"""
import importlib.metadata
import json
import os
import pathlib
import sys
import tempfile

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest

with tempfile.TemporaryDirectory(prefix="lha_image_probe_") as raw:
    root = pathlib.Path(raw)
    test_path = root / "test_probe.py"
    report_path = root / "report.json"
    test_path.write_text("def test_probe():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    code = pytest.main(
        [
            "-q",
            "-p",
            "pytest_jsonreport.plugin",
            "--json-report",
            f"--json-report-file={report_path}",
            "-p",
            "no:cacheprovider",
            str(test_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    if code != 0 or summary.get("passed") != 1 or summary.get("total") != 1:
        raise SystemExit("minimal Pytest execution did not pass")

payload = {
    "python_version": ".".join(str(value) for value in sys.version_info[:3]),
    "pytest_version": importlib.metadata.version("pytest"),
    "pytest_json_report_version": importlib.metadata.version("pytest-json-report"),
}
print("LHA_DOCKER_IMAGE_PROBE " + json.dumps(payload, sort_keys=True, separators=(",", ":")))
""".strip()
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_CODEX_EVENT_TYPES = frozenset(
    {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "item.started",
        "item.updated",
        "item.completed",
        "error",
    }
)
_CODEX_TEXT_ITEMS = frozenset({"agent_message", "reasoning", "todo_list"})


class _Transient(Exception):
    """A backend error that should be retried / excluded, not counted as a result."""


class ScoreOutcome(str, Enum):
    """A Pytest measurement, keeping test failures separate from missing evidence."""

    PASS = "PASS"
    TEST_FAIL = "TEST_FAIL"
    INFRA_ERROR = "INFRA_ERROR"


@dataclass(frozen=True)
class PytestResult:
    outcome: ScoreOutcome
    returncode: int | None
    detail: str
    messages: tuple[str, ...] = ()
    evidence_sha256: str = ""
    expected_tests: int = 0
    passed_tests: int = 0

    @property
    def passed(self) -> bool:
        return self.outcome is ScoreOutcome.PASS


@dataclass(frozen=True)
class ScorerEvidenceBinding:
    """Identity that prevents a valid receipt from grading a different cell."""

    task: str
    rep: int
    artifact_sha256: str
    input_snapshot_sha256: str
    scorer_backend: str
    scorer_image_id: str | None


@dataclass
class RunRecord:
    task: str
    condition: str
    rep: int
    status: str  # DONE | FAILED | ERROR
    claimed_success: bool  # the condition's own decision (internal gate for gate/verify)
    artifact_correct: bool  # independent scorer verdict for the frozen patch bytes
    true_success: bool  # the condition delivered an independently correct artifact
    false_success: bool  # delivered, but the artifact is independently incorrect
    repairs: int
    detail: str = ""
    # internal-gate prediction (None for trust, which runs no gate)
    gate_prediction: bool | None = None
    artifact_sha256: str = ""
    # Final Pytest scorer classification. Older reports legitimately omit it.
    scorer_outcome: str | None = None
    scorer_evidence_sha256: str = ""
    scorer_expected_tests: int = 0
    scorer_passed_tests: int = 0


@dataclass
class ConditionStats:
    condition: str
    blurb: str
    n: int
    claimed_success_rate: float
    artifact_correct_rate: float
    true_success_rate: float
    false_success_rate: float
    mean_repairs: float
    errors: int = 0
    # gate-vs-truth confusion (None for trust — no prediction is made)
    tp: int | None = None
    fp: int | None = None
    tn: int | None = None
    fn: int | None = None
    precision: float | None = None
    recall: float | None = None
    fpr: float | None = None
    fnr: float | None = None
    # task-cluster bootstrap 95% CIs
    artifact_ci: tuple[float, float] | None = None
    true_ci: tuple[float, float] | None = None
    false_ci: tuple[float, float] | None = None


@dataclass
class AblationProvenance:
    """Public, secret-free inputs needed to reproduce or audit a report."""

    schema_version: int = 1
    generated_at: str | None = None
    harness_version: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    source_tree_sha256: str | None = None
    source_files: dict[str, str] = field(default_factory=dict)
    requested_llm_backend: str = "unknown"
    actual_llm_backend: str = "unknown"
    model: str | None = None
    cli_version: str | None = None
    cli_executable_sha256: str | None = None
    backend_library_version: str | None = None
    reasoning_effort: str | None = None
    backend_details: str | None = None
    agent_backend: str = "unknown"
    scorer_requested: str = "unknown"
    scorer_backend: str = "unknown"
    scorer_image: str | None = None
    scorer_image_id: str | None = None
    platform: str | None = None
    python_version: str | None = None
    pytest_version: str | None = None
    pytest_json_report_version: str | None = None
    runtime_packages: dict[str, str | None] = field(default_factory=dict)
    task_paths: dict[str, str] = field(default_factory=dict)
    corpus_paths: dict[str, str] = field(default_factory=dict)
    task_files_sha256: dict[str, str] = field(default_factory=dict)
    corpus_sha256: dict[str, str] = field(default_factory=dict)
    input_snapshot_sha256: dict[str, str] = field(default_factory=dict)
    cell_fingerprints: dict[str, str] = field(default_factory=dict)
    formal_corpus_manifest_path: str | None = None
    formal_corpus_manifest_sha256: str | None = None
    preregistration_commit: str | None = None
    formal_attempt_id: str | None = None
    formal_attempt_registry_path: str | None = None
    formal_attempt_registry_sha256: str | None = None
    formal_attempt_protocol_sha256: str | None = None
    formal_attempt_registration_commit: str | None = None
    formal_attempt_witness_remote_name: str | None = None
    formal_attempt_witness_remote_url: str | None = None
    formal_attempt_witness_ref: str | None = None
    formal_attempt_witness_commit: str | None = None
    formal_run_header_path: str | None = None
    formal_run_header_sha256: str | None = None
    formal_outcome_key: str | None = None
    # Historical run provenance. Release validation checks the binding and
    # report fingerprint; it does not require another host to have these bytes.
    git_executable: dict[str, Any] = field(default_factory=dict)
    docker_executable: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationReport:
    llm: str
    model: str
    reps: int
    tasks: list[str]
    records: list[RunRecord] = field(default_factory=list)
    stats: list[ConditionStats] = field(default_factory=list)
    scorer: str = "trusted-local"
    fingerprint: str = ""
    backend_version: str = ""
    schema_version: int = _REPORT_SCHEMA
    provenance: AblationProvenance | None = None
    llm_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_markdown(self) -> str:
        model = self.model or "(backend default)"
        lines = [
            "# Verification ablation",
            "",
            f"implementer: `{self.llm}`"
            + (f" ({self.backend_version})" if self.backend_version else "")
            + f" · model: `{model}` · tasks: {len(self.tasks)} · "
            f"repetitions: {self.reps} · paired (trust/gate score the same attempt) · "
            f"final scorer: `{self.scorer}` (fresh copy, canonical tests, independent of "
            "the internal gate)",
            "",
            "| condition | n | delivered | artifact correct (95% CI) "
            "| delivered correct (95% CI) | delivered wrong (95% CI) "
            "| mean repairs | errors |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in self.stats:
            lines.append(
                f"| `{s.condition}` | {s.n} | {_pct(s.claimed_success_rate)} | "
                f"{_pct(s.artifact_correct_rate)}{_ci(s.artifact_ci)} | "
                f"{_pct(s.true_success_rate)}{_ci(s.true_ci)} | "
                f"{_pct(s.false_success_rate)}{_ci(s.false_ci)} | "
                f"{s.mean_repairs:.2f} | {s.errors} |"
            )
        lines.append("")
        if self.schema_version >= 3:
            lines += [
                "`n` counts usable measurements. `errors` stay in the scheduled denominator "
                "(`n + errors`) and are never relabelled as incorrect patches.",
                "",
            ]
        lines.append("Conditions:")
        for name, blurb in CONDITIONS:
            lines.append(f"- `{name}` — {blurb}.")
        gate_lines = _gate_quality_lines(self.stats)
        if gate_lines:
            lines += ["", "Internal gate vs artifact correctness (per attempt):", *gate_lines]
        mcnemar_lines = _paired_mcnemar_lines(
            self.records,
            precise=self.schema_version >= 4,
        )
        if mcnemar_lines:
            lines += ["", "Paired contrasts:", *mcnemar_lines]
        summary = _summary(self.stats)
        if summary:
            lines += ["", summary]
        lines += ["", "## Per-task outcomes", ""]
        lines += ["| task | trust | gate | verify |", "|---|---|---|---|"]
        for task in self.tasks:
            cells = [_task_cell(self.records, task, c) for c in ("trust", "gate", "verify")]
            lines.append(f"| `{task}` | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines += [
            "",
            "Legend: pass = delivered and independently correct · fail = not delivered · "
            "false-pass = delivered but independently wrong. Artifact correctness is "
            "reported separately from delivery.",
            "",
        ]
        if self.provenance is not None:
            git_commit = self.provenance.git_commit or "unknown"
            dirty = (
                "unknown"
                if self.provenance.git_dirty is None
                else ("yes" if self.provenance.git_dirty else "no")
            )
            lines += [
                "Provenance:",
                f"- source: `{self.provenance.source_tree_sha256 or 'unknown'}`",
                f"- git: `{git_commit}` · dirty: `{dirty}`",
            ]
            if (
                self.provenance.formal_attempt_id
                and self.provenance.formal_attempt_registration_commit
                and self.provenance.formal_attempt_registry_path
            ):
                lines += [
                    f"- formal attempt: `{self.provenance.formal_attempt_id}`",
                    "- registration: "
                    f"`{self.provenance.formal_attempt_registration_commit}` · "
                    f"registry: `{self.provenance.formal_attempt_registry_path}`",
                ]
            lines += [
                f"- runtime: Python `{self.provenance.python_version or 'unknown'}` · "
                f"pytest `{self.provenance.pytest_version or 'unknown'}`",
                f"- LLM call audits: {len(self.llm_calls)} · "
                f"loaded from cell cache: "
                f"{sum(bool(call.get('cache_hit')) for call in self.llm_calls)}",
                "",
            ]
        return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return ""
    return f" ({_pct(ci[0])}–{_pct(ci[1])})"


def _task_cell(records: list[RunRecord], task: str, cond: str) -> str:
    all_recs = [r for r in records if r.task == task and r.condition == cond]
    recs = [r for r in all_recs if r.status != "ERROR"]
    if not recs:
        return f"error {len(all_recs)}/{len(all_recs)}" if all_recs else "—"

    def sym(r: RunRecord) -> str:
        return "false-pass" if r.false_success else ("pass" if r.true_success else "fail")

    syms = [sym(r) for r in recs]
    counts = [
        f"{name} {syms.count(name)}/{len(all_recs)}"
        for name in ("pass", "fail", "false-pass")
        if syms.count(name)
    ]
    errors = len(all_recs) - len(recs)
    if errors:
        counts.append(f"error {errors}/{len(all_recs)}")
    return " · ".join(counts)


def _gate_quality_lines(stats: list[ConditionStats]) -> list[str]:
    out = []
    for s in stats:
        if s.tp is None:
            continue
        out.append(
            f"- `{s.condition}`: TP={s.tp} FP={s.fp} TN={s.tn} FN={s.fn} · "
            f"precision={_opt_pct(s.precision)} recall={_opt_pct(s.recall)} "
            f"FPR={_opt_pct(s.fpr)} FNR={_opt_pct(s.fnr)}"
        )
    return out


def _opt_pct(x: float | None) -> str:
    return "n/a" if x is None else _pct(x)


def _summary(stats: list[ConditionStats]) -> str:
    by = {s.condition: s for s in stats}
    trust, gate, verify = by.get("trust"), by.get("gate"), by.get("verify")
    if not (trust and gate and verify):
        return ""
    return (
        f"Without the gate, {_pct(trust.false_success_rate)} of accepted fixes are wrong "
        f"(scorer-graded); the gate (same attempts) reduces that to "
        f"{_pct(gate.false_success_rate)}, and the repair loop raises true success from "
        f"{_pct(gate.true_success_rate)} to {_pct(verify.true_success_rate)}. "
        f"The gate discarded {gate.fn or 0} independently correct artifact(s) "
        "(false negatives)."
    )


# --- core mechanics ---------------------------------------------------------
_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".pytest_cache", ".lha_pytest.json", ".cocoindex_code", ".git"
)
_DIFF_IGNORE = {"__pycache__", ".pytest_cache", ".lha_pytest.json", ".cocoindex_code", ".git"}


def _stable_file_signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _consume_stable_regular_file(
    path: Path,
    consume: Callable[[bytes], object],
    *,
    max_bytes: int | None = None,
    reject_hardlinks: bool = True,
) -> None:
    """Read one stable regular file without following links or growing unbounded."""
    path_before = path.lstat()
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{path} is not a regular file")
    if reject_hardlinks and path_before.st_nlink != 1:
        raise ValueError(f"{path} has multiple hard links")
    if max_bytes is not None and path_before.st_size > max_bytes:
        raise ValueError(f"{path} exceeds the {max_bytes}-byte limit")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stable_file_signature(
            opened
        ) != _stable_file_signature(path_before):
            raise ValueError(f"{path} changed before it could be read")
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError(f"{path} exceeds the {max_bytes}-byte limit")
            consume(chunk)
        descriptor_after = os.fstat(descriptor)
        path_after = path.lstat()
        expected = _stable_file_signature(opened)
        if (
            _stable_file_signature(descriptor_after) != expected
            or _stable_file_signature(path_after) != expected
            or total != opened.st_size
        ):
            raise ValueError(f"{path} changed while it was being read")
    finally:
        os.close(descriptor)


def _read_bounded_bytes(
    path: Path,
    *,
    max_bytes: int,
    reject_hardlinks: bool = True,
) -> bytes:
    buffer = bytearray()
    _consume_stable_regular_file(
        path,
        buffer.extend,
        max_bytes=max_bytes,
        reject_hardlinks=reject_hardlinks,
    )
    return bytes(buffer)


def _read_bounded_text(
    path: Path,
    *,
    max_bytes: int,
    errors: str = "strict",
    reject_hardlinks: bool = True,
) -> str:
    return _read_bounded_bytes(
        path,
        max_bytes=max_bytes,
        reject_hardlinks=reject_hardlinks,
    ).decode("utf-8", errors=errors)


def _sha256_regular_file(path: Path, *, reject_hardlinks: bool = False) -> str:
    digest = hashlib.sha256()
    _consume_stable_regular_file(
        path,
        digest.update,
        reject_hardlinks=reject_hardlinks,
    )
    return digest.hexdigest()


def _sanitize(patch: Patch) -> Patch:
    """Keep only source edits, so a patch can never rewrite the test oracle or config.

    Uses the same protected-file policy the run loop enforces (tools.policy);
    the ablation strips rather than rejects because the paired design scores
    whatever source edits the first attempt made.
    """
    return policy.strip_protected(patch)


def _empty_bundle() -> ContextBundle:
    return ContextBundle(query="", freshness=Freshness(index_version="none", indexed_at=now()))


def _fix_step(task: TaskSpec) -> Step:
    return Step(
        step_id="s2-fix",
        kind="code",
        action="edit_code",
        goal=task.description or task.title,
        verifiers=["pytest"],
    )


def _copy_repo(source: Path, dst: Path, *, include_tests: bool) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(source, dst, ignore=_IGNORE)
    # Content-addressed input snapshots are read-only. Working copies must still
    # be writable so patches never target the snapshot itself.
    for path in dst.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    if not include_tests:
        shutil.rmtree(dst / "tests", ignore_errors=True)


class _PromptAuditClient(LLMClient):
    """Record the exact prompt and response bytes without persisting either.

    The wrapper deliberately uses ``LLMClient.propose_patch`` so the hash covers
    the same system and user strings sent to Codex. The inner client's process,
    usage, and protocol metadata remain the source of truth.
    """

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__) or "unknown"
        self.last_prompt_sha256: str | None = None
        self.last_response_sha256: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def complete(self, system: str, prompt: str) -> str:
        self.last_prompt_sha256 = _canonical_digest(
            {
                "schema_version": 1,
                "system": system,
                "prompt": prompt,
            }
        )
        self.last_response_sha256 = None
        response = self.inner.complete(system, prompt)
        if not isinstance(response, str):
            raise TypeError("LLM completion must return text")
        self.last_response_sha256 = _sha256_bytes(response.encode("utf-8"))
        return response


def _patch_sha256(patch: Patch) -> str:
    return _canonical_digest(
        {
            "schema_version": 1,
            "patch": patch.model_dump(mode="json"),
        }
    )


def _safe_call_audit(
    llm: LLMClient, *, label: str, status: str, error: Exception | None = None
) -> dict[str, Any]:
    """Copy the backend's call metadata without prompts, responses, paths, or credentials."""
    raw = getattr(llm, "last_call", None)
    audit: dict[str, Any] = {
        "label": label,
        "status": status,
        "backend": getattr(llm, "name", type(llm).__name__) or "unknown",
    }
    if isinstance(raw, dict):
        for name in (
            "status",
            "cli_version",
            "model",
            "reasoning_effort",
            "sandbox_mode",
            "permission_model",
            "permission_profile",
            "credential_barrier",
            "cli_executable_sha256",
            "cli_executable_trusted",
            "externally_sandboxed",
            "retries",
            "attempt_count",
            "duration_s",
            "event_summary",
            "error_type",
            "retryable",
        ):
            if name in raw:
                audit[name] = raw[name]
        attempts = raw.get("attempts")
        if isinstance(attempts, list):
            audit["attempts"] = [
                {
                    name: attempt[name]
                    for name in (
                        "attempt",
                        "status",
                        "duration_s",
                        "error_type",
                        "event_summary",
                    )
                    if isinstance(attempt, dict) and name in attempt
                }
                for attempt in attempts
                if isinstance(attempt, dict)
            ]
    usage = getattr(llm, "last_usage", None)
    if isinstance(usage, dict):
        audit["usage"] = {
            name: usage.get(name)
            for name in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "cost_usd",
                "model",
            )
        }
    prompt_sha256 = getattr(llm, "last_prompt_sha256", None)
    response_sha256 = getattr(llm, "last_response_sha256", None)
    if isinstance(prompt_sha256, str):
        audit["prompt_sha256"] = prompt_sha256
    if isinstance(response_sha256, str):
        audit["response_sha256"] = response_sha256
    if error is not None and "error_type" not in audit:
        audit["error_type"] = type(error).__name__
        audit["retryable"] = bool(getattr(error, "retryable", False))
    # Backends own their metadata types. A non-JSON value is omitted rather than
    # stringified because reprs can contain paths or other process-local details.
    try:
        return json.loads(json.dumps(audit))
    except (TypeError, ValueError):
        return {
            "label": label,
            "status": status,
            "backend": getattr(llm, "name", type(llm).__name__) or "unknown",
            "metadata": None,
        }


def _retry(
    fn,
    label: str,
    *,
    llm: LLMClient | None = None,
    audit_log: list[dict[str, Any]] | None = None,
):
    last: Exception | None = None
    for i in range(_LLM_RETRIES):
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001 - backend types declare retry safety
            if llm is not None and audit_log is not None:
                audit_log.append(_safe_call_audit(llm, label=label, status="failed", error=e))
            last = e
            if getattr(e, "retryable", None) is False:
                raise _Transient(f"{label}: non-retryable {type(e).__name__}: {e}") from e
            logger.warning("transient LLM error [%s] %d/%d: %s", label, i + 1, _LLM_RETRIES, e)
            time.sleep(2 * (i + 1))
        else:
            if llm is not None and audit_log is not None:
                audit = _safe_call_audit(llm, label=label, status="succeeded")
                if isinstance(result, Patch):
                    audit["patch_sha256"] = _patch_sha256(result)
                audit_log.append(audit)
            return result
    raise _Transient(f"{label}: {last}")


def _pytest(
    workdir: Path,
    exec_backend: ExecutionBackend,
    *,
    isolated_interpreter: bool = False,
) -> PytestResult:
    """Run the prediction-side gate and retain an explicit infrastructure state."""
    step = Step(
        step_id="grade", kind="code", action="edit_code", goal="grade", verifiers=["pytest"]
    )
    check = PytestVerifier(isolated_interpreter=isolated_interpreter).verify(
        None, VerifyContext(workdir=workdir, step=step, exec=exec_backend)
    )
    raw_outcome = check.detail.get("outcome")
    try:
        outcome = ScoreOutcome(raw_outcome)
    except ValueError:
        outcome = ScoreOutcome.INFRA_ERROR
    messages = check.detail.get("messages", [])
    return PytestResult(
        outcome=outcome,
        returncode=(
            check.detail.get("returncode") if type(check.detail.get("returncode")) is int else None
        ),
        detail=str(check.detail.get("summary") or "pytest produced no summary"),
        messages=tuple(str(message) for message in messages if isinstance(message, str)),
    )


def _first_attempt(
    llm: LLMClient,
    source: Path,
    task: TaskSpec,
    scratch: Path,
    audit_log: list[dict[str, Any]] | None = None,
) -> Patch:
    """One leak-free first attempt: implement against source with NO tests present."""
    wd = scratch / "attempt"
    _copy_repo(source, wd, include_tests=False)
    patch = _retry(
        lambda: _sanitize(Implementer(llm).implement(_fix_step(task), _empty_bundle(), wd)),
        "first",
        llm=llm,
        audit_log=audit_log,
    )
    return patch


# --- frozen artifact + independent scorer ------------------------------------
def _iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _DIFF_IGNORE for part in rel.parts) or rel.name in _DIFF_IGNORE:
            continue
        yield rel.as_posix(), p


def _frozen_diff(source: Path, wd: Path) -> dict[str, str | None]:
    """The effective source change: workdir files that differ from the pristine repo.

    Protected (oracle/config) paths are excluded — the scorer always grades
    against canonical tests, whatever happened in the working copy. ``None``
    marks a deletion.
    """
    src_files = dict(_iter_files(source))
    wd_files = dict(_iter_files(wd))
    frozen: dict[str, str | None] = {}
    for rel in sorted(set(src_files) | set(wd_files)):
        if policy.is_protected(rel):
            continue
        s, w = src_files.get(rel), wd_files.get(rel)
        if w is None:
            frozen[rel] = None
        else:
            text = _read_bounded_text(
                w,
                max_bytes=_MAX_SOURCE_TEXT_BYTES,
                errors="replace",
                reject_hardlinks=False,
            )
            if (
                s is None
                or _read_bounded_text(
                    s,
                    max_bytes=_MAX_SOURCE_TEXT_BYTES,
                    errors="replace",
                    reject_hardlinks=False,
                )
                != text
            ):
                frozen[rel] = text
    return frozen


def _frozen_artifact_bytes(frozen: dict[str, str | None]) -> bytes:
    """Canonical, content-addressable representation of an effective patch."""
    for rel, content in frozen.items():
        path = PurePosixPath(rel)
        if (
            not rel
            or rel == "."
            or "\x00" in rel
            or path.is_absolute()
            or "\\" in rel
            or path.as_posix() != rel
            or any(part in {"", ".", ".."} for part in path.parts)
            or not (content is None or isinstance(content, str))
        ):
            raise ValueError(f"invalid frozen artifact entry: {rel!r}")
        if policy.is_protected(rel):
            raise ValueError(f"frozen artifact contains protected path: {rel!r}")
    payload = json.dumps(
        {"schema_version": _FROZEN_ARTIFACT_SCHEMA, "files": frozen},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"frozen artifact exceeds the {_MAX_ARTIFACT_BYTES}-byte limit")
    return payload


def _artifact_digest(frozen: dict[str, str | None]) -> str:
    return _sha256_bytes(_frozen_artifact_bytes(frozen))


def _store_frozen_artifact(frozen: dict[str, str | None], artifact_dir: Path) -> str:
    """Persist frozen bytes under their SHA-256 and reject conflicting content."""
    payload = _frozen_artifact_bytes(frozen)
    digest = _sha256_bytes(payload)
    path = artifact_dir / f"{digest}.json"
    if path.exists():
        if _read_bounded_bytes(path, max_bytes=_MAX_ARTIFACT_BYTES) != payload:
            raise RuntimeError(f"frozen artifact store is corrupt for {digest}")
        return digest
    _atomic_write_bytes(path, payload)
    return digest


def _artifact_file_is_valid(path: Path, expected_digest: str) -> bool:
    """Validate a cached artifact before trusting the records that reference it."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        return False
    try:
        payload = _read_bounded_bytes(path, max_bytes=_MAX_ARTIFACT_BYTES)
        if _sha256_bytes(payload) != expected_digest:
            return False
        raw = json.loads(payload)
        if (
            not isinstance(raw, dict)
            or set(raw) != {"schema_version", "files"}
            or raw.get("schema_version") != _FROZEN_ARTIFACT_SCHEMA
            or not isinstance(raw.get("files"), dict)
        ):
            return False
        files = raw["files"]
        if not all(
            isinstance(rel, str) and (content is None or isinstance(content, str))
            for rel, content in files.items()
        ):
            return False
        return payload == _frozen_artifact_bytes(files)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _scorer_evidence_file_is_valid(
    path: Path,
    expected_digest: str,
    record: RunRecord,
    expected_binding: ScorerEvidenceBinding,
) -> bool:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        return False
    try:
        payload = _read_bounded_bytes(
            path,
            max_bytes=_MAX_SCORER_EVIDENCE_BYTES,
        )
        if _sha256_bytes(payload) != expected_digest:
            return False
        evidence = json.loads(payload)
        if payload != _scorer_evidence_bytes(evidence):
            return False
        outcome, expected, passed = _validate_scorer_evidence(
            evidence,
            expected_binding=expected_binding,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return (
        record.scorer_outcome == outcome.value
        and record.scorer_expected_tests == expected
        and record.scorer_passed_tests == passed
        and record.artifact_correct == (outcome is ScoreOutcome.PASS)
    )


def _score(
    source: Path,
    frozen: dict[str, str | None],
    scratch: Path,
    label: str,
    scorer: ExecutionBackend,
    evidence_dir: Path | None = None,
    evidence_binding: ScorerEvidenceBinding | None = None,
) -> PytestResult:
    """Final truth: canonical repo + frozen diff, graded in a fresh copy.

    Shares no state with the internal gate: fresh directory, canonical tests,
    its own execution backend. The final decision comes from the backend's
    process result, not a JSON file in the candidate-writable repository.
    """
    try:
        # Validate direct callers as well as artifacts produced by _frozen_diff.
        # This prevents a corrupt cache from turning the scorer setup into a
        # path traversal or an oracle rewrite.
        _frozen_artifact_bytes(frozen)
        wd = scratch / f"score_{label}"
        _copy_repo(source, wd, include_tests=True)
        expected, inventory_error = _collect_expected_nodeids(wd, scorer)
        if inventory_error is not None:
            return PytestResult(
                ScoreOutcome.INFRA_ERROR,
                inventory_error.returncode,
                f"scorer inventory failed: {inventory_error.detail}",
            )
        for rel, content in frozen.items():
            target = wd / rel
            if content is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        return _control_plane_pytest(
            wd,
            scorer,
            expected_nodeids=expected,
            evidence_dir=evidence_dir,
            evidence_binding=evidence_binding,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return PytestResult(
            ScoreOutcome.INFRA_ERROR,
            None,
            f"scorer setup failed: {type(error).__name__}",
        )


def _collect_expected_nodeids(
    workdir: Path,
    scorer: ExecutionBackend,
) -> tuple[tuple[str, ...], PytestResult | None]:
    """Collect node IDs from the pristine corpus before candidate bytes are applied."""
    inventory = collect_inventory(workdir, scorer)
    if not inventory.ready:
        return (), PytestResult(
            ScoreOutcome.INFRA_ERROR,
            inventory.driver.returncode,
            inventory.driver.detail,
        )
    return inventory.expected_nodeids, None


def _classify_scorer_receipt(
    *,
    process_returncode: int,
    receipt: dict[str, Any],
    expected_nodeids: tuple[str, ...],
) -> tuple[ScoreOutcome, int]:
    """Cross-check the runner return code, hook receipt, and pristine inventory."""
    outcome, passed = classify_receipt(
        process_returncode=process_returncode,
        receipt=receipt,
        expected_nodeids=expected_nodeids,
    )
    return ScoreOutcome(outcome.value), passed


def _scorer_evidence_bytes(evidence: dict[str, Any]) -> bytes:
    payload = canonical_json_bytes(evidence)
    if len(payload) > _MAX_SCORER_EVIDENCE_BYTES:
        raise ValueError(f"scorer evidence exceeds the {_MAX_SCORER_EVIDENCE_BYTES}-byte limit")
    return payload


def _store_scorer_evidence(evidence: dict[str, Any], evidence_dir: Path) -> str:
    payload = _scorer_evidence_bytes(evidence)
    digest = _sha256_bytes(payload)
    path = evidence_dir / f"{digest}.json"
    if path.exists():
        if (
            _read_bounded_bytes(
                path,
                max_bytes=_MAX_SCORER_EVIDENCE_BYTES,
            )
            != payload
        ):
            raise RuntimeError(f"scorer evidence store is corrupt for {digest}")
        return digest
    _atomic_write_bytes(path, payload)
    return digest


def _validate_scorer_evidence(
    evidence: Any,
    *,
    expected_binding: ScorerEvidenceBinding | None = None,
) -> tuple[ScoreOutcome, int, int]:
    """Validate schema-v2 scorer evidence and recompute its classification."""
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"schema_version", "binding", "pytest_evidence"}
        or evidence.get("schema_version") != _SCORER_EVIDENCE_SCHEMA
        or not isinstance(evidence.get("binding"), dict)
        or not isinstance(evidence.get("pytest_evidence"), dict)
    ):
        raise ValueError("invalid scorer evidence envelope")
    binding_raw = evidence["binding"]
    if (
        set(binding_raw)
        != {
            "task",
            "rep",
            "artifact_sha256",
            "input_snapshot_sha256",
            "scorer_backend",
            "scorer_image_id",
        }
        or not isinstance(binding_raw.get("task"), str)
        or not binding_raw["task"]
        or len(binding_raw["task"]) > 512
        or type(binding_raw.get("rep")) is not int
        or binding_raw["rep"] < 0
        or not isinstance(binding_raw.get("artifact_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", binding_raw["artifact_sha256"]) is None
        or not isinstance(binding_raw.get("input_snapshot_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", binding_raw["input_snapshot_sha256"]) is None
        or not isinstance(binding_raw.get("scorer_backend"), str)
        or not binding_raw["scorer_backend"]
        or len(binding_raw["scorer_backend"]) > 128
    ):
        raise ValueError("invalid scorer evidence binding")
    image_id = binding_raw.get("scorer_image_id")
    if image_id is not None and (
        not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
    ):
        raise ValueError("invalid scorer evidence image ID")
    if binding_raw["scorer_backend"] == "docker" and image_id is None:
        raise ValueError("Docker scorer evidence is missing an immutable image ID")
    binding = ScorerEvidenceBinding(**binding_raw)
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("scorer evidence binding does not match the record")
    pytest_evidence = evidence["pytest_evidence"]
    if pytest_evidence.get("schema_version") != PYTEST_EVIDENCE_SCHEMA:
        raise ValueError("invalid nested Pytest evidence schema")
    outcome, expected, passed = validate_evidence(pytest_evidence)
    return ScoreOutcome(outcome.value), expected, passed


def _control_plane_pytest(
    workdir: Path,
    scorer: ExecutionBackend,
    *,
    expected_nodeids: tuple[str, ...] | None = None,
    evidence_dir: Path | None = None,
    evidence_binding: ScorerEvidenceBinding | None = None,
) -> PytestResult:
    """Require a normal Pytest return plus a nonce-bound post-session receipt.

    The receipt is created only after ``pytest.main`` returns. A candidate import
    that prints a forged summary and calls ``os._exit(0)`` therefore has no
    receipt and fails closed. The random path and nonce are defense in depth,
    not a privilege boundary: trusted-local scoring cannot contain arbitrary
    same-user code; Docker remains required for untrusted repositories.
    """
    if expected_nodeids is None:
        expected_nodeids, inventory_error = _collect_expected_nodeids(workdir, scorer)
        if inventory_error is not None:
            return inventory_error
    result = run_with_evidence(
        workdir,
        scorer,
        expected_nodeids=expected_nodeids,
    )
    if result.receipt is None:
        return PytestResult(
            ScoreOutcome.INFRA_ERROR,
            result.returncode,
            f"scorer: {result.detail}",
        )
    outcome = ScoreOutcome(result.outcome.value)
    pytest_evidence = {
        "schema_version": PYTEST_EVIDENCE_SCHEMA,
        "expected_nodeids": list(expected_nodeids),
        "process_returncode": result.returncode,
        "receipt": result.receipt,
        "receipt_sha256": result.receipt_sha256,
        "classification": outcome.value,
    }
    if evidence_dir is not None and evidence_binding is None:
        return PytestResult(
            ScoreOutcome.INFRA_ERROR,
            result.returncode,
            "scorer: missing evidence binding",
        )
    evidence = {
        "schema_version": _SCORER_EVIDENCE_SCHEMA,
        "binding": asdict(evidence_binding) if evidence_binding is not None else {},
        "pytest_evidence": pytest_evidence,
    }
    evidence_sha256 = (
        _store_scorer_evidence(evidence, evidence_dir)
        if evidence_dir is not None and outcome is not ScoreOutcome.INFRA_ERROR
        else ""
    )
    if outcome is ScoreOutcome.PASS:
        detail = "scorer: tests pass with complete control-plane evidence"
    elif outcome is ScoreOutcome.TEST_FAIL:
        detail = "scorer: tests fail with complete control-plane evidence"
    else:
        detail = "scorer: inconsistent Pytest control-plane evidence"
    return PytestResult(
        outcome,
        result.returncode,
        detail,
        evidence_sha256=evidence_sha256,
        expected_tests=len(expected_nodeids),
        passed_tests=result.passed_tests,
    )


def _bind_latest_success_artifact(
    audits: list[dict[str, Any]] | None,
    *,
    label: str,
    artifact_sha256: str,
) -> None:
    if audits is None:
        return
    for audit in reversed(audits):
        if audit.get("label") == label and audit.get("status") == "succeeded":
            audit["result_artifact_sha256"] = artifact_sha256
            return
    raise RuntimeError(f"missing successful {label!r} call audit")


def _evaluate(
    llm: LLMClient,
    source: Path,
    task: TaskSpec,
    patch: Patch,
    scratch: Path,
    rep: int,
    agent_exec: ExecutionBackend,
    scorer: ExecutionBackend,
    artifact_dir: Path,
    input_snapshot_sha256: str,
    scorer_backend: str,
    scorer_image_id: str | None,
    audit_log: list[dict[str, Any]] | None = None,
) -> list[RunRecord]:
    name = task.inputs.get("_name", task.title)
    out: list[RunRecord] = []

    # ONE working copy for the first attempt: trust and gate score the same
    # artifact (paired). The internal gate predicts; the scorer decides truth.
    wd = scratch / "first"
    _copy_repo(source, wd, include_tests=True)
    if patch.file_contents:
        apply_patch(patch, wd)
    first_gate = _pytest(wd, agent_exec)
    frozen = _frozen_diff(source, wd)
    sha = _store_frozen_artifact(frozen, artifact_dir)
    _bind_latest_success_artifact(
        audit_log,
        label="first",
        artifact_sha256=sha,
    )
    scorer_evidence_dir = artifact_dir.parent / "scorer_evidence"
    first_binding = ScorerEvidenceBinding(
        task=name,
        rep=rep,
        artifact_sha256=sha,
        input_snapshot_sha256=input_snapshot_sha256,
        scorer_backend=scorer_backend,
        scorer_image_id=scorer_image_id,
    )
    first_score = _score(
        source,
        frozen,
        scratch,
        "first",
        scorer,
        scorer_evidence_dir,
        first_binding,
    )
    first_truth_available = first_score.outcome is not ScoreOutcome.INFRA_ERROR
    first_artifact_correct = first_score.outcome is ScoreOutcome.PASS

    out.append(
        RunRecord(
            task=name,
            condition="trust",
            rep=rep,
            status="DONE" if first_truth_available else "ERROR",
            claimed_success=first_truth_available,
            artifact_correct=first_artifact_correct if first_truth_available else False,
            true_success=first_artifact_correct if first_truth_available else False,
            false_success=first_truth_available and not first_artifact_correct,
            repairs=0,
            detail=first_score.detail,
            gate_prediction=None,
            artifact_sha256=sha,
            scorer_outcome=first_score.outcome.value,
            scorer_evidence_sha256=first_score.evidence_sha256,
            scorer_expected_tests=first_score.expected_tests,
            scorer_passed_tests=first_score.passed_tests,
        )
    )
    # The scorer grades the SAME artifact even when the gate refused it, so a
    # refusal of a correct fix is counted (false negative), not invisible.
    gate_error = (
        first_gate.outcome is ScoreOutcome.INFRA_ERROR
        or first_score.outcome is ScoreOutcome.INFRA_ERROR
    )
    gate_pred = first_gate.outcome is ScoreOutcome.PASS
    out.append(
        RunRecord(
            task=name,
            condition="gate",
            rep=rep,
            status="ERROR" if gate_error else ("DONE" if gate_pred else "FAILED"),
            claimed_success=False if gate_error else gate_pred,
            artifact_correct=False if gate_error else first_artifact_correct,
            true_success=False if gate_error else gate_pred and first_artifact_correct,
            false_success=False if gate_error else gate_pred and not first_artifact_correct,
            repairs=0,
            detail=(
                f"gate: {first_gate.detail}; {first_score.detail}"
                if gate_error
                else first_score.detail
            ),
            gate_prediction=(None if first_gate.outcome is ScoreOutcome.INFRA_ERROR else gate_pred),
            artifact_sha256=sha,
            scorer_outcome=first_score.outcome.value,
            scorer_evidence_sha256=first_score.evidence_sha256,
            scorer_expected_tests=first_score.expected_tests,
            scorer_passed_tests=first_score.passed_tests,
        )
    )

    # verify: same first attempt, then repair with failing-test feedback.
    wd2 = scratch / "verify"
    _copy_repo(source, wd2, include_tests=True)
    if patch.file_contents:
        apply_patch(patch, wd2)
    verify_gate = _pytest(wd2, agent_exec)
    failures = list(verify_gate.messages)
    repairs = 0
    while verify_gate.outcome is ScoreOutcome.TEST_FAIL and repairs < _MAX_REPAIRS:
        repairs += 1
        step = _fix_step(task).as_repair(failures)
        repair = _retry(
            lambda s=step: _sanitize(Implementer(llm).implement(s, _empty_bundle(), wd2)),
            "repair",
            llm=llm,
            audit_log=audit_log,
        )
        if not repair.file_contents:
            _bind_latest_success_artifact(
                audit_log,
                label="repair",
                artifact_sha256=_artifact_digest(_frozen_diff(source, wd2)),
            )
            break  # nothing new to try
        apply_patch(repair, wd2)
        _bind_latest_success_artifact(
            audit_log,
            label="repair",
            artifact_sha256=_artifact_digest(_frozen_diff(source, wd2)),
        )
        verify_gate = _pytest(wd2, agent_exec)
        failures = list(verify_gate.messages)
    frozen2 = _frozen_diff(source, wd2)
    sha2 = _store_frozen_artifact(frozen2, artifact_dir)
    verify_binding = ScorerEvidenceBinding(
        task=name,
        rep=rep,
        artifact_sha256=sha2,
        input_snapshot_sha256=input_snapshot_sha256,
        scorer_backend=scorer_backend,
        scorer_image_id=scorer_image_id,
    )
    score2 = _score(
        source,
        frozen2,
        scratch,
        "verify",
        scorer,
        scorer_evidence_dir,
        verify_binding,
    )
    verify_error = (
        verify_gate.outcome is ScoreOutcome.INFRA_ERROR
        or score2.outcome is ScoreOutcome.INFRA_ERROR
    )
    ok = verify_gate.outcome is ScoreOutcome.PASS
    artifact_correct2 = score2.outcome is ScoreOutcome.PASS
    out.append(
        RunRecord(
            task=name,
            condition="verify",
            rep=rep,
            status="ERROR" if verify_error else ("DONE" if ok else "FAILED"),
            claimed_success=False if verify_error else ok,
            artifact_correct=False if verify_error else artifact_correct2,
            true_success=False if verify_error else ok and artifact_correct2,
            false_success=False if verify_error else ok and not artifact_correct2,
            repairs=repairs,
            detail=(
                f"gate: {verify_gate.detail}; {score2.detail}" if verify_error else score2.detail
            ),
            gate_prediction=(None if verify_gate.outcome is ScoreOutcome.INFRA_ERROR else ok),
            artifact_sha256=sha2,
            scorer_outcome=score2.outcome.value,
            scorer_evidence_sha256=score2.evidence_sha256,
            scorer_expected_tests=score2.expected_tests,
            scorer_passed_tests=score2.passed_tests,
        )
    )
    return out


def _make_llm(
    llm: str, model: str | None, *, cli_path: str = "claude", effort: str = "medium"
) -> LLMClient:
    if llm == "stub":
        from .llm.stub import DeterministicStub

        return DeterministicStub()
    if llm == "claude_cli":
        from .llm.claude_cli import ClaudeCLIClient

        # no_tools => single-shot completion; the implementer cannot read the oracle.
        return ClaudeCLIClient(cli_path=cli_path, model=model, no_tools=True)
    if llm == "codex_cli":
        from .llm.codex_cli import CodexCLIClient

        # no_tools rejects every result produced with a tool. The generated
        # permission profile also prevents tool subprocesses from reading the
        # credential home or files outside the empty attempt workspace.
        return CodexCLIClient(
            cli_path=cli_path, model=model, reasoning_effort=effort, no_tools=True
        )
    if llm == "anthropic":
        from .llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=model or "claude-opus-4-8")
    raise ValueError(f"unknown llm backend: {llm!r}")


def _atomic_write(path: Path, text: str, *, anchor: Path | None = None) -> None:
    atomic_replace_text(path, text, anchor=anchor)


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    anchor: Path | None = None,
) -> None:
    atomic_replace_bytes(path, payload, anchor=anchor)


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _regular_file_open_flags() -> int:
    return os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _safe_named_regular_file(
    descriptor: int,
    *,
    directory_descriptor: int,
    name: str,
) -> os.stat_result:
    opened = os.fstat(descriptor)
    named = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or not _same_inode(opened, named)
        or opened.st_uid != os.geteuid()
        or named.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) & 0o022
    ):
        raise OSError(f"unsafe ablation control file: {name}")
    return opened


@dataclass(frozen=True)
class _FormalOutputLease:
    path: Path
    directory_descriptor: int
    device: int
    inode: int


def _open_or_create_formal_output(
    path: Path,
    *,
    require_existing: bool = False,
) -> tuple[Path, int]:
    """Walk every lexical component with ``openat`` and reject directory links."""
    output = Path(os.path.abspath(os.fspath(path)))
    if not output.is_absolute() or not output.anchor:
        raise OSError("formal ablation output path is not absolute")
    descriptor = os.open(output.anchor, _directory_open_flags())
    try:
        for part in output.parts[1:]:
            if not part or part in {".", ".."} or "/" in part or os.sep in part:
                raise OSError("formal ablation output path component is unsafe")
            created = False
            if not require_existing:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
            child = os.open(part, _directory_open_flags(), dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                named = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or not stat.S_ISDIR(named.st_mode)
                    or stat.S_ISLNK(named.st_mode)
                    or not _same_inode(opened, named)
                ):
                    raise OSError("formal ablation output path component is unsafe")
                if created:
                    os.fsync(child)
                    os.fsync(descriptor)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return output, descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _formal_ablation_lock(
    out_dir: Path,
    *,
    require_existing: bool = False,
) -> Iterator[_FormalOutputLease]:
    """Hold a fail-closed lock from preflight through report and cleanup completion."""
    try:
        # Walking from the filesystem root intentionally rejects every symbolic
        # component, including macOS aliases such as /var. Formal callers should
        # provide the canonical spelling (/private/var) rather than weakening the
        # evidence boundary for a convenience alias.
        output, directory_descriptor = _open_or_create_formal_output(
            out_dir,
            require_existing=require_existing,
        )
    except OSError as error:
        raise RuntimeError("formal ablation output directory is unsafe") from error
    lock_descriptor: int | None = None
    locked = False
    try:
        opened_directory = os.fstat(directory_descriptor)
        named_directory = output.lstat()
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or not stat.S_ISDIR(named_directory.st_mode)
            or stat.S_ISLNK(named_directory.st_mode)
            or not _same_inode(opened_directory, named_directory)
            or opened_directory.st_uid != os.geteuid()
            or stat.S_IMODE(opened_directory.st_mode) & 0o022
        ):
            raise RuntimeError("formal ablation output directory is unsafe")

        flags = _regular_file_open_flags()
        created = False
        try:
            if require_existing:
                lock_descriptor = os.open(
                    _FORMAL_OUTPUT_LOCK_NAME,
                    flags,
                    dir_fd=directory_descriptor,
                )
            else:
                lock_descriptor = os.open(
                    _FORMAL_OUTPUT_LOCK_NAME,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_descriptor,
                )
                created = True
        except FileExistsError:
            try:
                lock_descriptor = os.open(
                    _FORMAL_OUTPUT_LOCK_NAME,
                    flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise RuntimeError("formal ablation output lock is unsafe") from error
        except OSError as error:
            raise RuntimeError("formal ablation output lock could not be created") from error

        try:
            _safe_named_regular_file(
                lock_descriptor,
                directory_descriptor=directory_descriptor,
                name=_FORMAL_OUTPUT_LOCK_NAME,
            )
        except OSError as error:
            raise RuntimeError("formal ablation output lock is unsafe") from error
        if created:
            os.fchmod(lock_descriptor, 0o600)
            os.fsync(lock_descriptor)
            os.fsync(directory_descriptor)

        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except (BlockingIOError, OSError) as error:
            raise RuntimeError("formal ablation output is already active") from error

        try:
            _safe_named_regular_file(
                lock_descriptor,
                directory_descriptor=directory_descriptor,
                name=_FORMAL_OUTPUT_LOCK_NAME,
            )
        except OSError as error:
            raise RuntimeError("formal ablation output lock changed during acquisition") from error
        yield _FormalOutputLease(
            path=output,
            directory_descriptor=directory_descriptor,
            device=opened_directory.st_dev,
            inode=opened_directory.st_ino,
        )
        final_directory = os.fstat(directory_descriptor)
        final_named_directory = output.lstat()
        if (
            not _same_inode(final_directory, opened_directory)
            or not _same_inode(final_named_directory, opened_directory)
        ):
            raise RuntimeError("formal ablation output directory changed during the run")
        try:
            _safe_named_regular_file(
                lock_descriptor,
                directory_descriptor=directory_descriptor,
                name=_FORMAL_OUTPUT_LOCK_NAME,
            )
        except OSError as error:
            raise RuntimeError("formal ablation output lock changed during the run") from error
    finally:
        if lock_descriptor is not None:
            if locked:
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(lock_descriptor)
        os.close(directory_descriptor)


def _open_lease_subdirectory(
    lease: _FormalOutputLease,
    parts: tuple[str, ...],
) -> int:
    """Open a directory below a held lease without resolving a path component."""
    descriptor = os.dup(lease.directory_descriptor)
    try:
        root = os.fstat(descriptor)
        if (root.st_dev, root.st_ino) != (lease.device, lease.inode):
            raise OSError("formal ablation output lease changed")
        for part in parts:
            if not part or part in {".", ".."} or "/" in part or os.sep in part:
                raise OSError("formal ablation evidence path is unsafe")
            created = False
            try:
                os.mkdir(part, 0o700, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            child = os.open(part, _directory_open_flags(), dir_fd=descriptor)
            try:
                child_metadata = os.fstat(child)
                named_metadata = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(child_metadata.st_mode)
                    or not stat.S_ISDIR(named_metadata.st_mode)
                    or stat.S_ISLNK(named_metadata.st_mode)
                    or not _same_inode(child_metadata, named_metadata)
                    or child_metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(child_metadata.st_mode) & 0o022
                ):
                    raise OSError("formal ablation evidence directory is unsafe")
                if created:
                    os.fsync(child)
                    os.fsync(descriptor)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_formal_cell_start(
    path: Path,
    payload: bytes,
    *,
    out_dir: Path,
    lease: _FormalOutputLease | None = None,
    label: str = "cell-start marker",
) -> None:
    """Create a cell-start record once and persist it before a model can run."""
    output = Path(os.path.abspath(os.fspath(out_dir)))
    target = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = target.relative_to(output)
    except ValueError as error:
        raise RuntimeError(f"formal ablation {label} escapes its output") from error
    if len(relative.parts) < 2:
        raise RuntimeError(f"formal ablation {label} has no evidence directory")

    if lease is not None:
        if output != lease.path:
            raise RuntimeError(f"formal ablation {label} uses a different output lease")
        try:
            parent_descriptor = _open_lease_subdirectory(lease, relative.parts[:-1])
        except OSError as error:
            raise RuntimeError(f"formal ablation {label} directory is unsafe") from error
    else:
        try:
            durable_mkdir_chain(output)
            parent = durable_mkdir_chain(target.parent, anchor=output)
            parent_descriptor = os.open(parent, _directory_open_flags())
        except (OSError, ValueError) as error:
            raise RuntimeError(f"formal ablation {label} directory is unsafe") from error

    descriptor: int | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                relative.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as error:
            raise RuntimeError(
                f"formal ablation {label} already exists; mark the registered "
                "attempt ABANDONED"
            ) from error
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("cell-start marker write made no progress")
                offset += written
            os.fsync(descriptor)
            _safe_named_regular_file(
                descriptor,
                directory_descriptor=parent_descriptor,
                name=relative.name,
            )
            os.fsync(parent_descriptor)
            _safe_named_regular_file(
                descriptor,
                directory_descriptor=parent_descriptor,
                name=relative.name,
            )
        except OSError as error:
            raise RuntimeError(f"formal ablation {label} was not durable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


# --- provenance fingerprint ---------------------------------------------------
def _repo_digest(source: Path) -> str:
    h = hashlib.sha256()
    for rel, p in sorted(_iter_files(source)):
        h.update(rel.encode())
        h.update(b"\0")
        _consume_stable_regular_file(
            p,
            h.update,
            reject_hardlinks=False,
        )
        h.update(b"\0")
    return h.hexdigest()


def _input_snapshot_digest(task_sha256: str, corpus_sha256: str) -> str:
    return _canonical_digest(
        {
            "schema_version": _INPUT_SNAPSHOT_SCHEMA,
            "task_sha256": task_sha256,
            "corpus_sha256": corpus_sha256,
        }
    )


def _freeze_ablation_input(
    *,
    out_dir: Path,
    name: str,
    task_path: Path,
    source: Path,
) -> tuple[Path, Path, str, str, str]:
    """Copy one task and corpus into an immutable, content-addressed snapshot.

    Digests are measured before and after the copy. A source that changes while
    the snapshot is being created is rejected instead of being cached under an
    earlier fingerprint. All later cells read the snapshot, so edits to the
    live corpus during a long run cannot affect results.
    """
    task_path = task_path.resolve()
    source = source.resolve()
    task_before = _read_bounded_bytes(task_path, max_bytes=_MAX_TASK_BYTES)
    corpus_before = _repo_digest(source)
    snapshots = out_dir / "input_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=snapshots))
    try:
        task_copy = temporary / "task.yaml"
        task_copy.write_bytes(task_before)
        repo_copy = temporary / "repo"
        _copy_repo(source, repo_copy, include_tests=True)
        task_after = _read_bounded_bytes(task_path, max_bytes=_MAX_TASK_BYTES)
        corpus_after = _repo_digest(source)
        task_digest = _sha256_bytes(task_before)
        corpus_digest = _repo_digest(repo_copy)
        if (
            task_before != task_after
            or corpus_before != corpus_after
            or corpus_digest != corpus_before
        ):
            raise RuntimeError(f"ablation input changed while snapshotting task {name!r}")
        snapshot_digest = _input_snapshot_digest(task_digest, corpus_digest)
        metadata = {
            "schema_version": _INPUT_SNAPSHOT_SCHEMA,
            "task": name,
            "task_sha256": task_digest,
            "corpus_sha256": corpus_digest,
            "snapshot_sha256": snapshot_digest,
        }
        (temporary / "snapshot.json").write_bytes(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        )
        destination = snapshots / snapshot_digest
        if destination.exists():
            if (
                destination.is_symlink()
                or not destination.is_dir()
                or _read_bounded_bytes(
                    destination / "task.yaml",
                    max_bytes=_MAX_TASK_BYTES,
                )
                != task_before
                or _repo_digest(destination / "repo") != corpus_digest
                or json.loads(
                    _read_bounded_text(
                        destination / "snapshot.json",
                        max_bytes=_MAX_TASK_BYTES,
                    )
                )
                != metadata
            ):
                raise RuntimeError(f"ablation input snapshot is corrupt for {snapshot_digest}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        for path in destination.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        return (
            destination / "task.yaml",
            destination / "repo",
            task_digest,
            corpus_digest,
            snapshot_digest,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(payload.encode())


def _canonical_json_object_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= 0
    )


def _validate_codex_event_summary(
    value: Any,
    *,
    successful: bool,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "total_events",
        "events",
        "items",
        "invalid_json_lines",
    }:
        raise ValueError("Codex event summary has an invalid shape")
    total = value["total_events"]
    invalid = value["invalid_json_lines"]
    events = value["events"]
    items = value["items"]
    item_events = ("item.started", "item.updated", "item.completed")
    if (
        not _nonnegative_int(total)
        or not _nonnegative_int(invalid)
        or not isinstance(events, dict)
        or not isinstance(items, dict)
        or any(
            not isinstance(name, str)
            or name not in _CODEX_EVENT_TYPES
            or type(count) is not int
            or count <= 0
            for name, count in events.items()
        )
        or any(
            not isinstance(name, str) or not name or type(count) is not int or count <= 0
            for name, count in items.items()
        )
        or total != sum(events.values())
        or sum(items.values()) != sum(events.get(name, 0) for name in item_events)
    ):
        raise ValueError("Codex event summary counts are inconsistent")
    if not successful:
        return
    if (
        invalid != 0
        or events.get("thread.started") != 1
        or events.get("turn.started") != 1
        or events.get("turn.completed") != 1
        or events.get("turn.failed", 0) != 0
        or events.get("error", 0) != 0
        or events.get("item.completed", 0) < 1
        or items.get("agent_message", 0) < 1
        or not set(items).issubset(_CODEX_TEXT_ITEMS)
    ):
        raise ValueError("successful Codex call is not a strict no-tools turn")


def _validate_llm_call_receipt(
    value: Any,
    *,
    expected_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one canonical receipt without trusting report-side call fields."""
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "binding", "call"}
        or value.get("schema_version") != _LLM_CALL_RECEIPT_SCHEMA
        or not isinstance(value.get("binding"), dict)
        or not isinstance(value.get("call"), dict)
    ):
        raise ValueError("invalid LLM call receipt envelope")
    binding = value["binding"]
    if set(binding) != {
        "task",
        "rep",
        "label",
        "ordinal",
        "cell_fingerprint",
        "input_snapshot_sha256",
        "formal_attempt_id",
        "formal_registration_registry_sha256",
        "formal_protocol_sha256",
        "formal_outcome_key",
        "prompt_sha256",
        "response_sha256",
        "patch_sha256",
        "result_artifact_sha256",
    }:
        raise ValueError("invalid LLM call receipt binding")
    if (
        not isinstance(binding["task"], str)
        or not binding["task"]
        or not _nonnegative_int(binding["rep"])
        or binding["label"] not in {"first", "repair"}
        or not _nonnegative_int(binding["ordinal"])
        or not all(
            isinstance(binding[name], str) and _HEX_64.fullmatch(binding[name])
            for name in (
                "cell_fingerprint",
                "input_snapshot_sha256",
                "prompt_sha256",
            )
        )
    ):
        raise ValueError("invalid LLM call receipt cell binding")
    formal_binding_fields = (
        "formal_attempt_id",
        "formal_registration_registry_sha256",
        "formal_protocol_sha256",
        "formal_outcome_key",
    )
    formal_values = tuple(binding[name] for name in formal_binding_fields)
    if not (
        all(value is None for value in formal_values)
        or all(
            isinstance(value, str) and _HEX_64.fullmatch(value)
            for value in formal_values
        )
    ):
        raise ValueError("invalid LLM call receipt formal-run binding")
    if expected_binding is not None and any(
        binding.get(name) != expected for name, expected in expected_binding.items()
    ):
        raise ValueError("LLM call receipt is bound to another cell")

    call = value["call"]
    required_call_fields = {
        "status",
        "backend",
        "cli_version",
        "model",
        "reasoning_effort",
        "sandbox_mode",
        "permission_model",
        "permission_profile",
        "credential_barrier",
        "cli_executable_sha256",
        "cli_executable_trusted",
        "externally_sandboxed",
        "retries",
        "attempt_count",
        "duration_s",
        "event_summary",
        "attempts",
        "usage",
    }
    optional_call_fields = {"error_type", "retryable"}
    if (
        not required_call_fields.issubset(call)
        or not set(call).issubset(required_call_fields | optional_call_fields)
        or call["status"] not in {"succeeded", "failed"}
        or call["backend"] != "codex_cli"
        or not isinstance(call["cli_version"], str)
        or not call["cli_version"]
        or not isinstance(call["model"], str)
        or not call["model"]
        or not isinstance(call["reasoning_effort"], str)
        or not call["reasoning_effort"]
        or call["sandbox_mode"] != "read-only"
        or call["permission_model"] != "profile"
        or call["permission_profile"] != "lha-read"
        or call["credential_barrier"] != "verified"
        or not isinstance(call["cli_executable_sha256"], str)
        or _HEX_64.fullmatch(call["cli_executable_sha256"]) is None
        or type(call["cli_executable_trusted"]) is not bool
        or call["externally_sandboxed"] is not False
        or not _nonnegative_int(call["retries"])
        or not _nonnegative_int(call["attempt_count"])
        or not _finite_nonnegative(call["duration_s"])
        or not isinstance(call["attempts"], list)
        or not isinstance(call["usage"], dict)
    ):
        raise ValueError("invalid LLM call receipt protocol")
    successful = call["status"] == "succeeded"
    response_sha256 = binding["response_sha256"]
    patch_sha256 = binding["patch_sha256"]
    artifact_sha256 = binding["result_artifact_sha256"]
    if successful:
        if not all(
            isinstance(digest, str) and _HEX_64.fullmatch(digest)
            for digest in (response_sha256, patch_sha256, artifact_sha256)
        ):
            raise ValueError("successful LLM call receipt lacks output bindings")
        if "error_type" in call or "retryable" in call:
            raise ValueError("successful LLM call receipt carries failure metadata")
    else:
        if (
            response_sha256 is not None
            or patch_sha256 is not None
            or artifact_sha256 is not None
            or not isinstance(call.get("error_type"), str)
            or not call["error_type"]
            or type(call.get("retryable")) is not bool
        ):
            raise ValueError("failed LLM call receipt has invalid outcome bindings")

    attempts = call["attempts"]
    if call["attempt_count"] != len(attempts) or call["retries"] != max(0, len(attempts) - 1):
        raise ValueError("LLM call receipt retry counters are inconsistent")
    for index, attempt in enumerate(attempts, start=1):
        if (
            not isinstance(attempt, dict)
            or attempt.get("attempt") != index
            or attempt.get("status") not in {"succeeded", "failed"}
            or not _finite_nonnegative(attempt.get("duration_s"))
            or not isinstance(attempt.get("event_summary"), dict)
        ):
            raise ValueError("LLM call receipt has an invalid inner attempt")
        attempt_succeeded = attempt["status"] == "succeeded"
        if attempt_succeeded:
            if index != len(attempts) or "error_type" in attempt:
                raise ValueError("LLM call receipt succeeded before its final attempt")
        elif not isinstance(attempt.get("error_type"), str) or not attempt["error_type"]:
            raise ValueError("failed LLM inner attempt lacks an error type")
        _validate_codex_event_summary(
            attempt["event_summary"],
            successful=attempt_succeeded,
        )
    if successful:
        if not attempts or attempts[-1]["status"] != "succeeded":
            raise ValueError("successful LLM call receipt has no successful final attempt")
    elif any(attempt["status"] == "succeeded" for attempt in attempts):
        raise ValueError("failed LLM call receipt contains a successful inner attempt")
    _validate_codex_event_summary(call["event_summary"], successful=successful)
    if attempts and call["event_summary"] != attempts[-1]["event_summary"]:
        raise ValueError("LLM call receipt event summary is not the final attempt")

    usage = call["usage"]
    if (
        set(usage)
        != {
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "cost_usd",
            "model",
        }
        or usage["model"] != call["model"]
    ):
        raise ValueError("LLM call receipt usage does not match the model")
    for name in ("input_tokens", "cached_input_tokens", "output_tokens"):
        token_count = usage[name]
        if successful:
            if not _nonnegative_int(token_count):
                raise ValueError("successful LLM call receipt lacks token usage")
        elif token_count is not None and not _nonnegative_int(token_count):
            raise ValueError("failed LLM call receipt has invalid token usage")
    if usage["cost_usd"] is not None and not _finite_nonnegative(usage["cost_usd"]):
        raise ValueError("LLM call receipt has invalid cost")
    return value


def _llm_call_receipt_bytes(value: dict[str, Any]) -> bytes:
    _validate_llm_call_receipt(value)
    payload = _canonical_json_object_bytes(value)
    if len(payload) > _MAX_LLM_CALL_RECEIPT_BYTES:
        raise ValueError("LLM call receipt exceeds the byte limit")
    return payload


def _store_llm_call_receipt(value: dict[str, Any], directory: Path) -> str:
    payload = _llm_call_receipt_bytes(value)
    digest = _sha256_bytes(payload)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("LLM call receipt store must be a real directory")
    path = directory / f"{digest}.json"
    if path.exists():
        if (
            _read_bounded_bytes(
                path,
                max_bytes=_MAX_LLM_CALL_RECEIPT_BYTES,
            )
            != payload
        ):
            raise RuntimeError(f"LLM call receipt store is corrupt for {digest}")
        return digest
    _atomic_write_bytes(path, payload)
    return digest


def _read_llm_call_receipt(
    path: Path,
    expected_digest: str,
    *,
    expected_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _HEX_64.fullmatch(expected_digest) is None:
        raise ValueError("invalid LLM call receipt digest")
    payload = _read_bounded_bytes(
        path,
        max_bytes=_MAX_LLM_CALL_RECEIPT_BYTES,
    )
    if _sha256_bytes(payload) != expected_digest:
        raise ValueError("LLM call receipt digest does not match its bytes")
    value = json.loads(payload)
    receipt = _validate_llm_call_receipt(
        value,
        expected_binding=expected_binding,
    )
    if payload != _canonical_json_object_bytes(receipt):
        raise ValueError("LLM call receipt is not canonical JSON")
    return receipt


def _report_fingerprint(raw: dict[str, Any]) -> str:
    """Hash every public report field except the digest itself."""
    payload = dict(raw)
    payload.pop("fingerprint", None)
    return _sha256_bytes(_canonical_json_object_bytes(payload))


def _repo_relative_evidence_path(
    repository_root: Path,
    value: Any,
    *,
    kind: str,
) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"formal corpus {kind} path is missing")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"formal corpus {kind} path is unsafe")
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("formal corpus repository root is unavailable") from error
    path = root
    try:
        for index, component in enumerate(relative.parts):
            path = path / component
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    f"formal corpus {kind} path must not contain a symlink"
                )
            if index < len(relative.parts) - 1 and not stat.S_ISDIR(
                metadata.st_mode
            ):
                raise ValueError(
                    f"formal corpus {kind} parent is not a directory"
                )
        path.resolve(strict=True).relative_to(root)
    except OSError as error:
        raise ValueError(f"formal corpus {kind} path is unavailable") from error
    return relative.as_posix(), path


def _load_formal_corpus_manifest(
    path: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], str]:
    payload = _read_bounded_bytes(
        path,
        max_bytes=_MAX_FORMAL_MANIFEST_BYTES,
    )
    digest = _sha256_bytes(payload)
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal corpus manifest is not valid JSON") from error
    if (
        not isinstance(raw, dict)
        or set(raw)
        != {
            "schema_version",
            "benchmark",
            "repetitions",
            "corpus_commit",
            "tasks",
        }
        or raw.get("schema_version") != _FORMAL_CORPUS_MANIFEST_SCHEMA
        or raw.get("benchmark") != "lha-verification-ablation"
        or raw.get("repetitions") != _FORMAL_REPETITIONS
        or not isinstance(raw.get("corpus_commit"), str)
        or _HEX_40.fullmatch(raw["corpus_commit"]) is None
        or not isinstance(raw.get("tasks"), list)
        or len(raw["tasks"]) != _FORMAL_TASK_COUNT
    ):
        raise ValueError("formal corpus manifest has an invalid envelope")
    names: list[str] = []
    task_paths: list[str] = []
    for entry in raw["tasks"]:
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "name",
                "task_path",
                "task_sha256",
                "corpus_path",
                "corpus_sha256",
            }
            or not isinstance(entry.get("name"), str)
            or not entry["name"]
            or not isinstance(entry.get("task_sha256"), str)
            or _HEX_64.fullmatch(entry["task_sha256"]) is None
            or not isinstance(entry.get("corpus_sha256"), str)
            or _HEX_64.fullmatch(entry["corpus_sha256"]) is None
        ):
            raise ValueError("formal corpus manifest has an invalid task entry")
        task_relative, task_path = _repo_relative_evidence_path(
            repository_root,
            entry["task_path"],
            kind="task",
        )
        corpus_relative, corpus_path = _repo_relative_evidence_path(
            repository_root,
            entry["corpus_path"],
            kind="corpus",
        )
        if (
            not task_path.is_file()
            or not corpus_path.is_dir()
            or Path(task_relative).stem != entry["name"]
            or _sha256_bytes(_read_bounded_bytes(task_path, max_bytes=_MAX_TASK_BYTES))
            != entry["task_sha256"]
            or _repo_digest(corpus_path) != entry["corpus_sha256"]
        ):
            raise ValueError(f"formal corpus bytes disagree for {entry['name']!r}")
        spec = TaskSpec.from_file(task_path)
        target = Path(spec.target_repo or ".")
        target = target if target.is_absolute() else repository_root / target
        try:
            target_relative = target.resolve().relative_to(repository_root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("formal task target_repo escapes the repository") from error
        if target_relative != corpus_relative:
            raise ValueError(f"formal task target_repo disagrees for {entry['name']!r}")
        names.append(entry["name"])
        task_paths.append(task_relative)
    if (
        len(names) != len(set(names))
        or len(task_paths) != len(set(task_paths))
        or names != sorted(names)
    ):
        raise ValueError("formal corpus task order is not unique and stable")
    return raw, digest


@dataclass(frozen=True)
class _FormalCorpusBinding:
    path: str
    sha256: str
    preregistration_commit: str
    git_executable: dict[str, Any]


@dataclass(frozen=True)
class _FormalAttemptBinding:
    attempt_id: str
    registry_path: str
    registry_sha256: str
    protocol_sha256: str
    registration_commit: str
    repository_root: Path
    git_path: str
    witness_remote_name: str
    witness_remote_url: str
    witness_credential_helper: FormalGitCredentialHelper
    witness_ref: str


@dataclass(frozen=True)
class _FormalRunBinding:
    attempt_id: str
    registration_registry_sha256: str
    protocol_sha256: str
    outcome_key: str
    header_path: str
    header_sha256: str
    witness_remote_name: str
    witness_remote_url: str
    witness_ref: str
    witness_commit: str

    def cell_fields(self) -> dict[str, str]:
        return {
            "formal_attempt_id": self.attempt_id,
            "formal_registration_registry_sha256": (
                self.registration_registry_sha256
            ),
            "formal_protocol_sha256": self.protocol_sha256,
            "formal_outcome_key": self.outcome_key,
        }


def _initialize_formal_run(
    attempt: _FormalAttemptBinding,
    lease: _FormalOutputLease,
) -> _FormalRunBinding:
    """Seal a fresh-attempt header and then consume its remote witness.

    A formal output directory is single use.  Even a header left by a failed
    witness push requires an ABANDONED registry event rather than deletion and
    retry. No model call can occur until both durable header and witness exist.
    """
    try:
        entries = set(os.listdir(lease.directory_descriptor))
    except OSError as error:
        raise RuntimeError("formal ablation output cannot be enumerated safely") from error
    if entries != {_FORMAL_OUTPUT_LOCK_NAME}:
        raise RuntimeError(
            "formal ablation output is not fresh; mark the registered attempt "
            "ABANDONED and use a new attempt"
        )

    outcome_key = secrets.token_hex(32)
    payload = _canonical_json_object_bytes(
        {
            "schema_version": _FORMAL_RUN_HEADER_SCHEMA,
            "formal_attempt_id": attempt.attempt_id,
            "registration_registry_sha256": attempt.registry_sha256,
            "protocol_sha256": attempt.protocol_sha256,
            "outcome_key": outcome_key,
        }
    )
    header_sha256 = hashlib.sha256(payload).hexdigest()
    descriptor: int | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(
            _FORMAL_RUN_HEADER_NAME,
            flags,
            0o600,
            dir_fd=lease.directory_descriptor,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("formal run header write made no progress")
            offset += written
        os.fsync(descriptor)
        _safe_named_regular_file(
            descriptor,
            directory_descriptor=lease.directory_descriptor,
            name=_FORMAL_RUN_HEADER_NAME,
        )
        os.fsync(lease.directory_descriptor)
        _safe_named_regular_file(
            descriptor,
            directory_descriptor=lease.directory_descriptor,
            name=_FORMAL_RUN_HEADER_NAME,
        )
    except OSError as error:
        raise RuntimeError(
            "formal ablation run header could not be sealed; mark the registered "
            "attempt ABANDONED"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)

    witness_commit = _create_formal_start_witness(
        attempt,
        outcome_key=outcome_key,
        run_header_sha256=header_sha256,
    )
    return _FormalRunBinding(
        attempt_id=attempt.attempt_id,
        registration_registry_sha256=attempt.registry_sha256,
        protocol_sha256=attempt.protocol_sha256,
        outcome_key=outcome_key,
        header_path=_FORMAL_RUN_HEADER_NAME,
        header_sha256=header_sha256,
        witness_remote_name=attempt.witness_remote_name,
        witness_remote_url=attempt.witness_remote_url,
        witness_ref=attempt.witness_ref,
        witness_commit=witness_commit,
    )


def _formal_git_output(
    git_path: str,
    arguments: list[str],
    *,
    repository_root: Path,
    label: str,
    input: str | None = None,
) -> str:
    result = run(
        [git_path, *arguments],
        cwd=repository_root,
        timeout=30,
        env=_git_control_env(),
        input=input,
    )
    if (
        result.returncode != 0
        or result.output_truncated
        or result.cleanup_unconfirmed
    ):
        raise RuntimeError(f"formal ablation attempt registry failed {label}")
    return result.stdout


def _formal_local_git_config_values(
    git_path: str,
    *,
    repository_root: Path,
    key: str,
) -> list[str]:
    """Read one exact repository-local key without includes or URL rewriting."""
    if (
        not Path(git_path).is_absolute()
        or not key
        or key.strip() != key
        or any(character.isspace() or ord(character) < 32 for character in key)
    ):
        raise RuntimeError("formal Git local configuration request is invalid")
    result = run(
        [
            git_path,
            "config",
            "--local",
            "--no-includes",
            "--null",
            "--get-all",
            key,
        ],
        cwd=repository_root,
        timeout=30,
        env=_git_local_config_env(),
    )
    if (
        result.returncode not in {0, 1}
        or result.output_truncated
        or result.cleanup_unconfirmed
        or (result.returncode == 1 and result.stdout)
    ):
        raise RuntimeError("formal Git local configuration could not be read")
    if result.returncode == 1:
        return []
    if not result.stdout.endswith("\0"):
        raise RuntimeError("formal Git local configuration output is truncated")
    values = result.stdout[:-1].split("\0")
    if not values or any(not value for value in values):
        raise RuntimeError("formal Git local configuration contains an empty value")
    return values


def _formal_witness_remote_url(
    git_path: str,
    *,
    repository_root: Path,
    remote_name: str,
) -> str:
    """Resolve only pushurl/url from local config and require one public URL."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", remote_name) is None:
        raise RuntimeError("formal witness remote name is invalid")
    prefix = f"remote.{remote_name}"
    push_urls = _formal_local_git_config_values(
        git_path,
        repository_root=repository_root,
        key=f"{prefix}.pushurl",
    )
    urls = push_urls or _formal_local_git_config_values(
        git_path,
        repository_root=repository_root,
        key=f"{prefix}.url",
    )
    if len(urls) != 1:
        raise RuntimeError(
            "formal witness remote must have exactly one configured URL"
        )
    try:
        return validate_formal_witness_remote_url(urls[0])
    except ValueError as error:
        raise RuntimeError(
            "formal witness remote must be a public HTTPS URL"
        ) from error


def _formal_git_credential_helper(
    host: str,
    *,
    expected: FormalGitCredentialHelper | None = None,
) -> FormalGitCredentialHelper:
    """Resolve and remeasure the exact gh binary used by Git authentication."""
    if expected is None:
        resolved_text = trusted_executable("gh", require_unwritable=False)
        if resolved_text is None:
            raise RuntimeError("formal Git credential helper is unavailable")
        path = Path(resolved_text)
    else:
        path = Path(expected.executable_path)
    try:
        resolved = path.resolve(strict=True)
        before = resolved.lstat()
    except (OSError, RuntimeError) as error:
        raise RuntimeError("formal Git credential helper is unavailable") from error
    if (
        not resolved.is_absolute()
        or resolved != path
        or resolved.name != "gh"
        or not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > _MAX_CONTROL_EXECUTABLE_BYTES
        or not os.access(resolved, os.X_OK)
    ):
        raise RuntimeError("formal Git credential helper is not a bounded executable")
    digest = hashlib.sha256()
    _consume_stable_regular_file(
        resolved,
        digest.update,
        max_bytes=_MAX_CONTROL_EXECUTABLE_BYTES,
        reject_hardlinks=False,
    )
    after = resolved.lstat()
    if _stable_file_signature(before) != _stable_file_signature(after):
        raise RuntimeError("formal Git credential helper changed while hashing")
    with tempfile.TemporaryDirectory(
        prefix="lha_formal_gh_identity_"
    ) as temporary:
        temporary_root = Path(temporary)
        result = run(
            [str(resolved), "--version"],
            cwd=temporary_root,
            timeout=30,
            env={
                "PATH": sanitized_absolute_path(
                    extra_dirs=(resolved.parent,),
                    require_unwritable=False,
                ),
                "HOME": str(temporary_root),
                "XDG_CONFIG_HOME": str(temporary_root / "config"),
                "XDG_STATE_HOME": str(temporary_root / "state"),
                "GH_CONFIG_DIR": str(temporary_root / "gh"),
                "LANG": "C",
                "LC_ALL": "C",
            },
        )
    version = result.stdout.rstrip("\n")
    if (
        result.returncode != 0
        or result.output_truncated
        or result.cleanup_unconfirmed
        or not version
        or len(version.encode("utf-8")) > 16 * 1024
        or version.strip() != version
        or "\x00" in version
    ):
        raise RuntimeError("formal Git credential helper version is unavailable")
    try:
        measured = FormalGitCredentialHelper(
            host=host,
            executable_path=str(resolved),
            executable_sha256=digest.hexdigest(),
            version=version,
            command=f"!{resolved} auth git-credential",
        )
    except ValueError as error:
        raise RuntimeError("formal Git credential helper identity is invalid") from error
    if expected is not None and measured != expected:
        raise RuntimeError(
            "formal Git credential helper differs from its registration"
        )
    return measured


def _preflight_formal_git_credential_helper(
    git_path: str,
    helper: FormalGitCredentialHelper,
) -> dict[str, Any]:
    """Verify helper output shape without returning or logging credential values."""
    measured = _formal_git_credential_helper(
        helper.host,
        expected=helper,
    )
    command = [
        git_path,
        "-c",
        "credential.helper=",
        "-c",
        (
            f"credential.https://{measured.host}.helper="
            f"{measured.command}"
        ),
        "credential",
        "fill",
    ]
    with _git_authenticated_push_env(measured) as authenticated_env:
        result = run(
            command,
            cwd=Path(authenticated_env["HOME"]),
            timeout=30,
            env=authenticated_env,
            input=f"protocol=https\nhost={measured.host}\n\n",
        )
        raw = result.stdout
        fields: dict[str, str] = {}
        field_names: tuple[str, ...] = ()
        try:
            if (
                result.returncode != 0
                or result.output_truncated
                or result.cleanup_unconfirmed
                or not raw
                or len(raw.encode("utf-8")) > 64 * 1024
            ):
                raise RuntimeError("formal Git credential helper preflight failed")
            for line in raw.splitlines():
                key, separator, value = line.partition("=")
                if (
                    not separator
                    or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) is None
                    or key in fields
                    or not value
                ):
                    raise RuntimeError(
                        "formal Git credential helper preflight returned invalid fields"
                    )
                fields[key] = value
            allowed = {
                "protocol",
                "host",
                "username",
                "password",
                "password_expiry_utc",
                "oauth_refresh_token",
            }
            if (
                not {"protocol", "host", "username", "password"}.issubset(fields)
                or not set(fields).issubset(allowed)
                or fields["protocol"] != "https"
                or fields["host"] != measured.host
            ):
                raise RuntimeError(
                    "formal Git credential helper preflight returned invalid fields"
                )
            field_names = tuple(sorted(fields))
        finally:
            # Credential values exist only in this local frame and the child
            # process environment. Never return them or attach them to errors.
            result.stdout = ""
            result.stderr = ""
            raw = ""
            fields.clear()
            del result
    return {
        "host": measured.host,
        "fields": field_names,
    }


def _formal_anonymous_git_output(
    git_path: str,
    arguments: list[str],
    *,
    label: str,
) -> str:
    """Probe a public remote without repository-local or user Git settings."""
    with tempfile.TemporaryDirectory(prefix="lha_formal_remote_") as temporary:
        result = run(
            [git_path, *arguments],
            cwd=Path(temporary),
            timeout=30,
            env=_git_control_env(),
        )
    if (
        result.returncode != 0
        or result.output_truncated
        or result.cleanup_unconfirmed
    ):
        raise RuntimeError(f"formal ablation failed {label}")
    return result.stdout


def _create_formal_start_witness(
    attempt: _FormalAttemptBinding,
    *,
    outcome_key: str,
    run_header_sha256: str,
) -> str:
    """Atomically consume a registration through its preregistered remote ref."""
    helper = _formal_git_credential_helper(
        attempt.witness_credential_helper.host,
        expected=attempt.witness_credential_helper,
    )
    tree = _formal_git_output(
        attempt.git_path,
        ["rev-parse", f"{attempt.registration_commit}^{{tree}}"],
        repository_root=attempt.repository_root,
        label="witness tree",
    ).strip()
    message = formal_ablation_witness_message(
        attempt_id=attempt.attempt_id,
        registration_registry_sha256=attempt.registry_sha256,
        protocol_sha256=attempt.protocol_sha256,
        outcome_key=outcome_key,
        run_header_sha256=run_header_sha256,
    )
    try:
        commit_payload = formal_ablation_witness_commit_bytes(
            tree=tree,
            parent=attempt.registration_commit,
            message=message,
        )
    except ValueError as error:
        raise RuntimeError("formal ablation witness inputs are invalid") from error
    witness_commit = formal_ablation_witness_commit_oid(commit_payload)
    stored_commit = _formal_git_output(
        attempt.git_path,
        ["hash-object", "-t", "commit", "-w", "--stdin"],
        repository_root=attempt.repository_root,
        label="witness object creation",
        input=commit_payload.decode("ascii"),
    ).strip()
    if stored_commit != witness_commit:
        raise RuntimeError("formal ablation witness object identity is inconsistent")

    refspec = f"{witness_commit}:{attempt.witness_ref}"
    with _git_authenticated_push_env(helper) as authenticated_env:
        push = run(
            [
                attempt.git_path,
                "-c",
                "credential.helper=",
                "-c",
                (
                    f"credential.https://{helper.host}.helper="
                    f"{helper.command}"
                ),
                "push",
                "--porcelain",
                "--atomic",
                "--no-verify",
                "--no-follow-tags",
                "--recurse-submodules=no",
                f"--force-with-lease={attempt.witness_ref}:",
                attempt.witness_remote_url,
                refspec,
            ],
            cwd=attempt.repository_root,
            timeout=60,
            env=authenticated_env,
        )
    _formal_git_credential_helper(
        helper.host,
        expected=helper,
    )
    expected_statuses = {
        f"*\t{refspec}\t[new branch]",
        f"*\t{refspec}\t[new reference]",
    }
    status_lines = {
        line
        for line in push.stdout.splitlines()
        if line[:2] in {"*\t", "=\t", "!\t", "+\t", "-\t"}
    }
    if (
        push.returncode != 0
        or push.output_truncated
        or push.cleanup_unconfirmed
        or status_lines.isdisjoint(expected_statuses)
        or any(not line.startswith("*\t") for line in status_lines)
    ):
        raise RuntimeError(
            "formal ablation start witness was not created; the registered "
            "attempt cannot run"
        )

    remote_ref = _formal_anonymous_git_output(
        attempt.git_path,
        ["ls-remote", "--refs", attempt.witness_remote_url, attempt.witness_ref],
        label="witness remote confirmation",
    ).strip()
    if remote_ref != f"{witness_commit}\t{attempt.witness_ref}":
        raise RuntimeError(
            "formal ablation start witness was not confirmed; the registered "
            "attempt cannot run"
        )
    return witness_commit


def _bind_formal_attempt(
    *,
    formal_corpus: _FormalCorpusBinding,
    formal_output_lease: _FormalOutputLease,
    model: str,
    reasoning_effort: str,
    docker_image_id: str,
    source_tree_sha256: str,
    codex_cli_version: str,
    codex_cli_executable_sha256: str,
    codex_client: FormalCodexClientConfig,
) -> _FormalAttemptBinding:
    """Require one committed open registration before snapshots or model calls."""
    repository_root = _project_root()
    if repository_root is None:
        raise RuntimeError("formal ablation attempt registry requires a project checkout")
    repository_root = repository_root.resolve(strict=True)
    try:
        output_path = formal_output_lease.path.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise RuntimeError(
            "formal ablation output must be a fixed repository-relative path"
        ) from error

    git_path = str(formal_corpus.git_executable.get("path", ""))
    if not Path(git_path).is_absolute():
        raise RuntimeError("formal ablation attempt registry has no trusted Git executable")
    head = _formal_git_output(
        git_path,
        ["rev-parse", "--verify", "HEAD"],
        repository_root=repository_root,
        label="HEAD resolution",
    ).strip()
    if head != formal_corpus.preregistration_commit:
        raise RuntimeError("formal ablation HEAD changed before attempt validation")
    branch = _formal_git_output(
        git_path,
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        repository_root=repository_root,
        label="branch resolution",
    ).strip()
    if not branch:
        raise RuntimeError("formal ablation requires a named registration branch")
    _formal_git_output(
        git_path,
        ["check-ref-format", f"refs/heads/{branch}"],
        repository_root=repository_root,
        label="branch validation",
    )
    trusted_source_files = _revalidate_formal_checkout(formal_corpus)
    if _source_tree_digest(trusted_source_files) != source_tree_sha256:
        raise RuntimeError(
            "formal ablation source differs from its registered HEAD bytes"
        )
    registry_relative = FORMAL_ABLATION_ATTEMPTS_PATH.as_posix()
    _formal_git_output(
        git_path,
        ["ls-files", "--error-unmatch", "--", registry_relative],
        repository_root=repository_root,
        label="tracked-file check",
    )
    status = _formal_git_output(
        git_path,
        ["status", "--porcelain=v1", "--untracked-files=normal"],
        repository_root=repository_root,
        label="worktree status",
    )
    if status:
        raise RuntimeError(
            "formal ablation attempt registry requires a clean committed worktree"
        )
    size_text = _formal_git_output(
        git_path,
        ["cat-file", "-s", f"{head}:{registry_relative}"],
        repository_root=repository_root,
        label="committed registry size",
    ).strip()
    try:
        committed_size = int(size_text)
    except ValueError as error:
        raise RuntimeError(
            "formal ablation attempt registry has an invalid Git size"
        ) from error
    if (
        committed_size < 0
        or committed_size > MAX_FORMAL_ABLATION_ATTEMPTS_BYTES
    ):
        raise RuntimeError("formal ablation attempt registry is too large")
    committed_text = _formal_git_output(
        git_path,
        ["show", f"{head}:{registry_relative}"],
        repository_root=repository_root,
        label="committed registry read",
    )
    committed_bytes = committed_text.encode("utf-8")
    if len(committed_bytes) != committed_size:
        raise RuntimeError("formal ablation attempt registry changed while reading")
    registry_path = repository_root / FORMAL_ABLATION_ATTEMPTS_PATH
    try:
        current_bytes = _read_bounded_bytes(
            registry_path,
            max_bytes=MAX_FORMAL_ABLATION_ATTEMPTS_BYTES,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError("formal ablation attempt registry is unsafe") from error
    if current_bytes != committed_bytes:
        raise RuntimeError(
            "formal ablation attempt registry differs from the committed HEAD"
        )
    try:
        registry = parse_formal_ablation_attempt_registry(committed_bytes)
    except ValueError as error:
        raise RuntimeError("formal ablation attempt registry is invalid") from error
    registration = registry.open_registration()
    if not isinstance(registration, RegisteredAttempt):
        raise RuntimeError(
            "formal ablation requires exactly one open REGISTERED attempt"
        )

    _formal_git_output(
        git_path,
        ["cat-file", "-e", f"{registration.source_commit}^{{commit}}"],
        repository_root=repository_root,
        label="registered source commit",
    )
    parents = _formal_git_output(
        git_path,
        [
            "rev-list",
            "--parents",
            "-n",
            "1",
            head,
        ],
        repository_root=repository_root,
        label="registration parents",
    ).strip().split()
    if parents != [head, registration.source_commit]:
        raise RuntimeError(
            "formal ablation registration commit must directly follow its source commit"
        )
    changed_paths = _formal_git_output(
        git_path,
        [
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            head,
        ],
        repository_root=repository_root,
        label="source/registration content",
    ).splitlines()
    if changed_paths != [registry_relative]:
        raise RuntimeError(
            "formal ablation registration commit may only change the attempt registry"
        )
    configured_witness_url = _formal_witness_remote_url(
        git_path,
        repository_root=repository_root,
        remote_name=registration.witness_remote_name,
    )
    if configured_witness_url != registration.witness_remote_url:
        raise RuntimeError(
            "formal ablation witness remote differs from its registration"
        )
    if registration.witness_credential_helper is None:
        raise RuntimeError(
            "formal ablation registration has no credential helper binding"
        )
    witness_credential_helper = _formal_git_credential_helper(
        registration.witness_credential_helper.host,
        expected=registration.witness_credential_helper,
    )
    _preflight_formal_git_credential_helper(
        git_path,
        witness_credential_helper,
    )
    protocol = FormalAblationProtocol(
        source_commit=registration.source_commit,
        source_tree_sha256=source_tree_sha256,
        manifest_sha256=formal_corpus.sha256,
        model=model,
        reasoning_effort=reasoning_effort,
        docker_image_id=docker_image_id,
        codex_cli_version=codex_cli_version,
        codex_cli_executable_sha256=codex_cli_executable_sha256,
        codex_client=codex_client,
        codex_client_sha256=formal_codex_client_sha256(codex_client),
        witness_credential_helper=witness_credential_helper,
    )
    protocol_sha256 = formal_ablation_protocol_sha256(protocol)
    expected = {
        "source_commit": protocol.source_commit,
        "source_tree_sha256": protocol.source_tree_sha256,
        "manifest_sha256": protocol.manifest_sha256,
        "output_path": output_path,
        "model": protocol.model,
        "reasoning_effort": protocol.reasoning_effort,
        "docker_image_id": protocol.docker_image_id,
        "codex_cli_version": protocol.codex_cli_version,
        "codex_cli_executable_sha256": (
            protocol.codex_cli_executable_sha256
        ),
        "codex_client": protocol.codex_client,
        "codex_client_sha256": protocol.codex_client_sha256,
        "witness_credential_helper": protocol.witness_credential_helper,
        "protocol_sha256": protocol_sha256,
    }
    if any(getattr(registration, field) != value for field, value in expected.items()):
        raise RuntimeError(
            "open formal ablation registration does not match this run"
        )
    remote_branch = _formal_anonymous_git_output(
        git_path,
        [
            "ls-remote",
            "--heads",
            registration.witness_remote_url,
            f"refs/heads/{branch}",
        ],
        label="registration branch confirmation",
    ).strip()
    if remote_branch.split() != [head, f"refs/heads/{branch}"]:
        raise RuntimeError(
            "formal ablation registration commit is not published on "
            "the current remote branch"
        )
    return _FormalAttemptBinding(
        attempt_id=registration.attempt_id,
        registry_path=registry_relative,
        registry_sha256=hashlib.sha256(committed_bytes).hexdigest(),
        protocol_sha256=protocol_sha256,
        registration_commit=head,
        repository_root=repository_root,
        git_path=git_path,
        witness_remote_name=registration.witness_remote_name,
        witness_remote_url=registration.witness_remote_url,
        witness_credential_helper=witness_credential_helper,
        witness_ref=registration.witness_ref,
    )


def _prepare_formal_corpus_binding(
    task_paths: list[str],
    *,
    repetitions: int,
) -> _FormalCorpusBinding | None:
    """Fail before model execution when a formal-grid run is not preregistered."""
    if len(task_paths) != _FORMAL_TASK_COUNT or repetitions != _FORMAL_REPETITIONS:
        return None
    repository_root = _project_root()
    if repository_root is None:
        raise RuntimeError("formal ablation requires a Git project checkout")
    manifest_path = repository_root / _FORMAL_CORPUS_MANIFEST_PATH
    manifest, manifest_sha256 = _load_formal_corpus_manifest(
        manifest_path,
        repository_root,
    )
    supplied_paths = [_provenance_path(path) for path in task_paths]
    registered_paths = [entry["task_path"] for entry in manifest["tasks"]]
    if supplied_paths != registered_paths:
        raise RuntimeError(
            "formal ablation tasks or task order differ from the preregistered manifest"
        )
    git_executable = _trusted_control_executable("git")
    git_path = str(git_executable["path"])
    commit, dirty = _git_provenance(git_path)
    if commit is None or _HEX_40.fullmatch(commit) is None or dirty is not False:
        raise RuntimeError("formal ablation must start from a clean committed Git checkout")
    _validate_formal_head_checkout(
        repository_root,
        git_path=git_path,
        head=commit,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    return _FormalCorpusBinding(
        path=_FORMAL_CORPUS_MANIFEST_PATH.as_posix(),
        sha256=manifest_sha256,
        preregistration_commit=commit,
        git_executable=git_executable,
    )


def _source_file_digests(root: Path | None = None) -> dict[str, str]:
    """Hash the complete installed ``lha`` source tree.

    A hand-maintained module list is unsafe here: adding a helper to the scorer,
    patcher, policy, verifier, or aggregation code would otherwise leave the
    cache bound to the old behavior. Package data is included as well as Python
    modules so moving a runtime resource under ``src/lha`` cannot evade the
    provenance record.
    """
    package_root = (root or Path(__file__).resolve().parent).resolve()
    files: dict[str, str] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files[path.relative_to(package_root).as_posix()] = _sha256_regular_file(path)
    return files


def _source_tree_digest(source_files: dict[str, str]) -> str:
    return _canonical_digest(source_files)


def _git_head_tree_entries(
    repository_root: Path,
    *,
    git_path: str,
    commit: str,
    paths: list[str],
) -> dict[str, tuple[str, int, str]]:
    """List regular blobs below fixed paths without consulting the index."""
    if (
        not Path(git_path).is_absolute()
        or _HEX_40.fullmatch(commit) is None
        or not paths
    ):
        raise RuntimeError("formal Git tree request is invalid")
    requested: list[str] = []
    for value in paths:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != value
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise RuntimeError("formal Git tree path is unsafe")
        if value not in requested:
            requested.append(value)
    output = _formal_git_output(
        git_path,
        [
            "ls-tree",
            "-r",
            "-l",
            "-z",
            "--full-tree",
            commit,
            "--",
            *requested,
        ],
        repository_root=repository_root,
        label="trusted HEAD tree read",
    )
    if "\ufffd" in output:
        raise RuntimeError("formal Git tree contains a non-UTF-8 path")
    entries: dict[str, tuple[str, int, str]] = {}
    total_bytes = 0
    for record in output.split("\0"):
        if not record:
            continue
        metadata, separator, name = record.partition("\t")
        fields = metadata.split()
        if (
            not separator
            or len(fields) != 4
            or fields[0] not in {"100644", "100755"}
            or fields[1] != "blob"
            or re.fullmatch(r"[0-9a-f]{40,64}", fields[2]) is None
        ):
            raise RuntimeError("formal Git tree contains a non-regular entry")
        try:
            size = int(fields[3])
        except ValueError as error:
            raise RuntimeError("formal Git tree contains an invalid blob size") from error
        relative = PurePosixPath(name)
        if (
            size < 0
            or relative.is_absolute()
            or relative.as_posix() != name
            or any(part in {"", ".", ".."} for part in relative.parts)
            or not any(
                name == prefix or name.startswith(f"{prefix}/")
                for prefix in requested
            )
            or name in entries
        ):
            raise RuntimeError("formal Git tree returned an unexpected entry")
        total_bytes += size
        if (
            len(entries) >= _MAX_FORMAL_HEAD_FILES
            or total_bytes > _MAX_FORMAL_HEAD_BYTES
        ):
            raise RuntimeError("formal Git tree exceeds the trusted byte bound")
        entries[name] = (fields[2], size, fields[0])
    return entries


def _validate_formal_git_tree_modes(
    repository_root: Path,
    entries: dict[str, tuple[str, int, str]],
) -> None:
    """Match Git's executable-bit model against the current worktree."""
    try:
        root = repository_root.resolve(strict=True)
        root_metadata = root.lstat()
    except (OSError, RuntimeError) as error:
        raise RuntimeError("formal worktree is unavailable") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("formal worktree root is unsafe")
    for name, (_oid, _size, git_mode) in entries.items():
        candidate = root
        parts = PurePosixPath(name).parts
        try:
            for component in parts[:-1]:
                candidate /= component
                parent_metadata = candidate.lstat()
                if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
                    parent_metadata.st_mode
                ):
                    raise RuntimeError("formal worktree file path is unsafe")
            candidate /= parts[-1]
            metadata = candidate.lstat()
        except OSError as error:
            raise RuntimeError("formal worktree file is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("formal worktree file is not regular")
        expected_executable = git_mode == "100755"
        actual_executable = bool(stat.S_IMODE(metadata.st_mode) & 0o111)
        if actual_executable != expected_executable:
            raise RuntimeError(
                "formal worktree file mode differs from the trusted Git tree"
            )


def _git_blob_batch(
    repository_root: Path,
    *,
    git_path: str,
    entries: dict[str, tuple[str, int, str]],
) -> dict[str, bytes]:
    """Read exact committed blob bytes through one bounded, isolated Git process."""
    if not process_group_cleanup_supported():
        raise RuntimeError("formal Git blob reads require POSIX process cleanup")
    ordered_oids = list(
        dict.fromkeys(oid for oid, _size, _mode in entries.values())
    )
    if not ordered_oids:
        return {}
    expected_sizes: dict[str, int] = {}
    for oid, size, _mode in entries.values():
        previous = expected_sizes.setdefault(oid, size)
        if previous != size:
            raise RuntimeError("formal Git blob size is inconsistent")
    request = b"".join(f"{oid}\n".encode("ascii") for oid in ordered_oids)
    try:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            [git_path, "cat-file", "--batch"],
            cwd=repository_root,
            env=_git_control_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise RuntimeError("formal Git blob reader could not start") from error
    timed_out = False
    interrupted: BaseException | None = None
    stdout = b""
    stderr = b""
    try:
        stdout, stderr = process.communicate(input=request, timeout=30)
    except subprocess.TimeoutExpired:
        timed_out = True
    except BaseException as error:
        interrupted = error
    cleanup = terminate_process_group(process)
    if timed_out:
        try:
            stdout, stderr = process.communicate(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if interrupted is not None:
        if not cleanup.confirmed:
            raise RuntimeError(
                "formal Git blob reader cleanup could not be confirmed"
            ) from interrupted
        raise interrupted
    if (
        timed_out
        or process.returncode != 0
        or not cleanup.confirmed
        or len(stderr) > 1024 * 1024
        or len(stdout)
        > _MAX_FORMAL_HEAD_BYTES + len(ordered_oids) * 160
    ):
        raise RuntimeError("formal Git blob reader failed closed")

    blobs_by_oid: dict[str, bytes] = {}
    offset = 0
    for expected_oid in ordered_oids:
        newline = stdout.find(b"\n", offset)
        if newline < 0:
            raise RuntimeError("formal Git blob stream is truncated")
        try:
            header = stdout[offset:newline].decode("ascii").split()
        except UnicodeDecodeError as error:
            raise RuntimeError("formal Git blob header is invalid") from error
        expected_size = expected_sizes[expected_oid]
        if (
            len(header) != 3
            or header[0] != expected_oid
            or header[1] != "blob"
            or header[2] != str(expected_size)
        ):
            raise RuntimeError("formal Git blob header disagrees with the tree")
        start = newline + 1
        end = start + expected_size
        if end >= len(stdout) or stdout[end : end + 1] != b"\n":
            raise RuntimeError("formal Git blob payload is truncated")
        blobs_by_oid[expected_oid] = stdout[start:end]
        offset = end + 1
    if offset != len(stdout):
        raise RuntimeError("formal Git blob stream contains trailing bytes")
    return {
        name: blobs_by_oid[oid]
        for name, (oid, _size, _mode) in entries.items()
    }


def _trusted_git_blobs(
    repository_root: Path,
    *,
    git_path: str,
    commit: str,
    paths: list[str],
) -> dict[str, bytes]:
    entries = _git_head_tree_entries(
        repository_root,
        git_path=git_path,
        commit=commit,
        paths=paths,
    )
    _validate_formal_git_tree_modes(repository_root, entries)
    blobs = _git_blob_batch(
        repository_root,
        git_path=git_path,
        entries=entries,
    )
    _validate_formal_git_tree_modes(repository_root, entries)
    return blobs


def _formal_tree_file_digests(root: Path) -> dict[str, str]:
    _validate_formal_tree_nodes(root, label="formal corpus")
    files: dict[str, str] = {}
    for relative, path in sorted(_iter_files(root)):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("formal corpus contains a non-regular file")
        files[relative] = _sha256_regular_file(path)
    _validate_formal_tree_nodes(root, label="formal corpus")
    return files


def _validate_formal_tree_nodes(root: Path, *, label: str) -> None:
    """Reject links and special nodes that ``copytree`` could follow or copy."""
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} tree is unavailable") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
        root_metadata.st_mode
    ):
        raise RuntimeError(f"{label} root is not a real directory")
    count = 0
    try:
        for path in root.rglob("*"):
            count += 1
            if count > _MAX_FORMAL_HEAD_FILES:
                raise RuntimeError(f"{label} tree exceeds the trusted file bound")
            metadata = path.lstat()
            if not (
                stat.S_ISDIR(metadata.st_mode)
                or (
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_nlink == 1
                )
            ):
                raise RuntimeError(
                    f"{label} tree contains a link or special node"
                )
    except OSError as error:
        raise RuntimeError(f"{label} tree could not be inspected") from error


def _formal_input_snapshot_is_valid(
    path: Path,
    *,
    task: str,
    task_sha256: str,
    corpus_sha256: str,
    snapshot_sha256: str,
) -> bool:
    """Recompute every byte binding referenced by a formal cell marker."""
    expected_metadata = {
        "schema_version": _INPUT_SNAPSHOT_SCHEMA,
        "task": task,
        "task_sha256": task_sha256,
        "corpus_sha256": corpus_sha256,
        "snapshot_sha256": snapshot_sha256,
    }
    if snapshot_sha256 != _input_snapshot_digest(task_sha256, corpus_sha256):
        return False
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            return False
        entries = {entry.name: entry for entry in path.iterdir()}
        if set(entries) != {"task.yaml", "repo", "snapshot.json"}:
            return False
        task_payload = _read_bounded_bytes(
            entries["task.yaml"],
            max_bytes=_MAX_TASK_BYTES,
        )
        metadata_payload = _read_bounded_bytes(
            entries["snapshot.json"],
            max_bytes=_MAX_TASK_BYTES,
        )
        if (
            _sha256_bytes(task_payload) != task_sha256
            or metadata_payload
            != _canonical_json_object_bytes(expected_metadata)
            or json.loads(metadata_payload) != expected_metadata
        ):
            return False
        repository = entries["repo"]
        _validate_formal_tree_nodes(
            repository,
            label="formal input snapshot",
        )
        if _repo_digest(repository) != corpus_sha256:
            return False
        _validate_formal_tree_nodes(
            repository,
            label="formal input snapshot",
        )
    except (
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


def _head_repo_digest(blobs: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, payload in sorted(blobs.items()):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_formal_head_checkout(
    repository_root: Path,
    *,
    git_path: str,
    head: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, str]:
    """Bind source, manifest, tasks, and corpora to exact committed blobs."""
    manifest_relative = _FORMAL_CORPUS_MANIFEST_PATH.as_posix()
    source_relative = "src/lha"
    registered_inputs = [
        value
        for entry in manifest["tasks"]
        for value in (entry["task_path"], entry["corpus_path"])
    ]
    requested = [
        source_relative,
        manifest_relative,
        *_FORMAL_CONTROL_FILES,
        *registered_inputs,
    ]
    head_blobs = _trusted_git_blobs(
        repository_root,
        git_path=git_path,
        commit=head,
        paths=requested,
    )
    manifest_path = _repo_relative_evidence_path(
        repository_root,
        manifest_relative,
        kind="manifest",
    )[1]
    current_manifest = _read_bounded_bytes(
        manifest_path,
        max_bytes=_MAX_FORMAL_MANIFEST_BYTES,
    )
    if (
        head_blobs.get(manifest_relative) != current_manifest
        or _sha256_bytes(current_manifest) != manifest_sha256
    ):
        raise RuntimeError(
            "formal corpus manifest differs from the trusted HEAD blob"
        )

    for control_relative in _FORMAL_CONTROL_FILES:
        expected_control = head_blobs.get(control_relative)
        candidate = repository_root / control_relative
        present = candidate.exists() or candidate.is_symlink()
        if expected_control is None:
            if present:
                raise RuntimeError(
                    f"formal control file {control_relative!r} is not in HEAD"
                )
            continue
        if not present:
            raise RuntimeError(
                f"formal control file {control_relative!r} is missing"
            )
        control_path = _repo_relative_evidence_path(
            repository_root,
            control_relative,
            kind="control",
        )[1]
        current_control = _read_bounded_bytes(
            control_path,
            max_bytes=_MAX_FORMAL_HEAD_BYTES,
        )
        if current_control != expected_control:
            raise RuntimeError(
                f"formal control file {control_relative!r} differs from HEAD"
            )

    source_root = _repo_relative_evidence_path(
        repository_root,
        source_relative,
        kind="source",
    )[1]
    if not source_root.is_dir():
        raise RuntimeError("formal source tree is unavailable")
    _validate_formal_tree_nodes(source_root, label="formal source")
    prefix = f"{source_relative}/"
    head_source_files = {
        name.removeprefix(prefix): _sha256_bytes(payload)
        for name, payload in head_blobs.items()
        if name.startswith(prefix)
        and "__pycache__" not in PurePosixPath(name).parts
        and PurePosixPath(name).suffix not in {".pyc", ".pyo"}
    }
    if (
        not head_source_files
        or _source_file_digests(source_root) != head_source_files
    ):
        raise RuntimeError(
            "formal source bytes differ from the trusted HEAD blobs"
        )
    _validate_formal_tree_nodes(source_root, label="formal source")

    for entry in manifest["tasks"]:
        task_relative = entry["task_path"]
        corpus_relative = entry["corpus_path"]
        task_path = _repo_relative_evidence_path(
            repository_root,
            task_relative,
            kind="task",
        )[1]
        current_task = _read_bounded_bytes(
            task_path,
            max_bytes=_MAX_TASK_BYTES,
        )
        if (
            head_blobs.get(task_relative) != current_task
            or _sha256_bytes(current_task) != entry["task_sha256"]
        ):
            raise RuntimeError(
                f"formal task {entry['name']!r} differs from the trusted HEAD blob"
            )
        corpus_root = _repo_relative_evidence_path(
            repository_root,
            corpus_relative,
            kind="corpus",
        )[1]
        corpus_prefix = f"{corpus_relative}/"
        head_corpus_bytes = {
            name.removeprefix(corpus_prefix): payload
            for name, payload in head_blobs.items()
            if name.startswith(corpus_prefix)
            and not any(
                part in _DIFF_IGNORE
                for part in PurePosixPath(name.removeprefix(corpus_prefix)).parts
            )
        }
        head_corpus_files = {
            name: _sha256_bytes(payload)
            for name, payload in head_corpus_bytes.items()
        }
        if (
            not head_corpus_files
            or _formal_tree_file_digests(corpus_root) != head_corpus_files
            or _head_repo_digest(head_corpus_bytes) != entry["corpus_sha256"]
        ):
            raise RuntimeError(
                f"formal corpus {entry['name']!r} differs from trusted HEAD blobs"
            )

    _formal_git_output(
        git_path,
        [
            "merge-base",
            "--is-ancestor",
            manifest["corpus_commit"],
            head,
        ],
        repository_root=repository_root,
        label="formal corpus commit ancestry",
    )
    corpus_commit_blobs = _trusted_git_blobs(
        repository_root,
        git_path=git_path,
        commit=manifest["corpus_commit"],
        paths=registered_inputs,
    )
    head_inputs = {
        name: payload
        for name, payload in head_blobs.items()
        if any(
            name == value or name.startswith(f"{value}/")
            for value in registered_inputs
        )
    }
    if corpus_commit_blobs != head_inputs:
        raise RuntimeError(
            "formal task or corpus bytes changed after the fixed corpus commit"
        )
    return head_source_files


def _revalidate_formal_checkout(
    formal_corpus: _FormalCorpusBinding,
) -> dict[str, str]:
    repository_root = _project_root()
    if repository_root is None:
        raise RuntimeError("formal ablation requires a Git project checkout")
    repository_root = repository_root.resolve(strict=True)
    git_path = str(formal_corpus.git_executable.get("path", ""))
    manifest, manifest_sha256 = _load_formal_corpus_manifest(
        repository_root / _FORMAL_CORPUS_MANIFEST_PATH,
        repository_root,
    )
    if manifest_sha256 != formal_corpus.sha256:
        raise RuntimeError("formal corpus manifest changed after preregistration")
    return _validate_formal_head_checkout(
        repository_root,
        git_path=git_path,
        head=formal_corpus.preregistration_commit,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )


def _write_ablation_reports(
    out: Path,
    *,
    report_json: str,
    report_markdown: str,
    formal_corpus: _FormalCorpusBinding | None,
    source_files: dict[str, str],
) -> None:
    if formal_corpus is None:
        _atomic_write(out / "ablation_report.json", report_json)
        _atomic_write(out / "ablation_report.md", report_markdown)
        return
    before_publish = _revalidate_formal_checkout(formal_corpus)
    if before_publish != source_files:
        raise RuntimeError("formal checkout changed before report publication")
    # JSON is the formal completion marker. A stop after the derived Markdown
    # write leaves no JSON and can therefore be closed as ABANDONED rather than
    # mistaken for a complete result.
    _atomic_write(
        out / "ablation_report.md",
        report_markdown,
        anchor=out,
    )
    after_markdown = _revalidate_formal_checkout(formal_corpus)
    if after_markdown != source_files:
        raise RuntimeError(
            "formal checkout changed before the report commit marker"
        )
    _atomic_write(
        out / "ablation_report.json",
        report_json,
        anchor=out,
    )


def _project_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lha").is_dir():
            return candidate
    return None


def _trusted_control_executable(
    name: str,
    *,
    executable: str | None = None,
) -> dict[str, Any]:
    """Bind a control command to immutable-looking bytes outside writable PATH entries."""
    if executable is None:
        resolved_text = trusted_executable(name, require_unwritable=True)
    else:
        configured = Path(executable)
        if not configured.is_absolute():
            raise RuntimeError(f"{name} executable must be an absolute path")
        try:
            resolved = configured.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(f"{name} executable is unavailable") from error
        if resolved.name != name:
            raise RuntimeError(f"{name} executable has an unexpected basename")
        resolved_text = trusted_executable(
            resolved.name,
            path="",
            extra_dirs=(resolved.parent,),
            require_unwritable=True,
        )
        if resolved_text != str(resolved):
            resolved_text = None
    if resolved_text is None:
        raise RuntimeError(f"{name} executable was not found in a non-writable installation")
    path = Path(resolved_text)
    before = path.lstat()
    if (
        not path.is_absolute()
        or not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > _MAX_CONTROL_EXECUTABLE_BYTES
    ):
        raise RuntimeError(f"{name} executable is not a bounded regular file")
    digest = hashlib.sha256()
    _consume_stable_regular_file(
        path,
        digest.update,
        max_bytes=_MAX_CONTROL_EXECUTABLE_BYTES,
        reject_hardlinks=False,
    )
    after = path.lstat()
    if _stable_file_signature(before) != _stable_file_signature(after):
        raise RuntimeError(f"{name} executable changed while it was bound")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": after.st_size,
        "trusted_install": True,
    }


def _git_control_env() -> dict[str, str]:
    """Minimal deterministic environment for absolute-path Git control calls."""
    env = {
        "PATH": sanitized_absolute_path(require_unwritable=True),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
    }
    return {key: value for key, value in env.items() if value}


def _git_local_config_env() -> dict[str, str]:
    """Permit only an explicit ``git config --local --no-includes`` read."""
    env = _git_control_env()
    env.pop("GIT_CONFIG", None)
    return env


def _gh_config_source_directory() -> Path:
    """Resolve the official gh config directory without following aliases."""
    configured = os.environ.get("GH_CONFIG_DIR")
    if configured:
        candidate = Path(configured)
    else:
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        candidate = (
            Path(xdg_config) / "gh"
            if xdg_config
            else Path.home() / ".config" / "gh"
        )
    if not candidate.is_absolute():
        raise RuntimeError("formal Git credential configuration is unavailable")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.lstat()
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            "formal Git credential configuration is unavailable"
        ) from error
    if (
        resolved != candidate
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise RuntimeError("formal Git credential configuration is unsafe")
    return resolved


def _gh_config_signatures(source: Path) -> dict[str, tuple[int, ...]]:
    """Bind the files gh may read without copying or decoding their contents."""
    signatures: dict[str, tuple[int, ...]] = {}
    for name in ("hosts.yml", "config.yml"):
        candidate = source / name
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            if name == "hosts.yml":
                raise RuntimeError(
                    "formal Git credential configuration is incomplete"
                ) from None
            continue
        except OSError as error:
            raise RuntimeError(
                "formal Git credential configuration is unsafe"
            ) from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_GH_CONFIG_BYTES
        ):
            raise RuntimeError("formal Git credential configuration is unsafe")
        signatures[name] = _stable_file_signature(metadata)
    return signatures


@contextmanager
def _git_authenticated_push_env(
    helper: FormalGitCredentialHelper,
) -> Iterator[dict[str, str]]:
    """Use a memory-only token with disposable writable gh directories."""
    source = _gh_config_source_directory()
    before = _gh_config_signatures(source)
    helper_path = helper.executable_path
    with tempfile.TemporaryDirectory(
        prefix="lha_formal_git_auth_"
    ) as temporary:
        root = Path(temporary)
        root.chmod(0o700)
        directories = {
            "home": root / "home",
            "config": root / "config",
            "state": root / "state",
            "cache": root / "cache",
            "data": root / "data",
            "runtime": root / "runtime",
            "tmp": root / "tmp",
            "gh": root / "gh",
        }
        for directory in directories.values():
            directory.mkdir(mode=0o700)
            directory.chmod(0o700)
        token_env = _git_control_env()
        token_env.update(
            {
                # On macOS, gh's keychain lookup is bound to the account home.
                # All gh write locations remain redirected below the disposable
                # root; only this read-only token lookup sees the real HOME.
                "HOME": str(Path.home().resolve()),
                "GH_CONFIG_DIR": str(source),
                "XDG_CONFIG_HOME": str(directories["config"]),
                "XDG_STATE_HOME": str(directories["state"]),
                "XDG_CACHE_HOME": str(directories["cache"]),
                "XDG_DATA_HOME": str(directories["data"]),
                "XDG_RUNTIME_DIR": str(directories["runtime"]),
                "TMPDIR": str(directories["tmp"]),
                "GH_HOST": helper.host,
                "GH_PROMPT_DISABLED": "1",
                "GH_NO_UPDATE_NOTIFIER": "1",
            }
        )
        token_result = run(
            [helper_path, "auth", "token", "--hostname", helper.host],
            cwd=directories["home"],
            timeout=30,
            env=token_env,
        )
        token = ""
        try:
            token = token_result.stdout.strip()
            invalid_token = (
                token_result.returncode != 0
                or token_result.output_truncated
                or token_result.cleanup_unconfirmed
                or not token
                or len(token.encode("utf-8")) > 16 * 1024
                or any(
                    character.isspace() or ord(character) < 32
                    for character in token
                )
            )
            config_changed = _gh_config_signatures(source) != before
        finally:
            token_result.stdout = ""
            token_result.stderr = ""
            del token_result
        if invalid_token or config_changed:
            token = ""
            raise RuntimeError("formal Git credential token preflight failed")
        env = _git_control_env()
        env.update(
            {
                "HOME": str(directories["home"]),
                "GH_CONFIG_DIR": str(directories["gh"]),
                "GH_TOKEN": token,
                "GH_HOST": helper.host,
                "XDG_CONFIG_HOME": str(directories["config"]),
                "XDG_STATE_HOME": str(directories["state"]),
                "XDG_CACHE_HOME": str(directories["cache"]),
                "XDG_DATA_HOME": str(directories["data"]),
                "XDG_RUNTIME_DIR": str(directories["runtime"]),
                "TMPDIR": str(directories["tmp"]),
                "GH_PROMPT_DISABLED": "1",
                "GH_NO_UPDATE_NOTIFIER": "1",
            }
        )
        try:
            yield env
        finally:
            env.pop("GH_TOKEN", None)
            token = ""


def _provenance_path(path: str | Path) -> str:
    value = Path(path)
    root = _project_root()
    if root is not None:
        try:
            return value.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return str(value)


def _git_provenance(
    git_executable: str | None = None,
) -> tuple[str | None, bool | None]:
    root = _project_root()
    if root is None:
        return None, None
    try:
        git_path = (
            git_executable
            if git_executable is not None
            else str(_trusted_control_executable("git")["path"])
        )
        head = run(
            [git_path, "rev-parse", "--verify", "HEAD"],
            cwd=root,
            timeout=10,
            env=_git_control_env(),
        )
        status = run(
            [
                git_path,
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ],
            cwd=root,
            timeout=10,
            env=_git_control_env(),
        )
    except (OSError, RuntimeError, ValueError):
        return None, None
    commit = (
        head.stdout.strip()
        if (
            head.returncode == 0
            and not head.output_truncated
            and not head.cleanup_unconfirmed
            and head.stdout.strip()
        )
        else None
    )
    dirty = (
        bool(status.stdout)
        if (
            status.returncode == 0
            and not status.output_truncated
            and not status.cleanup_unconfirmed
        )
        else None
    )
    return commit, dirty


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cli_version(llm: str, client: LLMClient, cli_path: str) -> str | None:
    if llm == "codex_cli":
        value = getattr(client, "_version", None)
        return str(value) if value else None
    if llm != "claude_cli":
        return None
    try:
        result = run([cli_path, "--version"], timeout=30)
    except OSError:
        return None
    value = result.stdout.strip() or result.stderr.strip()
    return value if result.returncode == 0 and value else None


def _docker_control_env() -> dict[str, str]:
    """Keep daemon selection while excluding credentials and writable PATH entries."""
    docker_env = {
        key: value
        for key in (
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_CONFIG",
            "DOCKER_CERT_PATH",
            "DOCKER_TLS_VERIFY",
            "XDG_RUNTIME_DIR",
        )
        if (value := os.environ.get(key))
    }
    safe_path = sanitized_absolute_path(require_unwritable=True)
    if safe_path:
        docker_env["PATH"] = safe_path
    return docker_env


def _inspect_docker_image_id(image: str, *, docker: str) -> str:
    """Bind a mutable image reference with a previously bound Docker executable."""
    if not Path(docker).is_absolute():
        raise RuntimeError("Docker image inspection requires an absolute executable")
    result = run(
        [docker, "image", "inspect", "--format", "{{.Id}}", image],
        timeout=30,
        env=_docker_control_env(),
    )
    value = result.stdout.strip()
    if (
        result.returncode != 0
        or result.output_truncated
        or result.cleanup_unconfirmed
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        raise RuntimeError(f"Docker image {image!r} could not be bound to an immutable image ID")
    return value


def _resolve_docker_image_id(image: str, *, docker: str = "docker") -> str:
    """Resolve Docker bytes and bind a mutable image reference to its image ID."""
    identity = resolve_docker_executable(docker)
    return _inspect_docker_image_id(image, docker=identity.path)


def _probe_docker_image(
    backend: ExecutionBackend,
    *,
    image_id: str,
    workdir: Path,
) -> dict[str, Any]:
    """Prove the pinned image can run the scorer before a model call is spent."""
    if (
        getattr(backend, "name", None) != "docker"
        or getattr(backend, "image", None) != image_id
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
    ):
        raise RuntimeError("Docker image probe is not bound to the pinned scorer image")
    result = backend.run(
        [backend.python(), "-I", "-c", _DOCKER_IMAGE_PROBE_SCRIPT],
        cwd=workdir,
        timeout=60,
    )
    if result.returncode != 0 or result.output_truncated or result.cleanup_unconfirmed:
        raise RuntimeError("pinned Docker image failed its scorer capability probe")
    payloads = [
        line.removeprefix(_DOCKER_IMAGE_PROBE_MARKER)
        for line in result.stdout.splitlines()
        if line.startswith(_DOCKER_IMAGE_PROBE_MARKER)
    ]
    if len(payloads) != 1:
        raise RuntimeError("pinned Docker image returned an invalid capability receipt")
    try:
        payload = json.loads(payloads[0])
    except json.JSONDecodeError as error:
        raise RuntimeError("pinned Docker image returned an invalid capability receipt") from error
    required = {
        "python_version",
        "pytest_version",
        "pytest_json_report_version",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or any(not isinstance(payload.get(field), str) or not payload[field] for field in required)
    ):
        raise RuntimeError("pinned Docker image returned an invalid capability receipt")
    return {
        "schema_version": _DOCKER_IMAGE_PROBE_SCHEMA,
        "image_id": image_id,
        "network": "none",
        "minimal_pytest": "passed",
        **payload,
    }


def _fingerprint(
    task_path: str,
    source: Path,
    llm: str,
    model: str | None,
    scorer: str = "trusted-local",
    backend_version: str = "",
    *,
    source_files: dict[str, str] | None = None,
    runtime: dict[str, Any] | None = None,
) -> str:
    """Everything that determines a cell's outcome. Any change busts the cache."""
    sources = source_files or _source_file_digests()
    payload: dict[str, Any] = {
        "cache_schema": _CACHE_SCHEMA,
        "harness_version": __version__,
        "task_sha256": _sha256_bytes(
            _read_bounded_bytes(Path(task_path), max_bytes=_MAX_TASK_BYTES)
        ),
        "corpus_sha256": _repo_digest(source),
        "source_tree_sha256": _source_tree_digest(sources),
        "llm_backend": llm,
        "model": model,
        "max_repairs": _MAX_REPAIRS,
        "llm_retries": _LLM_RETRIES,
        "backend_version": backend_version or None,
        "scorer": scorer,
        "runtime": runtime or {},
    }
    return _canonical_digest(payload)


def _materialize_call_receipts(
    audits: list[dict[str, Any]],
    *,
    task: str,
    rep: int,
    cell_fingerprint: str,
    input_snapshot_sha256: str,
    directory: Path,
    formal_binding: dict[str, str] | None = None,
) -> list[str]:
    call_fields = {
        "status",
        "backend",
        "cli_version",
        "model",
        "reasoning_effort",
        "sandbox_mode",
        "permission_model",
        "permission_profile",
        "credential_barrier",
        "cli_executable_sha256",
        "cli_executable_trusted",
        "externally_sandboxed",
        "retries",
        "attempt_count",
        "duration_s",
        "event_summary",
        "attempts",
        "usage",
        "error_type",
        "retryable",
    }
    receipts: list[str] = []
    for ordinal, audit in enumerate(audits):
        receipt = {
            "schema_version": _LLM_CALL_RECEIPT_SCHEMA,
            "binding": {
                "task": task,
                "rep": rep,
                "label": audit.get("label"),
                "ordinal": ordinal,
                "cell_fingerprint": cell_fingerprint,
                "input_snapshot_sha256": input_snapshot_sha256,
                "formal_attempt_id": (
                    formal_binding.get("formal_attempt_id")
                    if formal_binding is not None
                    else None
                ),
                "formal_registration_registry_sha256": (
                    formal_binding.get("formal_registration_registry_sha256")
                    if formal_binding is not None
                    else None
                ),
                "formal_protocol_sha256": (
                    formal_binding.get("formal_protocol_sha256")
                    if formal_binding is not None
                    else None
                ),
                "formal_outcome_key": (
                    formal_binding.get("formal_outcome_key")
                    if formal_binding is not None
                    else None
                ),
                "prompt_sha256": audit.get("prompt_sha256"),
                "response_sha256": audit.get("response_sha256"),
                "patch_sha256": audit.get("patch_sha256"),
                "result_artifact_sha256": audit.get("result_artifact_sha256"),
            },
            "call": {name: audit[name] for name in call_fields if name in audit},
        }
        receipts.append(_store_llm_call_receipt(receipt, directory))
    return receipts


def _validate_cell_call_sequence(
    receipts: list[dict[str, Any]],
    *,
    repairs: int,
    max_outer_attempts: int,
    max_inner_attempts: int,
    terminal_error: bool = False,
) -> None:
    if not receipts or max_outer_attempts <= 0 or max_inner_attempts <= 0:
        raise ValueError("cell has no bounded LLM call sequence")
    for ordinal, receipt in enumerate(receipts):
        binding = receipt["binding"]
        call = receipt["call"]
        if binding["ordinal"] != ordinal:
            raise ValueError("cell LLM call ordinals are not contiguous")
        if call["attempt_count"] > max_inner_attempts:
            raise ValueError("cell LLM call exceeded its inner retry budget")

    first_end = next(
        (
            index
            for index, receipt in enumerate(receipts)
            if receipt["binding"]["label"] == "repair"
        ),
        len(receipts),
    )
    first_calls = receipts[:first_end]
    first_ended_in_error = terminal_error and first_end == len(receipts)
    expected_first_terminal_status = "failed" if first_ended_in_error else "succeeded"
    if (
        not first_calls
        or len(first_calls) > max_outer_attempts
        or any(receipt["binding"]["label"] != "first" for receipt in first_calls)
        or any(receipt["call"]["status"] != "failed" for receipt in first_calls[:-1])
        or first_calls[-1]["call"]["status"] != expected_first_terminal_status
    ):
        raise ValueError("cell first-call retry sequence is invalid")
    if first_ended_in_error:
        if repairs != 0:
            raise ValueError("failed first-call sequence cannot contain completed repairs")
        return

    repair_calls = receipts[first_end:]
    repair_successes = 0
    segment_length = 0
    for index, receipt in enumerate(repair_calls):
        if receipt["binding"]["label"] != "repair":
            raise ValueError("cell returned to first-call evidence after repair")
        segment_length += 1
        if segment_length > max_outer_attempts:
            raise ValueError("cell repair exceeded its outer retry budget")
        if receipt["call"]["status"] == "succeeded":
            if terminal_error and index == len(repair_calls) - 1:
                raise ValueError("terminal-error sequence ended in a successful repair")
            repair_successes += 1
            segment_length = 0
    if terminal_error:
        if not repair_calls or repair_calls[-1]["call"]["status"] != "failed":
            raise ValueError("terminal-error sequence has no terminal failed call")
    elif segment_length:
        raise ValueError("cell repair sequence ended in a failed call")
    if repair_successes != repairs:
        raise ValueError("cell repair receipt count is stale")


def _is_schema4_formal_error_sequence(receipts: list[dict[str, Any]]) -> bool:
    """Schema 4 can represent only a cell that failed before producing a patch."""
    return bool(receipts) and all(
        receipt["binding"]["label"] == "first" and receipt["call"]["status"] == "failed"
        for receipt in receipts
    )


@dataclass(frozen=True)
class _CachedCell:
    records: list[RunRecord]
    call_receipts: list[str]
    legacy_calls: list[dict[str, Any]]


def _record_from_raw(raw: dict[str, Any], *, schema_version: int) -> RunRecord:
    """Decode historical records while keeping their old truth field explicit.

    Through schema 3, ``true_success`` meant artifact correctness even when a
    condition rejected the artifact. Schema 4 stores ``artifact_correct``
    separately and reserves ``true_success`` for a correct artifact that was
    actually delivered.
    """
    values = dict(raw)
    if schema_version < 4 and "artifact_correct" not in values:
        old_truth = values.get("true_success")
        claimed = values.get("claimed_success")
        if isinstance(old_truth, bool) and isinstance(claimed, bool):
            values["artifact_correct"] = old_truth
            values["true_success"] = claimed and old_truth
    return RunRecord(**values)


def _decode_cache(data: Any) -> tuple[str | None, list[RunRecord]] | None:
    if isinstance(data, list):
        raw_records = data
        fingerprint = None
        schema_version = 1
    elif isinstance(data, dict) and isinstance(data.get("records"), list):
        raw_records = data["records"]
        value = data.get("fingerprint")
        fingerprint = value if isinstance(value, str) and value else None
        raw_schema = data.get("schema_version", 1)
        schema_version = raw_schema if type(raw_schema) is int else 1
    else:
        return None
    try:
        records = [
            _record_from_raw(record, schema_version=schema_version)
            for record in raw_records
            if isinstance(record, dict)
        ]
        if len(records) != len(raw_records):
            return None
        return fingerprint, records
    except (TypeError, KeyError):
        return None


def _read_cache(path: Path) -> tuple[str | None, list[RunRecord]] | None:
    """Decode a bounded, stable legacy or current cache file."""
    try:
        data = json.loads(_read_bounded_text(path, max_bytes=_MAX_CACHE_BYTES))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None  # corrupt/partial cache -> recompute
    return _decode_cache(data)


def _load_cached_cell(
    path: Path,
    fingerprint: str,
    *,
    input_snapshot_sha256: str | None = None,
    scorer_backend: str | None = None,
    scorer_image_id: str | None = None,
    require_call_receipts: bool = False,
    expected_task: str | None = None,
    expected_rep: int | None = None,
    receipt_dir: Path | None = None,
    max_outer_attempts: int = _LLM_RETRIES,
    max_inner_attempts: int = 1,
    formal_evidence: bool = False,
    expected_formal_binding: dict[str, str] | None = None,
) -> _CachedCell | None:
    try:
        cache_envelope = json.loads(_read_bounded_text(path, max_bytes=_MAX_CACHE_BYTES))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if (
        not isinstance(cache_envelope, dict)
        or cache_envelope.get("schema_version") != _CACHE_SCHEMA
    ):
        return None
    formal_fields = (
        "formal_attempt_id",
        "formal_registration_registry_sha256",
        "formal_protocol_sha256",
        "formal_outcome_key",
    )
    if formal_evidence:
        if (
            not isinstance(expected_formal_binding, dict)
            or set(expected_formal_binding) != set(formal_fields)
            or any(
                not isinstance(expected_formal_binding[field], str)
                or _HEX_64.fullmatch(expected_formal_binding[field]) is None
                or cache_envelope.get(field) != expected_formal_binding[field]
                for field in formal_fields
            )
        ):
            return None
    elif expected_formal_binding is not None:
        return None
    decoded = _decode_cache(cache_envelope)
    if decoded is None:
        return None
    cached_fingerprint, records = decoded
    if cached_fingerprint != fingerprint:
        return None
    terminal_error = cache_envelope.get("terminal_error")
    if type(terminal_error) is not bool:
        return None
    has_error = any(record.status == "ERROR" for record in records)
    if has_error != terminal_error:
        return None
    if terminal_error:
        expected_conditions = {name for name, _ in CONDITIONS}
        if (
            not require_call_receipts
            or len(records) != len(CONDITIONS)
            or {record.condition for record in records} != expected_conditions
            or len({record.detail for record in records}) != 1
            or len({record.repairs for record in records}) != 1
            or any(
                record.status != "ERROR"
                or record.task != expected_task
                or record.rep != expected_rep
                or record.scorer_outcome != ScoreOutcome.INFRA_ERROR.value
                or record.claimed_success
                or record.artifact_correct
                or record.true_success
                or record.false_success
                or record.gate_prediction is not None
                or record.artifact_sha256
                or record.scorer_evidence_sha256
                or record.scorer_expected_tests != 0
                or record.scorer_passed_tests != 0
                for record in records
            )
        ):
            return None
    if any(
        record.true_success != (record.claimed_success and record.artifact_correct)
        or record.false_success != (record.claimed_success and not record.artifact_correct)
        for record in records
    ):
        return None
    if (
        not isinstance(input_snapshot_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", input_snapshot_sha256) is None
        or not isinstance(scorer_backend, str)
        or not scorer_backend
    ):
        return None
    if not terminal_error:
        artifact_dir = path.parent.parent / "artifacts"
        if any(
            not _artifact_file_is_valid(
                artifact_dir / f"{record.artifact_sha256}.json",
                record.artifact_sha256,
            )
            for record in records
        ):
            return None
        evidence_dir = path.parent.parent / "scorer_evidence"
        if any(
            not _scorer_evidence_file_is_valid(
                evidence_dir / f"{record.scorer_evidence_sha256}.json",
                record.scorer_evidence_sha256,
                record,
                ScorerEvidenceBinding(
                    task=record.task,
                    rep=record.rep,
                    artifact_sha256=record.artifact_sha256,
                    input_snapshot_sha256=input_snapshot_sha256,
                    scorer_backend=scorer_backend,
                    scorer_image_id=scorer_image_id,
                ),
            )
            for record in records
        ):
            return None
    calls = cache_envelope.get("llm_calls")
    legacy_calls = (
        [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
    )
    if isinstance(calls, list) and len(legacy_calls) != len(calls):
        return None
    receipt_values = cache_envelope.get("llm_call_receipts")
    call_receipts = (
        list(receipt_values)
        if isinstance(receipt_values, list)
        and all(isinstance(value, str) for value in receipt_values)
        else []
    )
    if require_call_receipts:
        if (
            not call_receipts
            or len(call_receipts) != len(set(call_receipts))
            or receipt_dir is None
            or receipt_dir.is_symlink()
            or not receipt_dir.is_dir()
            or not isinstance(expected_task, str)
            or not expected_task
            or not _nonnegative_int(expected_rep)
        ):
            return None
        decoded_receipts: list[dict[str, Any]] = []
        try:
            for ordinal, digest in enumerate(call_receipts):
                decoded_receipts.append(
                    _read_llm_call_receipt(
                        receipt_dir / f"{digest}.json",
                        digest,
                        expected_binding={
                            "task": expected_task,
                            "rep": expected_rep,
                            "ordinal": ordinal,
                            "cell_fingerprint": fingerprint,
                            "input_snapshot_sha256": input_snapshot_sha256,
                            **{
                                field: (
                                    expected_formal_binding[field]
                                    if expected_formal_binding is not None
                                    else None
                                )
                                for field in formal_fields
                            },
                        },
                    )
                )
            verify = next(record for record in records if record.condition == "verify")
            _validate_cell_call_sequence(
                decoded_receipts,
                repairs=verify.repairs,
                max_outer_attempts=max_outer_attempts,
                max_inner_attempts=max_inner_attempts,
                terminal_error=terminal_error,
            )
            if (
                terminal_error
                and formal_evidence
                and not _is_schema4_formal_error_sequence(decoded_receipts)
            ):
                raise RuntimeError(
                    "formal schema-4 ERROR cache contains a successful or repair call"
                )
        except (OSError, StopIteration, TypeError, ValueError):
            return None
    return _CachedCell(
        records=records,
        call_receipts=call_receipts,
        legacy_calls=legacy_calls,
    )


def _load_cached(
    path: Path,
    fingerprint: str,
    *,
    input_snapshot_sha256: str | None = None,
    scorer_backend: str | None = None,
    scorer_image_id: str | None = None,
) -> list[RunRecord] | None:
    cached = _load_cached_cell(
        path,
        fingerprint,
        input_snapshot_sha256=input_snapshot_sha256,
        scorer_backend=scorer_backend,
        scorer_image_id=scorer_image_id,
    )
    return cached.records if cached is not None else None


def _run_cell(
    llm: LLMClient,
    source: Path,
    task: TaskSpec,
    rep: int,
    out_dir: Path,
    fingerprint: str,
    source_sha256: str,
    input_snapshot_sha256: str,
    agent_exec: ExecutionBackend,
    scorer: ExecutionBackend,
    scorer_backend: str,
    scorer_image_id: str | None,
    audit_log: list[dict[str, Any]] | None = None,
    require_call_receipts: bool = False,
    max_inner_attempts: int = 1,
    formal_evidence: bool = False,
    *,
    formal_output_lease: _FormalOutputLease | None = None,
    formal_run_binding: _FormalRunBinding | None = None,
) -> list[RunRecord]:
    name = task.inputs.get("_name", task.title)
    cache = out_dir / "results" / f"{name}__r{rep}.json"
    attempt_marker = out_dir / "results" / f"{name}__r{rep}.started.json"
    receipt_dir = out_dir / "llm_call_receipts"
    cell_audits: list[dict[str, Any]] = []
    cell_receipts: list[str] = []
    if formal_output_lease is not None and formal_run_binding is None:
        raise RuntimeError("formal ablation cell has no sealed run header")
    formal_cell_fields = (
        formal_run_binding.cell_fields()
        if formal_run_binding is not None
        else {}
    )

    def error_records(detail: str, *, repairs: int = 0) -> list[RunRecord]:
        return [
            RunRecord(
                task=name,
                condition=condition,
                rep=rep,
                status="ERROR",
                claimed_success=False,
                artifact_correct=False,
                true_success=False,
                false_success=False,
                repairs=repairs,
                detail=detail[:200],
                scorer_outcome=ScoreOutcome.INFRA_ERROR.value,
            )
            for condition, _ in CONDITIONS
        ]

    def append_call_audits(receipts: list[str], *, cache_hit: bool) -> None:
        if audit_log is None:
            return
        if require_call_receipts:
            audit_log.extend(
                {
                    "task": name,
                    "rep": rep,
                    "ordinal": ordinal,
                    "receipt_sha256": digest,
                    "cache_hit": cache_hit,
                }
                for ordinal, digest in enumerate(receipts)
            )
        else:
            audit_log.extend(
                {"task": name, "rep": rep, "cache_hit": cache_hit, **call}
                for call in cell_audits
            )

    def persist_receipts() -> tuple[list[str], list[dict[str, Any]]]:
        receipts = _materialize_call_receipts(
            cell_audits,
            task=name,
            rep=rep,
            cell_fingerprint=fingerprint,
            input_snapshot_sha256=input_snapshot_sha256,
            directory=receipt_dir,
            formal_binding=(
                formal_cell_fields if formal_run_binding is not None else None
            ),
        )
        decoded_receipts = [
            _read_llm_call_receipt(
                receipt_dir / f"{digest}.json",
                digest,
                expected_binding={
                    "task": name,
                    "rep": rep,
                    "ordinal": ordinal,
                    "cell_fingerprint": fingerprint,
                    "input_snapshot_sha256": input_snapshot_sha256,
                    "formal_attempt_id": formal_cell_fields.get(
                        "formal_attempt_id"
                    ),
                    "formal_registration_registry_sha256": (
                        formal_cell_fields.get(
                            "formal_registration_registry_sha256"
                        )
                    ),
                    "formal_protocol_sha256": formal_cell_fields.get(
                        "formal_protocol_sha256"
                    ),
                    "formal_outcome_key": formal_cell_fields.get(
                        "formal_outcome_key"
                    ),
                },
            )
            for ordinal, digest in enumerate(receipts)
        ]
        return receipts, decoded_receipts

    def write_cache(records: list[RunRecord], *, terminal_error: bool) -> None:
        payload = json.dumps(
            {
                "schema_version": _CACHE_SCHEMA,
                "fingerprint": fingerprint,
                "terminal_error": terminal_error,
                "records": [asdict(record) for record in records],
                **formal_cell_fields,
                **(
                    {"llm_call_receipts": cell_receipts}
                    if require_call_receipts
                    else {"llm_calls": cell_audits}
                ),
            },
            indent=2,
        )
        if formal_evidence:
            _write_formal_cell_start(
                cache,
                payload.encode("utf-8"),
                out_dir=out_dir,
                lease=formal_output_lease,
                label="terminal cell seal",
            )
        else:
            _atomic_write(cache, payload)

    def snapshot_matches() -> bool:
        try:
            return _repo_digest(source) == source_sha256
        except (OSError, RuntimeError):
            return False

    marker_payload = _canonical_json_object_bytes(
        {
            "schema_version": _CELL_ATTEMPT_SCHEMA,
            "task": name,
            "rep": rep,
            "cell_fingerprint": fingerprint,
            "input_snapshot_sha256": input_snapshot_sha256,
            **formal_cell_fields,
        }
    )
    marker_exists = attempt_marker.exists() or attempt_marker.is_symlink()
    cache_exists = cache.exists() or cache.is_symlink()
    if formal_evidence:
        if marker_exists or cache_exists:
            raise RuntimeError(
                "formal ablation does not resume or reuse cells; mark the "
                "registered attempt ABANDONED and use a new attempt"
            )

    if not snapshot_matches():
        if formal_evidence:
            raise RuntimeError("formal ablation input snapshot failed validation")
        return error_records("content-addressed input snapshot failed validation")
    cached = (
        None
        if formal_evidence
        else _load_cached_cell(
            cache,
            fingerprint,
            input_snapshot_sha256=input_snapshot_sha256,
            scorer_backend=scorer_backend,
            scorer_image_id=scorer_image_id,
            require_call_receipts=require_call_receipts,
            expected_task=name,
            expected_rep=rep,
            receipt_dir=receipt_dir,
            max_outer_attempts=_LLM_RETRIES,
            max_inner_attempts=max_inner_attempts,
            formal_evidence=False,
        )
    )
    if cached is not None:
        if require_call_receipts:
            cell_receipts = cached.call_receipts
            append_call_audits(cell_receipts, cache_hit=True)
        elif audit_log is not None:
            audit_log.extend(
                {"task": name, "rep": rep, "cache_hit": True, **call}
                for call in cached.legacy_calls
            )
        return cached.records
    if formal_evidence:
        _write_formal_cell_start(
            attempt_marker,
            marker_payload,
            out_dir=out_dir,
            lease=formal_output_lease,
        )
    with tempfile.TemporaryDirectory(prefix="lha_abl_") as tmp:
        scratch = Path(tmp)
        try:
            patch = _first_attempt(llm, source, task, scratch, cell_audits)
            records = _evaluate(
                llm,
                source,
                task,
                patch,
                scratch,
                rep,
                agent_exec,
                scorer,
                out_dir / "artifacts",
                input_snapshot_sha256,
                scorer_backend,
                scorer_image_id,
                cell_audits,
            )
            if require_call_receipts:
                cell_receipts, decoded_receipts = persist_receipts()
                verify = next(record for record in records if record.condition == "verify")
                _validate_cell_call_sequence(
                    decoded_receipts,
                    repairs=verify.repairs,
                    max_outer_attempts=_LLM_RETRIES,
                    max_inner_attempts=max_inner_attempts,
                )
        except _Transient as e:
            logger.error("transient failure on %s rep %d: %s", name, rep, e)
            if require_call_receipts:
                if not cell_audits:
                    if formal_evidence:
                        raise RuntimeError(
                            "formal ablation cell failed before an auditable LLM call"
                        ) from e
                else:
                    cell_receipts, decoded_receipts = persist_receipts()
                    repairs = sum(
                        receipt["binding"]["label"] == "repair"
                        and receipt["call"]["status"] == "succeeded"
                        for receipt in decoded_receipts
                    )
                    _validate_cell_call_sequence(
                        decoded_receipts,
                        repairs=repairs,
                        max_outer_attempts=_LLM_RETRIES,
                        max_inner_attempts=max_inner_attempts,
                        terminal_error=True,
                    )
                    if formal_evidence and not _is_schema4_formal_error_sequence(
                        decoded_receipts
                    ):
                        raise RuntimeError(
                            "formal schema-4 ERROR cannot contain a successful or repair call"
                        ) from e
                    records = error_records(
                        f"transient cell failure: {type(e).__name__}",
                        repairs=repairs,
                    )
                    if not snapshot_matches():
                        if formal_evidence:
                            raise RuntimeError(
                                "formal ablation input changed before sealing an ERROR cell"
                            ) from e
                        append_call_audits(cell_receipts, cache_hit=False)
                        return records
                    try:
                        write_cache(records, terminal_error=True)
                    except Exception as cache_error:
                        if formal_evidence:
                            raise RuntimeError(
                                "formal ablation could not seal an ERROR cell"
                            ) from cache_error
                        logger.error(
                            "ERROR cell cache write failure on %s rep %d",
                            name,
                            rep,
                            exc_info=True,
                        )
                        append_call_audits(cell_receipts, cache_hit=False)
                        return records
                    append_call_audits(cell_receipts, cache_hit=False)
                    return records
            append_call_audits([], cache_hit=False)
            return error_records(f"transient cell failure: {type(e).__name__}")
        except Exception as error:
            # A malformed patch, filesystem error, or corrupt evidence store is
            # a missing cell measurement. Keep it in the denominator and let
            # later cells run. BaseException subclasses still interrupt.
            logger.error(
                "infrastructure failure on %s rep %d: %s",
                name,
                rep,
                type(error).__name__,
                exc_info=True,
            )
            if require_call_receipts:
                if not cell_audits:
                    if formal_evidence:
                        raise RuntimeError(
                            "formal ablation cell failed before an auditable LLM call"
                        ) from error
                else:
                    cell_receipts, decoded_receipts = persist_receipts()
                    repairs = sum(
                        receipt["binding"]["label"] == "repair"
                        and receipt["call"]["status"] == "succeeded"
                        for receipt in decoded_receipts
                    )
                    try:
                        _validate_cell_call_sequence(
                            decoded_receipts,
                            repairs=repairs,
                            max_outer_attempts=_LLM_RETRIES,
                            max_inner_attempts=max_inner_attempts,
                            terminal_error=True,
                        )
                    except ValueError as sequence_error:
                        if formal_evidence:
                            raise RuntimeError(
                                "formal ablation failure did not end in a failed LLM call"
                            ) from sequence_error
                        append_call_audits(cell_receipts, cache_hit=False)
                    else:
                        if formal_evidence and not _is_schema4_formal_error_sequence(
                            decoded_receipts
                        ):
                            raise RuntimeError(
                                "formal schema-4 ERROR cannot contain a successful or repair call"
                            ) from error
                        records = error_records(
                            f"cell infrastructure failure: {type(error).__name__}",
                            repairs=repairs,
                        )
                        if snapshot_matches():
                            try:
                                write_cache(records, terminal_error=True)
                            except Exception as cache_error:
                                if formal_evidence:
                                    raise RuntimeError(
                                        "formal ablation could not seal an ERROR cell"
                                    ) from cache_error
                                logger.error(
                                    "ERROR cell cache write failure on %s rep %d",
                                    name,
                                    rep,
                                    exc_info=True,
                                )
                                append_call_audits(cell_receipts, cache_hit=False)
                                return records
                        elif formal_evidence:
                            raise RuntimeError(
                                "formal ablation input changed before sealing an ERROR cell"
                            ) from error
                        else:
                            append_call_audits(cell_receipts, cache_hit=False)
                            return records
                        append_call_audits(cell_receipts, cache_hit=False)
                        return records
            else:
                append_call_audits([], cache_hit=False)
            return error_records(f"cell infrastructure failure: {type(error).__name__}")
    if any(record.status == "ERROR" for record in records):
        if formal_evidence:
            raise RuntimeError(
                "formal ablation evaluator returned infrastructure ERROR without "
                "a terminal failed LLM call"
            )
        append_call_audits(cell_receipts, cache_hit=False)
        return records
    if not snapshot_matches():
        if formal_evidence:
            raise RuntimeError("formal ablation input changed during the cell")
        append_call_audits(cell_receipts, cache_hit=False)
        return error_records("content-addressed input snapshot changed during the cell")
    try:
        write_cache(records, terminal_error=False)
    except Exception as error:
        logger.error(
            "cache write failure on %s rep %d: %s",
            name,
            rep,
            type(error).__name__,
            exc_info=True,
        )
        if formal_evidence:
            raise RuntimeError("formal ablation could not seal a completed cell") from error
        append_call_audits(cell_receipts, cache_hit=False)
        return error_records(f"cell cache write failure: {type(error).__name__}")
    append_call_audits(cell_receipts, cache_hit=False)
    return records


# --- aggregation ---------------------------------------------------------------
def _rate_ci(
    records: list[RunRecord], metric: str, *, n: int = _BOOTSTRAP_N, seed: int = 0
) -> tuple[float, float] | None:
    """95% interval for one reported rate.

    Interior rates use the task-cluster bootstrap because repetitions are
    nested within a task. At an all-zero or all-one boundary, that bootstrap
    collapses to a zero-width interval, so use the Wilson score interval over
    the observed cells instead of presenting false certainty.
    """
    successes = sum(bool(getattr(record, metric)) for record in records)
    total = len(records)
    if total and successes in (0, total):
        from .bench.stats import wilson_interval

        return wilson_interval(successes, total)

    by_task: dict[str, list[float]] = {}
    for r in records:
        by_task.setdefault(r.task, []).append(float(getattr(r, metric)))
    if len(by_task) < 2:
        return None  # a single task cannot express between-task variation
    from .bench.stats import cluster_bootstrap_ci

    return cluster_bootstrap_ci(by_task, n=n, seed=seed)


def _paired_mcnemar_lines(
    records: list[RunRecord],
    *,
    precise: bool = False,
) -> list[str]:
    """Exact McNemar on the paired (task, rep) cells for the headline contrasts."""
    from .bench.stats import mcnemar_exact

    def outcomes(cond: str, metric: str) -> dict[tuple[str, int], bool]:
        return {
            (r.task, r.rep): bool(getattr(r, metric))
            for r in records
            if r.condition == cond and r.status != "ERROR"
        }

    lines: list[str] = []
    for a, b, metric in (
        ("trust", "gate", "false_success"),
        ("gate", "verify", "true_success"),
    ):
        oa, ob = outcomes(a, metric), outcomes(b, metric)
        pairs = sorted(set(oa) & set(ob))
        if not pairs:
            continue
        only_a = sum(oa[k] and not ob[k] for k in pairs)
        only_b = sum(ob[k] and not oa[k] for k in pairs)
        p = mcnemar_exact(only_a, only_b)
        p_text = f"{p:.4e}" if precise and 0 < p < 0.0001 else f"{p:.{4 if precise else 2}f}"
        lines.append(
            f"- `{a}` vs `{b}` on {metric.replace('_', ' ')}: "
            f"discordant {only_a}/{only_b} of {len(pairs)} pairs · "
            f"exact McNemar p = {p_text}"
        )
    return lines


def _aggregate(records: list[RunRecord]) -> list[ConditionStats]:
    stats: list[ConditionStats] = []
    for name, blurb in CONDITIONS:
        all_recs = [r for r in records if r.condition == name]
        recs = [r for r in all_recs if r.status != "ERROR"]
        errors = len(all_recs) - len(recs)
        n = len(recs)
        if n == 0:
            stats.append(
                ConditionStats(
                    condition=name,
                    blurb=blurb,
                    n=0,
                    claimed_success_rate=0.0,
                    artifact_correct_rate=0.0,
                    true_success_rate=0.0,
                    false_success_rate=0.0,
                    mean_repairs=0.0,
                    errors=errors,
                )
            )
            continue
        s = ConditionStats(
            condition=name,
            blurb=blurb,
            n=n,
            claimed_success_rate=sum(r.claimed_success for r in recs) / n,
            artifact_correct_rate=sum(r.artifact_correct for r in recs) / n,
            true_success_rate=sum(r.true_success for r in recs) / n,
            false_success_rate=sum(r.false_success for r in recs) / n,
            mean_repairs=sum(r.repairs for r in recs) / n,
            errors=errors,
            artifact_ci=_rate_ci(recs, "artifact_correct"),
            true_ci=_rate_ci(recs, "true_success"),
            false_ci=_rate_ci(recs, "false_success"),
        )
        preds = [r for r in recs if r.gate_prediction is not None]
        if preds:
            tp = sum(bool(r.gate_prediction) and r.artifact_correct for r in preds)
            fp = sum(bool(r.gate_prediction) and not r.artifact_correct for r in preds)
            tn = sum(not r.gate_prediction and not r.artifact_correct for r in preds)
            fn = sum(not r.gate_prediction and r.artifact_correct for r in preds)
            s.tp, s.fp, s.tn, s.fn = tp, fp, tn, fn
            s.precision = tp / (tp + fp) if (tp + fp) else None
            s.recall = tp / (tp + fn) if (tp + fn) else None
            s.fpr = fp / (fp + tn) if (fp + tn) else None
            s.fnr = fn / (fn + tp) if (fn + tp) else None
        stats.append(s)
    return stats


def _client_runtime(
    llm: str,
    client: LLMClient,
    *,
    model: str | None,
    cli_path: str,
    backend_details: str,
) -> dict[str, Any]:
    actual_model = getattr(client, "model", model)
    if actual_model is not None and not isinstance(actual_model, str):
        actual_model = str(actual_model)
    reasoning_effort = getattr(client, "reasoning_effort", getattr(client, "effort", None))
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        reasoning_effort = str(reasoning_effort)
    safe_configuration: dict[str, Any] = {}
    for name in (
        "timeout",
        "no_tools",
        "sandbox_mode",
        "permission_model",
        "permission_profile",
        "credential_barrier",
        "externally_sandboxed",
        "max_retries",
        "retry_backoff_s",
        "max_tokens",
    ):
        value = getattr(client, name, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_configuration[name] = value
    try:
        client_source = inspect.getsource(type(client)).encode()
    except (OSError, TypeError):
        client_source_sha256 = None
    else:
        client_source_sha256 = _sha256_bytes(client_source)
    return {
        "requested_backend": llm,
        "actual_backend": getattr(client, "name", type(client).__name__) or "unknown",
        "client_type": f"{type(client).__module__}.{type(client).__qualname__}",
        "client_source_sha256": client_source_sha256,
        "model": actual_model,
        "cli_version": _cli_version(llm, client, cli_path),
        "backend_library_version": (_package_version("anthropic") if llm == "anthropic" else None),
        "reasoning_effort": reasoning_effort,
        "backend_details": backend_details or None,
        "configuration": safe_configuration,
    }


def _bind_client_operation_lease(
    client: LLMClient,
    run_dir: Path,
) -> bool:
    """Bind the first lease-aware client in a wrapper chain to this run."""
    current: Any = client
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        setter = getattr(current, "set_operation_lease_dir", None)
        if callable(setter):
            setter(run_dir)
            return True
        current = getattr(current, "inner", None)
    return False


def _execution_runtime(
    backend: ExecutionBackend,
    *,
    requested: str,
    requested_image: str | None = None,
    pinned_image_id: str | None = None,
    control_executable: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image = getattr(backend, "image", None)
    return {
        "requested": requested,
        "actual": getattr(backend, "name", type(backend).__name__) or "unknown",
        "implementation": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "image": requested_image
        if isinstance(requested_image, str) and requested_image
        else (image if isinstance(image, str) and image else None),
        "execution_image": image if isinstance(image, str) and image else None,
        "image_id": pinned_image_id,
        "control_executable": dict(control_executable or {}),
    }


def _operation_lease_directories_are_empty(run_dir: Path) -> None:
    """Reject any lease or cidfile residue, including ignored temporary entries."""
    for name in ("active-operations", "active-container-ids"):
        directory = run_dir / name
        if not directory.exists() and not directory.is_symlink():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"Docker operation store is unsafe: {directory}")
        if any(directory.iterdir()):
            raise RuntimeError(f"Docker operation store still contains entries: {directory}")


def _assert_no_lha_containers(docker: str) -> None:
    """Confirm the daemon has no container carrying LHA's ownership label."""
    if not Path(docker).is_absolute():
        raise RuntimeError("Docker container audit requires an absolute executable")
    result = run(
        [
            docker,
            "container",
            "ls",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            "label=lha.operation_id",
        ],
        timeout=30,
        env=_docker_control_env(),
    )
    if result.returncode != 0 or result.output_truncated or result.cleanup_unconfirmed:
        raise RuntimeError("Docker daemon could not confirm absence of LHA-owned containers")
    if result.stdout.strip():
        raise RuntimeError("Docker daemon still has LHA-owned containers")


def _recover_docker_operations(
    backend: DockerBackend,
    run_dir: Path,
    *,
    allow_recovered: bool,
) -> int:
    """Reap crash residue and prove the formal output owns no live container."""
    recovery = backend.recover_active_operations(run_dir)
    if not recovery.confirmed:
        raise RuntimeError(
            "Docker operation recovery could not be confirmed"
            + (f": {recovery.detail}" if recovery.detail else "")
        )
    _operation_lease_directories_are_empty(run_dir)
    _assert_no_lha_containers(backend.docker)
    recovered = len(recovery.recovered_operation_ids)
    if recovered and not allow_recovered:
        raise RuntimeError("Docker operations remained active after the formal ablation")
    return recovered


def _read_condition_stats(raw: dict[str, Any]) -> ConditionStats:
    values = dict(raw)
    if "artifact_correct_rate" not in values:
        # Historical reports used true_success_rate for artifact correctness.
        values["artifact_correct_rate"] = values.get("true_success_rate", 0.0)
    for name in ("artifact_ci", "true_ci", "false_ci"):
        interval = values.get(name)
        if isinstance(interval, list) and len(interval) == 2:
            values[name] = (float(interval[0]), float(interval[1]))
    return ConditionStats(**values)


def _ablation_report_from_raw(raw: dict[str, Any]) -> AblationReport:
    raw_schema = raw.get("schema_version", 1)
    schema_version = raw_schema if type(raw_schema) is int else 1
    provenance_raw = raw.get("provenance")
    provenance = AblationProvenance(**provenance_raw) if isinstance(provenance_raw, dict) else None
    return AblationReport(
        llm=str(raw.get("llm", "unknown")),
        model=str(raw.get("model", "")),
        reps=int(raw.get("reps", 0)),
        tasks=[str(task) for task in raw.get("tasks", [])],
        records=[
            _record_from_raw(record, schema_version=schema_version)
            for record in raw.get("records", [])
            if isinstance(record, dict)
        ],
        stats=[_read_condition_stats(stat) for stat in raw.get("stats", [])],
        scorer=str(raw.get("scorer", "unknown")),
        fingerprint=str(raw.get("fingerprint", "")),
        backend_version=str(raw.get("backend_version", "")),
        schema_version=schema_version,
        provenance=provenance,
        llm_calls=[call for call in raw.get("llm_calls", []) if isinstance(call, dict)],
    )


def load_ablation_report(path: str | Path) -> AblationReport:
    """Read a bounded, stable historical or current ablation report."""
    raw = json.loads(_read_bounded_text(Path(path), max_bytes=_MAX_REPORT_BYTES))
    if not isinstance(raw, dict):
        raise ValueError("ablation report must contain a JSON object")
    return _ablation_report_from_raw(raw)


def run_ablation(
    base: Config,
    task_paths: list[str],
    *,
    llm: str = "codex_cli",
    model: str | None = None,
    reps: int = 1,
    out_dir: Path | None = None,
    llm_client: LLMClient | None = None,
    scorer_backend: str = "trusted-local",
) -> AblationReport:
    if reps <= 0:
        raise ValueError("reps must be greater than zero")
    out = Path(out_dir) if out_dir else (Path(base.runs_dir) / "ablation")
    formal_corpus = _prepare_formal_corpus_binding(
        task_paths,
        repetitions=reps,
    )
    if formal_corpus is not None:
        repository_root = _project_root()
        if repository_root is None:
            raise RuntimeError("formal ablation requires a Git project checkout")
        # Corpus binding is read-only preregistration work. The output lifecycle
        # begins under a repository lock that cannot be replaced with runs/.
        # The output lock is always second, giving terminal commands the same
        # global -> output ordering.
        with formal_attempt_lock(repository_root):
            formal_corpus = _prepare_formal_corpus_binding(
                task_paths,
                repetitions=reps,
            )
            if formal_corpus is None:
                raise RuntimeError("formal ablation corpus binding is unavailable")
            with _formal_ablation_lock(out) as formal_output_lease:
                return _run_ablation_with_binding(
                    base,
                    task_paths,
                    llm=llm,
                    model=model,
                    reps=reps,
                    out_dir=out,
                    llm_client=llm_client,
                    scorer_backend=scorer_backend,
                    formal_corpus=formal_corpus,
                    formal_output_lease=formal_output_lease,
                )
    return _run_ablation_with_binding(
        base,
        task_paths,
        llm=llm,
        model=model,
        reps=reps,
        out_dir=out,
        llm_client=llm_client,
        scorer_backend=scorer_backend,
        formal_corpus=None,
        formal_output_lease=None,
    )


def _run_ablation_with_binding(
    base: Config,
    task_paths: list[str],
    *,
    llm: str,
    model: str | None,
    reps: int,
    out_dir: Path,
    llm_client: LLMClient | None,
    scorer_backend: str,
    formal_corpus: _FormalCorpusBinding | None,
    formal_output_lease: _FormalOutputLease | None,
) -> AblationReport:
    if formal_corpus is not None and llm_client is not None:
        raise ValueError(
            "formal ablation does not accept an injected LLM client; "
            "the built-in Codex CLI backend is required"
        )
    client_probe: Any = llm_client
    seen_clients: set[int] = set()
    rejects_formal_evidence = False
    while client_probe is not None and id(client_probe) not in seen_clients:
        seen_clients.add(id(client_probe))
        if getattr(client_probe, "formal_evidence_supported", None) is False:
            rejects_formal_evidence = True
            break
        client_probe = getattr(client_probe, "inner", None)
    if llm == "claude_cli" or rejects_formal_evidence:
        raise ValueError(
            "claude_cli is experimental and cannot produce ablation evidence; "
            "use codex_cli with protocol validation"
        )
    out = Path(out_dir)
    report_path = out / "ablation_report.json"
    if formal_corpus is not None and (report_path.exists() or report_path.is_symlink()):
        raise RuntimeError(
            "formal ablation output already contains ablation_report.json; "
            "reports are immutable, so use a new output directory"
        )
    if formal_corpus is not None:
        if formal_output_lease is None:
            raise RuntimeError("formal ablation output lock is not held")
    else:
        out.mkdir(parents=True, exist_ok=True)
    # The backend's own env vars apply here exactly as in `lha run`; an explicit
    # --model wins. The resolved name feeds the provenance fingerprint.
    if llm == "codex_cli":
        model = model or (base.codex_model or None)
        cli_path, effort = base.codex_cli_path, base.codex_reasoning_effort
    else:
        model = model or (base.claude_cli_model or None)
        cli_path, effort = base.claude_cli_path, "medium"
    if formal_corpus is not None and (
        llm != "codex_cli" or scorer_backend != "docker" or not isinstance(model, str) or not model
    ):
        raise ValueError(
            "formal ablation requires Codex CLI, an explicit model, and Docker scoring"
        )
    requested_docker_image: str | None = None
    pinned_scorer_image_id: str | None = None
    docker_executable: dict[str, Any] = {}
    formal_attempt_binding: _FormalAttemptBinding | None = None
    formal_run_binding: _FormalRunBinding | None = None
    registered_source_tree_sha256: str | None = None
    formal_preflight_done = False
    formal_cli_executable_sha256: str | None = None
    formal_codex_client: FormalCodexClientConfig | None = None
    if scorer_backend == "docker":
        requested_docker_image = base.exec_image
        docker_identity = resolve_docker_executable()
        docker_executable = docker_identity.as_provenance()
        # Resolve the mutable tag before constructing the model client or
        # running any cell. Both prediction and truth execute the same immutable
        # image bytes even if the tag moves during a long experiment.
        pinned_scorer_image_id = _inspect_docker_image_id(
            requested_docker_image,
            docker=docker_identity.path,
        )
    client = llm_client or (
        make_formal_codex_client(
            cli_path=cli_path,
            model=model,
            reasoning_effort=effort,
        )
        if formal_corpus is not None
        and llm == "codex_cli"
        and isinstance(model, str)
        and model
        else _make_llm(llm, model, cli_path=cli_path, effort=effort)
    )
    if formal_corpus is not None:
        from .llm.codex_cli import CodexCLIClient

        if (
            formal_output_lease is None
            or pinned_scorer_image_id is None
            or not isinstance(model, str)
            or not model
        ):
            raise RuntimeError("formal ablation attempt registration inputs are incomplete")
        if (
            type(client) is not CodexCLIClient
            or client.name != "codex_cli"
            or client.no_tools is not True
            or client.model != model
            or client.reasoning_effort != effort
            or client.cli_path != cli_path
            or client.sandbox_mode != "read-only"
            or client.externally_sandboxed is not False
        ):
            raise RuntimeError(
                "formal ablation did not construct the required Codex CLI client"
            )
        client.preflight()
        formal_preflight_done = True
        identity = client._cli_identity
        cli_version = client._version
        if (
            not isinstance(identity, tuple)
            or len(identity) != 7
            or not isinstance(identity[5], str)
            or _HEX_64.fullmatch(identity[5]) is None
            or not isinstance(cli_version, str)
            or not cli_version
            or cli_version == "unknown"
        ):
            raise RuntimeError(
                "formal ablation Codex preflight did not resolve a stable CLI identity"
            )
        if (
            client.permission_model != "profile"
            or client.permission_profile != "lha-read"
            or client.credential_barrier != "verified"
        ):
            raise RuntimeError(
                "formal ablation Codex preflight did not verify the fixed "
                "permission boundary"
            )
        formal_cli_executable_sha256 = identity[5]
        formal_codex_client = formal_codex_client_config_from_runtime(client)
        registered_source_tree_sha256 = _source_tree_digest(
            _revalidate_formal_checkout(formal_corpus)
        )
        formal_attempt_binding = _bind_formal_attempt(
            formal_corpus=formal_corpus,
            formal_output_lease=formal_output_lease,
            model=model,
            reasoning_effort=effort,
            docker_image_id=pinned_scorer_image_id,
            source_tree_sha256=registered_source_tree_sha256,
            codex_cli_version=cli_version,
            codex_cli_executable_sha256=formal_cli_executable_sha256,
            codex_client=formal_codex_client,
        )
        assert formal_attempt_binding is not None
        assert formal_output_lease is not None
        formal_run_binding = _initialize_formal_run(
            formal_attempt_binding,
            formal_output_lease,
        )
    codex_operation_lease_bound = _bind_client_operation_lease(client, out)
    if formal_corpus is not None and not codex_operation_lease_bound:
        raise RuntimeError("formal ablation requires Codex processes to use the output lease store")
    # Prediction and truth use separate backend instances, but the same
    # isolation class. Selecting Docker must never execute model-influenced code
    # in a host-side gate before the container scorer runs.
    docker_operations_recovered_before_run = 0
    if scorer_backend == "docker":
        if pinned_scorer_image_id is None:  # defensive; resolution above is fail-closed
            raise RuntimeError("Docker scorer image was not pinned")
        docker_path = docker_executable.get("path")
        if not isinstance(docker_path, str) or not Path(docker_path).is_absolute():
            raise RuntimeError("Docker executable identity is incomplete")
        agent_exec = make_backend(
            "docker",
            image=pinned_scorer_image_id,
            docker=docker_path,
            operation_lease_dir=out,
        )
        scorer = make_backend(
            "docker",
            image=pinned_scorer_image_id,
            docker=docker_path,
            operation_lease_dir=out,
        )
        if any(
            getattr(backend, "name", None) != "docker"
            or getattr(backend, "image", None) != pinned_scorer_image_id
            for backend in (agent_exec, scorer)
        ):
            raise RuntimeError("Docker execution backends did not retain the pinned image ID")
        for backend in (agent_exec, scorer):
            bind_control_plane = getattr(backend, "bind_control_plane", None)
            if not callable(bind_control_plane):
                if formal_corpus is not None:
                    raise RuntimeError(
                        "formal ablation Docker backend cannot bind its control executable"
                    )
                continue
            if bind_control_plane(verify_digest=True) != docker_executable:
                raise RuntimeError("Docker execution backend disagrees with the bound executable")
        if formal_corpus is not None:
            if not isinstance(agent_exec, DockerBackend) or not isinstance(
                scorer,
                DockerBackend,
            ):
                raise RuntimeError("formal ablation requires the built-in Docker backend")
            if (
                agent_exec.operation_lease_dir != out.resolve()
                or scorer.operation_lease_dir != out.resolve()
            ):
                raise RuntimeError("formal Docker backends do not share the output lease store")
            docker_operations_recovered_before_run = _recover_docker_operations(
                agent_exec,
                out,
                allow_recovered=False,
            )
        agent_requested = "docker"
    else:
        agent_exec = TrustedLocalBackend()
        scorer = make_backend(scorer_backend)
        agent_requested = "trusted-local"
    docker_image_probe: dict[str, Any] | None = None
    if scorer_backend == "docker":
        if pinned_scorer_image_id is None:
            raise RuntimeError("Docker scorer image was not pinned")
        docker_image_probe = _probe_docker_image(
            agent_exec,
            image_id=pinned_scorer_image_id,
            workdir=out,
        )
    preflight = getattr(client, "preflight", None)
    if callable(preflight) and not formal_preflight_done:
        # The shared operation store is recovered first. Preflight then proves
        # CLI setup and credential cleanup without spending a model call.
        preflight()
    require_call_receipts = getattr(client, "name", type(client).__name__) == "codex_cli"
    measurement_client: LLMClient = _PromptAuditClient(client) if require_call_receipts else client
    configured_inner_retries = getattr(client, "max_retries", 0)
    max_inner_attempts = (
        configured_inner_retries + 1 if _nonnegative_int(configured_inner_retries) else 1
    )
    # Pin whatever the backend can say about itself (CLI version, reasoning
    # effort) into the fingerprint, so an upgrade or a settings change re-samples
    # instead of quietly mixing generations of results.
    backend_version = ""
    client_provenance = getattr(client, "backend_provenance", None)
    if callable(client_provenance):
        try:
            backend_version = str(client_provenance())
        except Exception:  # a probe failure must not stop the experiment
            logger.warning("could not read the backend provenance", exc_info=True)
    source_files = (
        _revalidate_formal_checkout(formal_corpus)
        if formal_corpus is not None
        else _source_file_digests()
    )
    source_tree_sha256 = _source_tree_digest(source_files)
    if (
        registered_source_tree_sha256 is not None
        and source_tree_sha256 != registered_source_tree_sha256
    ):
        raise RuntimeError(
            "formal ablation source changed after attempt registration; "
            "mark the attempt ABANDONED"
        )
    llm_runtime = _client_runtime(
        llm,
        client,
        model=model,
        cli_path=cli_path,
        backend_details=backend_version,
    )
    agent_runtime = _execution_runtime(
        agent_exec,
        requested=agent_requested,
        requested_image=requested_docker_image,
        pinned_image_id=pinned_scorer_image_id,
        control_executable=docker_executable,
    )
    scorer_runtime = _execution_runtime(
        scorer,
        requested=scorer_backend,
        requested_image=requested_docker_image,
        pinned_image_id=pinned_scorer_image_id,
        control_executable=docker_executable,
    )
    runtime_packages = {
        name: _package_version(name)
        for name in ("lha", "pydantic", "PyYAML", "pytest", "pytest-json-report")
    }
    runtime_fingerprint: dict[str, Any] = {
        "llm": llm_runtime,
        "agent": agent_runtime,
        "scorer": scorer_runtime,
        "formal_git": (formal_corpus.git_executable if formal_corpus is not None else None),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "packages": runtime_packages,
        "docker_image_probe": docker_image_probe,
    }

    tasks: list[tuple[str, TaskSpec, Path, str]] = []
    task_files_sha256: dict[str, str] = {}
    corpus_sha256: dict[str, str] = {}
    input_snapshot_sha256: dict[str, str] = {}
    task_path_map: dict[str, str] = {}
    corpus_path_map: dict[str, str] = {}
    for tp in task_paths:
        name = Path(tp).stem
        if name in task_files_sha256:
            raise ValueError(f"duplicate ablation task name {name!r}")
        original_task = Path(tp)
        original_spec = TaskSpec.from_file(original_task)
        original_source = Path(original_spec.target_repo or ".")
        (
            frozen_task,
            source,
            task_digest,
            corpus_digest,
            snapshot_digest,
        ) = _freeze_ablation_input(
            out_dir=out,
            name=name,
            task_path=original_task,
            source=original_source,
        )
        spec = TaskSpec.from_file(frozen_task)
        spec.inputs["_name"] = name
        task_path_map[name] = _provenance_path(tp)
        corpus_path_map[name] = _provenance_path(original_source)
        task_files_sha256[name] = task_digest
        corpus_sha256[name] = corpus_digest
        input_snapshot_sha256[name] = snapshot_digest
        tasks.append(
            (
                name,
                spec,
                source,
                _fingerprint(
                    str(frozen_task),
                    source,
                    llm,
                    model,
                    scorer_backend,
                    backend_version,
                    source_files=source_files,
                    runtime=runtime_fingerprint,
                ),
            )
        )

    records: list[RunRecord] = []
    llm_calls: list[dict[str, Any]] = []
    total = len(tasks) * reps
    i = 0
    for rep in range(reps):
        for name, spec, source, fp in tasks:
            i += 1
            logger.info("ablation %d/%d: %s (rep %d)", i, total, name, rep)
            records.extend(
                _run_cell(
                    measurement_client,
                    source,
                    spec,
                    rep,
                    out,
                    fp,
                    corpus_sha256[name],
                    input_snapshot_sha256[name],
                    agent_exec,
                    scorer,
                    str(scorer_runtime["actual"]),
                    pinned_scorer_image_id,
                    llm_calls,
                    require_call_receipts,
                    max_inner_attempts,
                    formal_corpus is not None,
                    formal_output_lease=formal_output_lease,
                    formal_run_binding=formal_run_binding,
                )
            )

    final_source_files = (
        _revalidate_formal_checkout(formal_corpus)
        if formal_corpus is not None
        else _source_file_digests()
    )
    if final_source_files != source_files:
        raise RuntimeError(
            "lha source tree changed during the ablation; "
            "refusing to publish a mixed-implementation report"
        )

    git_executable: dict[str, Any] = {}
    if formal_corpus is not None:
        git_executable = _trusted_control_executable(
            "git",
            executable=str(formal_corpus.git_executable["path"]),
        )
        if git_executable != formal_corpus.git_executable:
            raise RuntimeError("Git executable changed during the formal ablation")
        git_commit, git_dirty = _git_provenance(str(git_executable["path"]))
        if git_commit != formal_corpus.preregistration_commit or git_dirty is not False:
            raise RuntimeError("Git state changed during the formal ablation; refusing its report")
    else:
        try:
            git_executable = _trusted_control_executable("git")
        except (OSError, RuntimeError, ValueError):
            git_executable = {}
        git_commit, git_dirty = _git_provenance(
            str(git_executable["path"]) if git_executable else None
        )
    if docker_executable:
        current_docker = resolve_docker_executable(str(docker_executable["path"])).as_provenance()
        if current_docker != docker_executable:
            raise RuntimeError("Docker executable changed during the ablation")
        for backend in (agent_exec, scorer):
            bind_control_plane = getattr(backend, "bind_control_plane", None)
            if callable(bind_control_plane) and (
                bind_control_plane(verify_digest=True) != docker_executable
            ):
                raise RuntimeError("Docker backend executable changed during the ablation")
        if formal_corpus is not None:
            assert isinstance(agent_exec, DockerBackend)
            _recover_docker_operations(
                agent_exec,
                out,
                allow_recovered=False,
            )
    configuration: dict[str, Any] = {
        "repetitions": reps,
        "task_count": len(tasks),
        "conditions": [name for name, _ in CONDITIONS],
        "max_repairs": _MAX_REPAIRS,
        "llm_retries": _LLM_RETRIES,
        "bootstrap_samples": _BOOTSTRAP_N,
        "cache_schema": _CACHE_SCHEMA,
        "report_schema": _REPORT_SCHEMA,
        "frozen_artifact_schema": _FROZEN_ARTIFACT_SCHEMA,
        "input_snapshot_schema": _INPUT_SNAPSHOT_SCHEMA,
        "scorer_evidence_schema": _SCORER_EVIDENCE_SCHEMA,
        "llm_call_receipt_schema": _LLM_CALL_RECEIPT_SCHEMA,
        "cell_attempt_schema": _CELL_ATTEMPT_SCHEMA,
        "formal_output_lock": (
            {
                "protocol": "flock-exclusive-nonblocking",
                "path": _FORMAL_OUTPUT_LOCK_NAME,
                "lifetime": "full-run",
            }
            if formal_corpus is not None
            else None
        ),
        "formal_fresh_run": (
            {
                "run_header_schema": _FORMAL_RUN_HEADER_SCHEMA,
                "run_header_path": _FORMAL_RUN_HEADER_NAME,
                "resume": False,
                "cache_reads": False,
                "expected_cell_starts": total,
                "expected_terminal_cells": total,
            }
            if formal_corpus is not None
            else None
        ),
        "codex_operation_lease_store": ("." if codex_operation_lease_bound else None),
        "docker_operation_lease_store": ("." if scorer_backend == "docker" else None),
        "docker_container_absence_filter": (
            "label=lha.operation_id" if scorer_backend == "docker" else None
        ),
        "docker_operations_recovered_before_run": (docker_operations_recovered_before_run),
        "docker_operations_recovered_at_completion": 0,
        "run_control_executables": {
            "git": git_executable,
            "docker": docker_executable,
        },
        "scorer_isolated_interpreter": True,
        "scorer_result_source": "nonce-bound-pytest-hook-receipt",
        "docker_image_probe": docker_image_probe,
        "client": llm_runtime["configuration"],
    }
    provenance = AblationProvenance(
        generated_at=now().isoformat(),
        harness_version=__version__,
        git_commit=git_commit,
        git_dirty=git_dirty,
        source_tree_sha256=source_tree_sha256,
        source_files=source_files,
        requested_llm_backend=llm,
        actual_llm_backend=str(llm_runtime["actual_backend"]),
        model=llm_runtime["model"] if isinstance(llm_runtime["model"], str) else None,
        cli_version=(
            llm_runtime["cli_version"] if isinstance(llm_runtime["cli_version"], str) else None
        ),
        cli_executable_sha256=formal_cli_executable_sha256,
        backend_library_version=(
            llm_runtime["backend_library_version"]
            if isinstance(llm_runtime["backend_library_version"], str)
            else None
        ),
        reasoning_effort=(
            llm_runtime["reasoning_effort"]
            if isinstance(llm_runtime["reasoning_effort"], str)
            else None
        ),
        backend_details=backend_version or None,
        agent_backend=str(agent_runtime["actual"]),
        scorer_requested=scorer_backend,
        scorer_backend=str(scorer_runtime["actual"]),
        scorer_image=(
            scorer_runtime["image"] if isinstance(scorer_runtime["image"], str) else None
        ),
        scorer_image_id=(
            scorer_runtime["image_id"] if isinstance(scorer_runtime["image_id"], str) else None
        ),
        platform=platform.platform(),
        python_version=platform.python_version(),
        pytest_version=_package_version("pytest"),
        pytest_json_report_version=_package_version("pytest-json-report"),
        runtime_packages=runtime_packages,
        task_paths=task_path_map,
        corpus_paths=corpus_path_map,
        task_files_sha256=task_files_sha256,
        corpus_sha256=corpus_sha256,
        input_snapshot_sha256=input_snapshot_sha256,
        cell_fingerprints={name: fingerprint for name, _spec, _source, fingerprint in tasks},
        formal_corpus_manifest_path=(formal_corpus.path if formal_corpus is not None else None),
        formal_corpus_manifest_sha256=(formal_corpus.sha256 if formal_corpus is not None else None),
        preregistration_commit=(
            formal_corpus.preregistration_commit if formal_corpus is not None else None
        ),
        formal_attempt_id=(
            formal_attempt_binding.attempt_id
            if formal_attempt_binding is not None
            else None
        ),
        formal_attempt_registry_path=(
            formal_attempt_binding.registry_path
            if formal_attempt_binding is not None
            else None
        ),
        formal_attempt_registry_sha256=(
            formal_attempt_binding.registry_sha256
            if formal_attempt_binding is not None
            else None
        ),
        formal_attempt_protocol_sha256=(
            formal_attempt_binding.protocol_sha256
            if formal_attempt_binding is not None
            else None
        ),
        formal_attempt_registration_commit=(
            formal_attempt_binding.registration_commit
            if formal_attempt_binding is not None
            else None
        ),
        formal_attempt_witness_remote_name=(
            formal_run_binding.witness_remote_name
            if formal_run_binding is not None
            else None
        ),
        formal_attempt_witness_remote_url=(
            formal_run_binding.witness_remote_url
            if formal_run_binding is not None
            else None
        ),
        formal_attempt_witness_ref=(
            formal_run_binding.witness_ref
            if formal_run_binding is not None
            else None
        ),
        formal_attempt_witness_commit=(
            formal_run_binding.witness_commit
            if formal_run_binding is not None
            else None
        ),
        formal_run_header_path=(
            formal_run_binding.header_path
            if formal_run_binding is not None
            else None
        ),
        formal_run_header_sha256=(
            formal_run_binding.header_sha256
            if formal_run_binding is not None
            else None
        ),
        formal_outcome_key=(
            formal_run_binding.outcome_key
            if formal_run_binding is not None
            else None
        ),
        git_executable=git_executable,
        docker_executable=docker_executable,
        configuration=configuration,
    )
    report = AblationReport(
        llm=llm,
        model=provenance.model or "",
        reps=reps,
        tasks=[t[0] for t in tasks],
        records=records,
        stats=_aggregate(records),
        scorer=scorer.name,
        fingerprint="",
        backend_version=backend_version,
        provenance=provenance,
        llm_calls=llm_calls,
    )
    artifact_digests = sorted(
        {record.artifact_sha256 for record in report.records if record.artifact_sha256}
    )
    scorer_evidence_digests = sorted(
        {
            record.scorer_evidence_sha256
            for record in report.records
            if record.scorer_evidence_sha256
        }
    )
    report_raw: dict[str, Any] = {
        "schema_version": report.schema_version,
        "llm": report.llm,
        "model": report.model,
        "reps": report.reps,
        "tasks": report.tasks,
        "scorer": report.scorer,
        "fingerprint": "",
        "backend_version": report.backend_version,
        "harness_version": __version__,
        "provenance": asdict(provenance),
        "artifact_store": {
            "schema_version": _FROZEN_ARTIFACT_SCHEMA,
            "path": "artifacts",
            "encoding": "canonical-json",
            "count": len(artifact_digests),
        },
        "scorer_evidence_store": {
            "schema_version": _SCORER_EVIDENCE_SCHEMA,
            "path": "scorer_evidence",
            "encoding": "canonical-json",
            "count": len(scorer_evidence_digests),
        },
        "llm_calls": report.llm_calls,
        "stats": [asdict(s) for s in report.stats],
        "records": [asdict(r) for r in report.records],
    }
    if require_call_receipts:
        receipt_digests = {
            call["receipt_sha256"]
            for call in report.llm_calls
            if isinstance(call.get("receipt_sha256"), str)
        }
        report_raw["llm_call_receipt_store"] = {
            "schema_version": _LLM_CALL_RECEIPT_SCHEMA,
            "path": "llm_call_receipts",
            "encoding": "canonical-json",
            "count": len(receipt_digests),
        }
    report.fingerprint = _report_fingerprint(report_raw)
    report_raw["fingerprint"] = report.fingerprint
    report_json = json.dumps(report_raw, indent=2)
    report_markdown = report.to_markdown()
    _write_ablation_reports(
        out,
        report_json=report_json,
        report_markdown=report_markdown,
        formal_corpus=formal_corpus,
        source_files=source_files,
    )
    return report
