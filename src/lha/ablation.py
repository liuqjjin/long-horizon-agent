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
  - Transient backend errors are retried and, if they persist, recorded as ``ERROR``
    (never cached; excluded from rates but counted and reported, never silently
    dropped).
  - Cached cells carry a provenance fingerprint over task/corpus bytes, the complete
    ``lha`` source tree, model/CLI settings, scorer identity, runtime versions, and
    repair configuration; any change recomputes.

A weaker implementer ``model`` calibrates difficulty, so first-attempt success lands
in a range where the gate has errors to catch. This runner isolates the gate
mechanism (single-step fix, no planning/context retrieval); it is not the full
harness loop.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .agents.implementer import Implementer
from .artifacts import Patch, Step
from .clock import now
from .config import Config
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
from .sandbox import ExecutionBackend, TrustedLocalBackend, make_backend
from .tasks.spec import TaskSpec
from .tools import policy
from .tools.patch import apply_patch
from .tools.shell import run
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
_CACHE_SCHEMA = 7
_REPORT_SCHEMA = 4
_FROZEN_ARTIFACT_SCHEMA = 1
_INPUT_SNAPSHOT_SCHEMA = 1
_SCORER_EVIDENCE_SCHEMA = 2
_BOOTSTRAP_N = 10_000
_READ_CHUNK_BYTES = 64 * 1024
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_SCORER_EVIDENCE_BYTES = 12 * 1024 * 1024
_MAX_CACHE_BYTES = 8 * 1024 * 1024
_MAX_REPORT_BYTES = 32 * 1024 * 1024
_MAX_TASK_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_TEXT_BYTES = 8 * 1024 * 1024


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
        mcnemar_lines = _paired_mcnemar_lines(self.records)
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
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stable_file_signature(opened) != _stable_file_signature(path_before)
        ):
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
                audit_log.append(_safe_call_audit(llm, label=label, status="succeeded"))
            return result
    raise _Transient(f"{label}: {last}")


def _pytest(
    workdir: Path,
    exec_backend: ExecutionBackend,
    *,
    isolated_interpreter: bool = False,
) -> PytestResult:
    """Run the prediction-side gate and retain an explicit infrastructure state."""
    step = Step(step_id="grade", kind="code", action="edit_code", goal="grade", verifiers=["pytest"])
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
            check.detail.get("returncode")
            if type(check.detail.get("returncode")) is int
            else None
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
        lambda: Implementer(llm).implement(_fix_step(task), _empty_bundle(), wd),
        "first",
        llm=llm,
        audit_log=audit_log,
    )
    return _sanitize(patch)


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
            if s is None or _read_bounded_text(
                s,
                max_bytes=_MAX_SOURCE_TEXT_BYTES,
                errors="replace",
                reject_hardlinks=False,
            ) != text:
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
        raise ValueError(
            f"frozen artifact exceeds the {_MAX_ARTIFACT_BYTES}-byte limit"
        )
    return payload


def _artifact_digest(frozen: dict[str, str | None]) -> str:
    return _sha256_bytes(_frozen_artifact_bytes(frozen))


def _store_frozen_artifact(frozen: dict[str, str | None], artifact_dir: Path) -> str:
    """Persist frozen bytes under their SHA-256 and reject conflicting content."""
    payload = _frozen_artifact_bytes(frozen)
    digest = _sha256_bytes(payload)
    path = artifact_dir / f"{digest}.json"
    if path.exists():
        if (
            _read_bounded_bytes(path, max_bytes=_MAX_ARTIFACT_BYTES)
            != payload
        ):
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
        raise ValueError(
            "scorer evidence exceeds "
            f"the {_MAX_SCORER_EVIDENCE_BYTES}-byte limit"
        )
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
        not isinstance(image_id, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
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
            gate_prediction=(
                None if first_gate.outcome is ScoreOutcome.INFRA_ERROR else gate_pred
            ),
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
            break  # nothing new to try
        apply_patch(repair, wd2)
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
                f"gate: {verify_gate.detail}; {score2.detail}"
                if verify_error
                else score2.detail
            ),
            gate_prediction=(
                None if verify_gate.outcome is ScoreOutcome.INFRA_ERROR else ok
            ),
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

        # no_tools here means "refuse a result produced with any tool": codex has
        # no deny-list flag, so leak-freedom is audited from the event stream.
        return CodexCLIClient(
            cli_path=cli_path, model=model, reasoning_effort=effort, no_tools=True
        )
    if llm == "anthropic":
        from .llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=model or "claude-opus-4-8")
    raise ValueError(f"unknown llm backend: {llm!r}")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


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
            raise RuntimeError(
                f"ablation input changed while snapshotting task {name!r}"
            )
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
                raise RuntimeError(
                    f"ablation input snapshot is corrupt for {snapshot_digest}"
                )
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
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        files[path.relative_to(package_root).as_posix()] = _sha256_regular_file(path)
    return files


def _source_tree_digest(source_files: dict[str, str]) -> str:
    return _canonical_digest(source_files)


def _project_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lha").is_dir():
            return candidate
    return None


def _provenance_path(path: str | Path) -> str:
    value = Path(path)
    root = _project_root()
    if root is not None:
        try:
            return value.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return str(value)


def _git_provenance() -> tuple[str | None, bool | None]:
    root = _project_root()
    if root is None:
        return None, None
    try:
        head = run(["git", "rev-parse", "--verify", "HEAD"], cwd=root, timeout=10)
        status = run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=root,
            timeout=10,
        )
    except OSError:
        return None, None
    commit = head.stdout.strip() if head.returncode == 0 and head.stdout.strip() else None
    dirty = bool(status.stdout) if status.returncode == 0 else None
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


def _resolve_docker_image_id(image: str, *, docker: str = "docker") -> str:
    """Resolve a mutable Docker reference once, before any experiment cell runs."""
    result = run(
        [docker, "image", "inspect", "--format", "{{.Id}}", image],
        timeout=30,
    )
    value = result.stdout.strip()
    if (
        result.returncode != 0
        or result.output_truncated
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        raise RuntimeError(
            f"Docker image {image!r} could not be bound to an immutable image ID"
        )
    return value


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
        data = json.loads(
            _read_bounded_text(path, max_bytes=_MAX_CACHE_BYTES)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None  # corrupt/partial cache -> recompute
    return _decode_cache(data)


def _load_cached(
    path: Path,
    fingerprint: str,
    *,
    input_snapshot_sha256: str | None = None,
    scorer_backend: str | None = None,
    scorer_image_id: str | None = None,
) -> list[RunRecord] | None:
    try:
        cache_envelope = json.loads(
            _read_bounded_text(path, max_bytes=_MAX_CACHE_BYTES)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if (
        not isinstance(cache_envelope, dict)
        or cache_envelope.get("schema_version") != _CACHE_SCHEMA
    ):
        return None
    decoded = _decode_cache(cache_envelope)
    if decoded is None:
        return None
    cached_fingerprint, records = decoded
    if cached_fingerprint != fingerprint:
        return None
    # ERROR is a missing measurement, never a durable observation. Refuse even
    # a hand-edited or old cache that contains one.
    if any(record.status == "ERROR" for record in records):
        return None
    if any(
        record.true_success != (record.claimed_success and record.artifact_correct)
        or record.false_success != (
            record.claimed_success and not record.artifact_correct
        )
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
    return records


def _read_cached_audits(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(
            _read_bounded_text(path, max_bytes=_MAX_CACHE_BYTES)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return []
    calls = raw.get("llm_calls") if isinstance(raw, dict) else None
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


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
) -> list[RunRecord]:
    name = task.inputs.get("_name", task.title)
    cache = out_dir / "results" / f"{name}__r{rep}.json"

    def error_records(detail: str) -> list[RunRecord]:
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
                repairs=0,
                detail=detail[:200],
            )
            for condition, _ in CONDITIONS
        ]

    def snapshot_matches() -> bool:
        try:
            return _repo_digest(source) == source_sha256
        except (OSError, RuntimeError):
            return False

    if not snapshot_matches():
        return error_records("content-addressed input snapshot failed validation")
    cached = _load_cached(
        cache,
        fingerprint,
        input_snapshot_sha256=input_snapshot_sha256,
        scorer_backend=scorer_backend,
        scorer_image_id=scorer_image_id,
    )
    if cached is not None:
        if audit_log is not None:
            audit_log.extend(
                {"task": name, "rep": rep, "cache_hit": True, **call}
                for call in _read_cached_audits(cache)
            )
        return cached
    cell_audits: list[dict[str, Any]] = []
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
        except _Transient as e:
            logger.error("transient failure on %s rep %d: %s — not caching", name, rep, e)
            if audit_log is not None:
                audit_log.extend(
                    {"task": name, "rep": rep, "cache_hit": False, **call}
                    for call in cell_audits
                )
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
            if audit_log is not None:
                audit_log.extend(
                    {"task": name, "rep": rep, "cache_hit": False, **call}
                    for call in cell_audits
                )
            return error_records(
                f"cell infrastructure failure: {type(error).__name__}"
            )
    if audit_log is not None:
        audit_log.extend(
            {"task": name, "rep": rep, "cache_hit": False, **call}
            for call in cell_audits
        )
    if any(record.status == "ERROR" for record in records):
        return records
    if not snapshot_matches():
        return error_records("content-addressed input snapshot changed during the cell")
    try:
        _atomic_write(
            cache,
            json.dumps(
                {
                    "schema_version": _CACHE_SCHEMA,
                    "fingerprint": fingerprint,
                    "records": [asdict(r) for r in records],
                    "llm_calls": cell_audits,
                },
                indent=2,
            ),
        )
    except Exception as error:
        logger.error(
            "cache write failure on %s rep %d: %s",
            name,
            rep,
            type(error).__name__,
            exc_info=True,
        )
        return error_records(f"cell cache write failure: {type(error).__name__}")
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


def _paired_mcnemar_lines(records: list[RunRecord]) -> list[str]:
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
        lines.append(
            f"- `{a}` vs `{b}` on {metric.replace('_', ' ')}: "
            f"discordant {only_a}/{only_b} of {len(pairs)} pairs · exact McNemar p = {p:.2f}"
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
    reasoning_effort = getattr(
        client, "reasoning_effort", getattr(client, "effort", None)
    )
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        reasoning_effort = str(reasoning_effort)
    safe_configuration: dict[str, Any] = {}
    for name in (
        "timeout",
        "no_tools",
        "sandbox_mode",
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
        "backend_library_version": (
            _package_version("anthropic") if llm == "anthropic" else None
        ),
        "reasoning_effort": reasoning_effort,
        "backend_details": backend_details or None,
        "configuration": safe_configuration,
    }


def _execution_runtime(
    backend: ExecutionBackend,
    *,
    requested: str,
    requested_image: str | None = None,
    pinned_image_id: str | None = None,
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
    }


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
    provenance = (
        AblationProvenance(**provenance_raw) if isinstance(provenance_raw, dict) else None
    )
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
        llm_calls=[
            call for call in raw.get("llm_calls", []) if isinstance(call, dict)
        ],
    )


def load_ablation_report(path: str | Path) -> AblationReport:
    """Read a bounded, stable historical or current ablation report."""
    raw = json.loads(
        _read_bounded_text(Path(path), max_bytes=_MAX_REPORT_BYTES)
    )
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
    out.mkdir(parents=True, exist_ok=True)
    # The backend's own env vars apply here exactly as in `lha run`; an explicit
    # --model wins. The resolved name feeds the provenance fingerprint.
    if llm == "codex_cli":
        model = model or (base.codex_model or None)
        cli_path, effort = base.codex_cli_path, base.codex_reasoning_effort
    else:
        model = model or (base.claude_cli_model or None)
        cli_path, effort = base.claude_cli_path, "medium"
    requested_docker_image: str | None = None
    pinned_scorer_image_id: str | None = None
    if scorer_backend == "docker":
        requested_docker_image = base.exec_image
        # Resolve the mutable tag before constructing the model client or
        # running any cell. Both prediction and truth execute the same immutable
        # image bytes even if the tag moves during a long experiment.
        pinned_scorer_image_id = _resolve_docker_image_id(requested_docker_image)
    client = llm_client or _make_llm(llm, model, cli_path=cli_path, effort=effort)
    # Pin whatever the backend can say about itself (CLI version, reasoning
    # effort) into the fingerprint, so an upgrade or a settings change re-samples
    # instead of quietly mixing generations of results.
    backend_version = ""
    probe = getattr(client, "backend_provenance", None)
    if callable(probe):
        try:
            backend_version = str(probe())
        except Exception:  # a probe failure must not stop the experiment
            logger.warning("could not read the backend provenance", exc_info=True)
    # Prediction and truth use separate backend instances, but the same
    # isolation class. Selecting Docker must never execute model-influenced code
    # in a host-side gate before the container scorer runs.
    if scorer_backend == "docker":
        if pinned_scorer_image_id is None:  # defensive; resolution above is fail-closed
            raise RuntimeError("Docker scorer image was not pinned")
        agent_exec = make_backend("docker", image=pinned_scorer_image_id)
        scorer = make_backend("docker", image=pinned_scorer_image_id)
        if any(
            getattr(backend, "name", None) != "docker"
            or getattr(backend, "image", None) != pinned_scorer_image_id
            for backend in (agent_exec, scorer)
        ):
            raise RuntimeError("Docker execution backends did not retain the pinned image ID")
        agent_requested = "docker"
    else:
        agent_exec = TrustedLocalBackend()
        scorer = make_backend(scorer_backend)
        agent_requested = "trusted-local"
    source_files = _source_file_digests()
    source_tree_sha256 = _source_tree_digest(source_files)
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
    )
    scorer_runtime = _execution_runtime(
        scorer,
        requested=scorer_backend,
        requested_image=requested_docker_image,
        pinned_image_id=pinned_scorer_image_id,
    )
    runtime_packages = {
        name: _package_version(name)
        for name in ("lha", "pydantic", "PyYAML", "pytest", "pytest-json-report")
    }
    runtime_fingerprint: dict[str, Any] = {
        "llm": llm_runtime,
        "agent": agent_runtime,
        "scorer": scorer_runtime,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "packages": runtime_packages,
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
                    client,
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
                )
            )

    git_commit, git_dirty = _git_provenance()
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
        "scorer_isolated_interpreter": True,
        "scorer_result_source": "nonce-bound-pytest-hook-receipt",
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
            llm_runtime["cli_version"]
            if isinstance(llm_runtime["cli_version"], str)
            else None
        ),
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
            scorer_runtime["image_id"]
            if isinstance(scorer_runtime["image_id"], str)
            else None
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
        configuration=configuration,
    )
    combined = _canonical_digest(
        {
            "report_schema": _REPORT_SCHEMA,
            "task_fingerprints": {name: fingerprint for name, _, _, fingerprint in tasks},
            "configuration": configuration,
            "runtime": runtime_fingerprint,
            "source_tree_sha256": source_tree_sha256,
            "git": {"commit": git_commit, "dirty": git_dirty},
        }
    )
    report = AblationReport(
        llm=llm,
        model=provenance.model or "",
        reps=reps,
        tasks=[t[0] for t in tasks],
        records=records,
        stats=_aggregate(records),
        scorer=scorer.name,
        fingerprint=combined,
        backend_version=backend_version,
        provenance=provenance,
        llm_calls=llm_calls,
    )
    artifact_digests = sorted(
        {
            record.artifact_sha256
            for record in report.records
            if record.artifact_sha256
        }
    )
    scorer_evidence_digests = sorted(
        {
            record.scorer_evidence_sha256
            for record in report.records
            if record.scorer_evidence_sha256
        }
    )
    _atomic_write(
        out / "ablation_report.json",
        json.dumps(
            {
                "schema_version": report.schema_version,
                "llm": report.llm,
                "model": report.model,
                "reps": report.reps,
                "tasks": report.tasks,
                "scorer": report.scorer,
                "fingerprint": report.fingerprint,
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
            },
            indent=2,
        ),
    )
    _atomic_write(out / "ablation_report.md", report.to_markdown())
    return report
