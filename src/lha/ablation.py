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

Every condition is scored by the final scorer, including attempts the gate refused,
so the gate's precision/recall/FPR/FNR and a full confusion matrix are measured
rather than assumed. (With a shared oracle, prediction==truth would hold by
construction; here divergence — stale caches, dirty workdirs, flaky tests — is
observable.)

Integrity properties:
  - Leak-free: the implementer is a single-shot completion with file tools denied
    (``no_tools``) and sees only non-test source, so it cannot read the oracle.
  - Tamper-proof: a patch may only edit source (tools.policy); the test oracle and
    config stay canonical, and the frozen diff excludes protected paths.
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
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .agents.implementer import Implementer
from .artifacts import Patch, Step
from .clock import now
from .config import Config
from .live_context.models import ContextBundle, Freshness
from .llm.base import LLMClient
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
_CACHE_SCHEMA = 4
_REPORT_SCHEMA = 2
_BOOTSTRAP_N = 10_000


class _Transient(Exception):
    """A backend error that should be retried / excluded, not counted as a result."""


@dataclass
class RunRecord:
    task: str
    condition: str
    rep: int
    status: str  # DONE | FAILED | ERROR
    claimed_success: bool  # the condition's own decision (internal gate for gate/verify)
    true_success: bool  # the independent final scorer's verdict on the frozen artifact
    false_success: bool  # claimed and not true
    repairs: int
    detail: str = ""
    # internal-gate prediction (None for trust, which runs no gate)
    gate_prediction: bool | None = None
    artifact_sha256: str = ""


@dataclass
class ConditionStats:
    condition: str
    blurb: str
    n: int
    claimed_success_rate: float
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
            "| condition | n | claimed | true success (95% CI) | false success (95% CI) "
            "| mean repairs | errors |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in self.stats:
            lines.append(
                f"| `{s.condition}` | {s.n} | {_pct(s.claimed_success_rate)} | "
                f"{_pct(s.true_success_rate)}{_ci(s.true_ci)} | "
                f"{_pct(s.false_success_rate)}{_ci(s.false_ci)} | "
                f"{s.mean_repairs:.2f} | {s.errors} |"
            )
        lines += ["", "Conditions:"]
        for name, blurb in CONDITIONS:
            lines.append(f"- `{name}` — {blurb}.")
        gate_lines = _gate_quality_lines(self.stats)
        if gate_lines:
            lines += ["", "Internal gate vs final scorer (per attempt):", *gate_lines]
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
            "Legend: pass = true success · fail = refused · false-pass = claimed but wrong. "
            "Exact counts across repetitions; outcomes are the final scorer's.",
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
        f"The gate discarded {gate.fn or 0} correct fix(es) (false negatives)."
    )


# --- core mechanics ---------------------------------------------------------
_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".pytest_cache", ".lha_pytest.json", ".cocoindex_code", ".git"
)
_DIFF_IGNORE = {"__pycache__", ".pytest_cache", ".lha_pytest.json", ".cocoindex_code", ".git"}


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


def _pytest(workdir: Path, exec_backend: ExecutionBackend) -> tuple[bool, list[str]]:
    """Run pytest via the given backend; return (passed, failing-assertion messages)."""
    step = Step(step_id="grade", kind="code", action="edit_code", goal="grade", verifiers=["pytest"])
    check = PytestVerifier().verify(
        None, VerifyContext(workdir=workdir, step=step, exec=exec_backend)
    )
    return check.passed, list(check.detail.get("messages", []))


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
            text = w.read_text(errors="replace")
            if s is None or s.read_text(errors="replace") != text:
                frozen[rel] = text
    return frozen


def _artifact_digest(frozen: dict[str, str | None]) -> str:
    h = hashlib.sha256()
    for rel in sorted(frozen):
        h.update(rel.encode())
        h.update(b"\0")
        content = frozen[rel]
        h.update(("\0<deleted>" if content is None else content).encode())
        h.update(b"\0")
    return h.hexdigest()


def _score(
    source: Path,
    frozen: dict[str, str | None],
    scratch: Path,
    label: str,
    scorer: ExecutionBackend,
) -> bool:
    """Final truth: canonical repo + frozen diff, graded in a fresh copy.

    Shares no state with the internal gate: fresh directory, canonical tests,
    its own execution backend.
    """
    wd = scratch / f"score_{label}"
    _copy_repo(source, wd, include_tests=True)
    for rel, content in frozen.items():
        target = wd / rel
        if content is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    ok, _ = _pytest(wd, scorer)
    return ok


def _evaluate(
    llm: LLMClient,
    source: Path,
    task: TaskSpec,
    patch: Patch,
    scratch: Path,
    rep: int,
    agent_exec: ExecutionBackend,
    scorer: ExecutionBackend,
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
    gate_pred, _ = _pytest(wd, agent_exec)
    frozen = _frozen_diff(source, wd)
    sha = _artifact_digest(frozen)
    truth = _score(source, frozen, scratch, "first", scorer)

    out.append(
        RunRecord(
            name, "trust", rep, "DONE", True, truth, not truth, 0,
            _truth_detail(truth), None, sha,
        )
    )
    # The scorer grades the SAME artifact even when the gate refused it, so a
    # refusal of a correct fix is counted (false negative), not invisible.
    out.append(
        RunRecord(
            name, "gate", rep, "DONE" if gate_pred else "FAILED",
            gate_pred, truth, gate_pred and not truth, 0,
            _truth_detail(truth), gate_pred, sha,
        )
    )

    # verify: same first attempt, then repair with failing-test feedback.
    wd2 = scratch / "verify"
    _copy_repo(source, wd2, include_tests=True)
    if patch.file_contents:
        apply_patch(patch, wd2)
    ok, failures = _pytest(wd2, agent_exec)
    repairs = 0
    while not ok and repairs < _MAX_REPAIRS:
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
        ok, failures = _pytest(wd2, agent_exec)
    frozen2 = _frozen_diff(source, wd2)
    sha2 = _artifact_digest(frozen2)
    truth2 = _score(source, frozen2, scratch, "verify", scorer)
    out.append(
        RunRecord(
            name, "verify", rep, "DONE" if ok else "FAILED",
            ok, truth2, ok and not truth2, repairs,
            _truth_detail(truth2), ok, sha2,
        )
    )
    return out


def _truth_detail(ok: bool) -> str:
    return "scorer: tests pass" if ok else "scorer: tests fail"


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


# --- provenance fingerprint ---------------------------------------------------
def _repo_digest(source: Path) -> str:
    h = hashlib.sha256()
    for rel, p in sorted(_iter_files(source)):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


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
        files[path.relative_to(package_root).as_posix()] = _sha256_bytes(path.read_bytes())
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


def _docker_image_id(backend: ExecutionBackend) -> str | None:
    image = getattr(backend, "image", None)
    docker = getattr(backend, "docker", None)
    if not isinstance(image, str) or not image or not isinstance(docker, str):
        return None
    try:
        result = run(
            [docker, "image", "inspect", "--format", "{{.Id}}", image],
            timeout=30,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


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
        "task_sha256": _sha256_bytes(Path(task_path).read_bytes()),
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


def _read_cache(path: Path) -> tuple[str | None, list[RunRecord]] | None:
    """Decode both legacy and current cache files without trusting legacy cells.

    Pre-fingerprint caches remain inspectable, but ``_load_cached`` only reuses
    records whose current fingerprint matches. This preserves compatibility
    without allowing an unbound historical result into a new report.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            raw_records = data
            fingerprint = None
        elif isinstance(data, dict) and isinstance(data.get("records"), list):
            raw_records = data["records"]
            value = data.get("fingerprint")
            fingerprint = value if isinstance(value, str) and value else None
        else:
            return None
        records = [RunRecord(**record) for record in raw_records]
        return fingerprint, records
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return None  # corrupt/partial cache -> recompute


def _load_cached(path: Path, fingerprint: str) -> list[RunRecord] | None:
    decoded = _read_cache(path)
    if decoded is None:
        return None
    cached_fingerprint, records = decoded
    if cached_fingerprint != fingerprint:
        return None
    # ERROR is a missing measurement, never a durable observation. Refuse even
    # a hand-edited or old cache that contains one.
    if any(record.status == "ERROR" for record in records):
        return None
    return records


def _read_cached_audits(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
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
    agent_exec: ExecutionBackend,
    scorer: ExecutionBackend,
    audit_log: list[dict[str, Any]] | None = None,
) -> list[RunRecord]:
    name = task.inputs.get("_name", task.title)
    cache = out_dir / "results" / f"{name}__r{rep}.json"
    cached = _load_cached(cache, fingerprint)
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
                cell_audits,
            )
        except _Transient as e:
            logger.error("transient failure on %s rep %d: %s — not caching", name, rep, e)
            if audit_log is not None:
                audit_log.extend(
                    {"task": name, "rep": rep, "cache_hit": False, **call}
                    for call in cell_audits
                )
            return [
                RunRecord(name, c, rep, "ERROR", False, False, False, 0, str(e)[:200])
                for c, _ in CONDITIONS
            ]
    if audit_log is not None:
        audit_log.extend(
            {"task": name, "rep": rep, "cache_hit": False, **call}
            for call in cell_audits
        )
    if any(record.status == "ERROR" for record in records):
        return records
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
            stats.append(ConditionStats(name, blurb, 0, 0.0, 0.0, 0.0, 0.0, errors=errors))
            continue
        s = ConditionStats(
            condition=name,
            blurb=blurb,
            n=n,
            claimed_success_rate=sum(r.claimed_success for r in recs) / n,
            true_success_rate=sum(r.true_success for r in recs) / n,
            false_success_rate=sum(r.false_success for r in recs) / n,
            mean_repairs=sum(r.repairs for r in recs) / n,
            errors=errors,
            true_ci=_rate_ci(recs, "true_success"),
            false_ci=_rate_ci(recs, "false_success"),
        )
        preds = [r for r in recs if r.gate_prediction is not None]
        if preds:
            tp = sum(bool(r.gate_prediction) and r.true_success for r in preds)
            fp = sum(bool(r.gate_prediction) and not r.true_success for r in preds)
            tn = sum(not r.gate_prediction and not r.true_success for r in preds)
            fn = sum(not r.gate_prediction and r.true_success for r in preds)
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
    backend: ExecutionBackend, *, requested: str
) -> dict[str, Any]:
    image = getattr(backend, "image", None)
    return {
        "requested": requested,
        "actual": getattr(backend, "name", type(backend).__name__) or "unknown",
        "implementation": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "image": image if isinstance(image, str) and image else None,
        "image_id": _docker_image_id(backend),
    }


def _read_condition_stats(raw: dict[str, Any]) -> ConditionStats:
    values = dict(raw)
    for name in ("true_ci", "false_ci"):
        interval = values.get(name)
        if isinstance(interval, list) and len(interval) == 2:
            values[name] = (float(interval[0]), float(interval[1]))
    return ConditionStats(**values)


def load_ablation_report(path: str | Path) -> AblationReport:
    """Read both the pre-provenance report format and schema-2 reports."""
    raw = json.loads(Path(path).read_text())
    provenance_raw = raw.get("provenance")
    provenance = (
        AblationProvenance(**provenance_raw) if isinstance(provenance_raw, dict) else None
    )
    return AblationReport(
        llm=str(raw.get("llm", "unknown")),
        model=str(raw.get("model", "")),
        reps=int(raw.get("reps", 0)),
        tasks=[str(task) for task in raw.get("tasks", [])],
        records=[RunRecord(**record) for record in raw.get("records", [])],
        stats=[_read_condition_stats(stat) for stat in raw.get("stats", [])],
        scorer=str(raw.get("scorer", "unknown")),
        fingerprint=str(raw.get("fingerprint", "")),
        backend_version=str(raw.get("backend_version", "")),
        schema_version=int(raw.get("schema_version", 1)),
        provenance=provenance,
        llm_calls=[
            call for call in raw.get("llm_calls", []) if isinstance(call, dict)
        ],
    )


def run_ablation(
    base: Config,
    task_paths: list[str],
    *,
    llm: str = "claude_cli",
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
    # The agent-side gate and the final scorer never share an execution backend
    # instance; the scorer can be a container while the gate stays local.
    agent_exec: ExecutionBackend = TrustedLocalBackend()
    scorer: ExecutionBackend = (
        make_backend("docker", image=base.exec_image)
        if scorer_backend == "docker"
        else make_backend(scorer_backend)
    )
    source_files = _source_file_digests()
    source_tree_sha256 = _source_tree_digest(source_files)
    llm_runtime = _client_runtime(
        llm,
        client,
        model=model,
        cli_path=cli_path,
        backend_details=backend_version,
    )
    agent_runtime = _execution_runtime(agent_exec, requested="trusted-local")
    scorer_runtime = _execution_runtime(scorer, requested=scorer_backend)
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
    task_path_map: dict[str, str] = {}
    corpus_path_map: dict[str, str] = {}
    for tp in task_paths:
        name = Path(tp).stem
        if name in task_files_sha256:
            raise ValueError(f"duplicate ablation task name {name!r}")
        spec = TaskSpec.from_file(tp)
        spec.inputs["_name"] = name
        source = Path(spec.target_repo or ".")
        task_path_map[name] = _provenance_path(tp)
        corpus_path_map[name] = _provenance_path(source)
        task_files_sha256[name] = _sha256_bytes(Path(tp).read_bytes())
        corpus_sha256[name] = _repo_digest(source)
        tasks.append(
            (
                name,
                spec,
                source,
                _fingerprint(
                    tp,
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
                    agent_exec,
                    scorer,
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
                "llm_calls": report.llm_calls,
                "stats": [asdict(s) for s in report.stats],
                "records": [asdict(r) for r in report.records],
            },
            indent=2,
        ),
    )
    _atomic_write(out / "ablation_report.md", report.to_markdown())
    return report
