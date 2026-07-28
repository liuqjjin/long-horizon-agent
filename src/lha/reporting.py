"""Read-only run inspection, self-contained HTML traces, and safe retention.

The reporting layer never guesses past damaged evidence. Checkpoints and ledgers
are loaded through their authoritative validators, recognized JSON artifacts are
validated before display or deletion, and pruning re-checks terminal state while
holding the same per-run lock used by the harness.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from .agents.experimenter import (
    ExperimentEvidence,
    ExperimentIntent,
    build_cmd,
    validate_experiment_result,
)
from .artifacts import ExperimentResult, Patch, Plan
from .clock import now
from .harness.approval import (
    ApprovalDecision,
    ApprovalRequest,
    approval_decision_path,
    approval_decision_ref,
    approval_request_path,
    read_approval_decision,
    read_approval_request,
    validate_decision_binding,
)
from .harness.checkpoint import load_state, read_ledger, run_lock
from .harness.errors import CheckpointCorrupt, RunLocked
from .harness.manifest import ArtifactManifest, saved_file_state, sha256_bytes
from .harness.state import RunState, StepRecord
from .harness.transaction import (
    PatchTransaction,
    attempt_artifact_dir,
    list_transactions,
    read_transaction_events,
    state_for_paths,
    transaction_log_path,
    transaction_path,
    validate_terminal_transaction_state,
    validate_transaction_journals,
)
from .live_context.models import ContextBundle
from .llm.trace import LLMUsageTotals, load_usage_checkpoint
from .repo_adapter import (
    RepoAdapterSpec,
    RepoStageEvidence,
    RepoStageIntent,
)
from .tools.patch import (
    backup_sha256,
    load_backup,
    render_review_diff,
    resolve_patch,
)
from .verifiers.verdict import Verdict

_TERMINAL = frozenset({"DONE", "FAILED"})
_SAFE_STEP_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
PruneAction = Literal["WOULD DELETE", "DELETED", "REFUSE"]


class ReportingError(RuntimeError):
    """A run cannot be inspected or safely pruned."""


@dataclass(frozen=True)
class NamedArtifact:
    path: str
    value: Any


@dataclass(frozen=True)
class UsageSummary:
    calls: int
    wall_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    trace_calls: int
    trace_complete: bool


@dataclass(frozen=True)
class RunReport:
    run_dir: Path
    updated_at: datetime
    state: RunState
    ledger: list[StepRecord]
    patches: list[NamedArtifact]
    approvals: list[NamedArtifact]
    verdicts: list[NamedArtifact]
    llm_records: list[dict[str, Any]]
    usage: UsageSummary


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    status: str
    task: str
    updated_at: datetime
    run_dir: Path
    error: str = ""


@dataclass(frozen=True)
class PruneEntry:
    run_id: str
    status: str
    action: PruneAction
    detail: str


@dataclass(frozen=True)
class PruneResult:
    entries: list[PruneEntry]

    @property
    def refused(self) -> int:
        return sum(entry.action == "REFUSE" for entry in self.entries)

    @property
    def deleted(self) -> int:
        return sum(entry.action == "DELETED" for entry in self.entries)


def collect_run(runs_dir: str | Path, run_id: str) -> RunReport:
    """Load and validate one run without trusting paths stored in its checkpoint."""
    run_dir = _resolve_run_dir(runs_dir, run_id)
    state_path = _regular_file(run_dir / "state.json", required=True)
    try:
        state = load_state(run_dir)
    except Exception as e:
        raise ReportingError(str(e)) from e
    if state.run_id != run_id:
        raise ReportingError(
            f"checkpoint run_id {state.run_id!r} does not match directory {run_id!r}"
        )
    ledger_path = run_dir / "ledger.jsonl"
    if ledger_path.exists() or ledger_path.is_symlink():
        _regular_file(ledger_path, required=True)
    try:
        ledger = read_ledger(run_dir)
    except Exception as e:
        raise ReportingError(str(e)) from e

    patches = [
        NamedArtifact(_relative(path, run_dir), _load_model(path, Patch))
        for path in _artifact_paths(run_dir, "patch.json")
    ]
    verdicts = [
        NamedArtifact(_relative(path, run_dir), _load_model(path, Verdict))
        for path in _artifact_paths(run_dir, "verify.json")
    ]
    approvals = _collect_approval_artifacts(run_dir, state)
    llm_records = _load_jsonl(run_dir / "llm_trace.jsonl")
    validate_recovery_evidence(run_dir)
    if state.status in _TERMINAL:
        validate_terminal_evidence(run_dir, state, ledger)
        validate_llm_attempt_evidence(
            run_dir, state, ledger, llm_records
        )
    try:
        updated_at = datetime.fromtimestamp(state_path.stat().st_mtime, tz=timezone.utc)
    except OSError as e:
        raise ReportingError(f"cannot stat checkpoint {state_path}: {e}") from e
    return RunReport(
        run_dir=run_dir,
        updated_at=updated_at,
        state=state,
        ledger=ledger,
        patches=patches,
        approvals=approvals,
        verdicts=verdicts,
        llm_records=llm_records,
        usage=_usage_summary(state, llm_records, run_dir),
    )


def discover_runs(runs_dir: str | Path) -> list[RunSummary]:
    """List run checkpoints; report damaged candidates instead of hiding them."""
    root = _runs_root(runs_dir)
    if not root.exists():
        return []
    summaries: list[RunSummary] = []
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as e:
        raise ReportingError(f"cannot list runs directory {root}: {e}") from e
    for child in children:
        if child.is_symlink():
            summaries.append(
                RunSummary(
                    run_id=child.name,
                    status="CORRUPT",
                    task="-",
                    updated_at=_mtime(child),
                    run_dir=child,
                    error="run directory is a symlink",
                )
            )
            continue
        if not child.is_dir():
            continue
        state_path = child / "state.json"
        if not state_path.exists() and not state_path.is_symlink():
            continue  # report directories such as ablation/ and horizon/ are not runs
        try:
            report = collect_run(root, child.name)
        except ReportingError as e:
            summaries.append(
                RunSummary(
                    run_id=child.name,
                    status="CORRUPT",
                    task="-",
                    updated_at=_mtime(state_path if state_path.exists() else child),
                    run_dir=child,
                    error=str(e),
                )
            )
            continue
        summaries.append(
            RunSummary(
                run_id=report.state.run_id,
                status=report.state.status,
                task=report.state.task.title,
                updated_at=report.updated_at,
                run_dir=report.run_dir,
            )
        )
    return sorted(summaries, key=lambda item: (item.updated_at, item.run_id), reverse=True)


def prune_runs(
    runs_dir: str | Path,
    *,
    older_than_days: int,
    apply: bool = False,
) -> PruneResult:
    """Select old runs and optionally delete only revalidated terminal runs."""
    if older_than_days < 0:
        raise ReportingError("--older-than-days must be >= 0")
    root = _runs_root(runs_dir)
    try:
        cutoff = now() - timedelta(days=older_than_days)
    except OverflowError as e:
        raise ReportingError("--older-than-days is too large") from e
    entries: list[PruneEntry] = []
    for summary in discover_runs(root):
        if summary.updated_at > cutoff:
            continue
        if summary.status not in _TERMINAL:
            detail = summary.error or f"status {summary.status} is not terminal"
            entries.append(PruneEntry(summary.run_id, summary.status, "REFUSE", detail))
            continue
        if not apply:
            entries.append(
                PruneEntry(
                    summary.run_id,
                    summary.status,
                    "WOULD DELETE",
                    f"checkpoint updated {summary.updated_at.isoformat()}",
                )
            )
            continue
        try:
            lock_path = summary.run_dir / ".run.lock"
            if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
                raise ReportingError(f"unsafe run lock path {lock_path}")
            with run_lock(summary.run_dir):
                # State, ledger, and recognized evidence are re-read under the
                # harness lock. A resume racing this command therefore wins or
                # makes the prune refuse; it can never be deleted mid-run.
                current = collect_run(root, summary.run_id)
                if current.state.status not in _TERMINAL:
                    raise ReportingError(
                        f"status changed to {current.state.status}; refusing deletion"
                    )
                if current.updated_at > cutoff:
                    raise ReportingError("checkpoint became newer than the retention cutoff")
                shutil.rmtree(current.run_dir)
        except (CheckpointCorrupt, ReportingError, RunLocked, OSError) as e:
            entries.append(PruneEntry(summary.run_id, summary.status, "REFUSE", str(e)))
        else:
            entries.append(
                PruneEntry(
                    summary.run_id,
                    summary.status,
                    "DELETED",
                    "permanently removed after terminal-state revalidation",
                )
            )
    return PruneResult(entries)


def write_html_trace(
    runs_dir: str | Path,
    run_id: str,
    *,
    out: str | Path | None = None,
) -> Path:
    report = collect_run(runs_dir, run_id)
    output = Path(out).expanduser() if out else report.run_dir / "trace.html"
    if output.is_symlink():
        raise ReportingError(f"refusing to overwrite symlink output {output}")
    temporary: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(fd, "w") as stream:
            stream.write(render_html(report))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except OSError as e:
        raise ReportingError(f"cannot write HTML trace {output}: {e}") from e
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def render_html(report: RunReport) -> str:
    """Render a dependency-free HTML document; every persisted value is escaped."""
    state_json = _json_text(report.state.model_dump(mode="json"))
    ledger_rows = (
        "".join(
            "<tr>"
            f"<td>{record.seq}</td>"
            f"<td>{_h(record.timestamp.isoformat())}</td>"
            f"<td>{_h(record.step_id)}</td>"
            f"<td><span class='pill phase'>{_h(record.phase)}</span></td>"
            f"<td>{_h(record.artifact_ref or record.verdict_ref or '—')}</td>"
            f"<td>{_h(record.notes or '—')}</td>"
            "</tr>"
            for record in report.ledger
        )
        or "<tr><td colspan='6' class='empty'>No ledger events.</td></tr>"
    )

    patch_html = "".join(_render_patch(item) for item in report.patches) or _empty(
        "No patch artifacts."
    )
    approval_html = "".join(
        _artifact_json(item.path, _model_json(item.value)) for item in report.approvals
    )
    approval_events = [
        record.model_dump(mode="json") for record in report.ledger if record.phase == "approval"
    ]
    if approval_events:
        approval_html += _artifact_json("ledger approval history", approval_events)
    if not approval_html:
        approval_html = _empty("No approval request, decision, or ledger event is persisted.")
    verdict_html = "".join(_render_verdict(item) for item in report.verdicts) or _empty(
        "No verdict artifacts."
    )
    usage_rows = (
        "".join(
            _render_usage_row(index, record)
            for index, record in enumerate(report.llm_records, start=1)
        )
        or "<tr><td colspan='8' class='empty'>No per-call LLM trace.</td></tr>"
    )
    status_class = (
        "good"
        if report.state.status == "DONE"
        else ("bad" if report.state.status == "FAILED" else "warn")
    )
    usage_warning = (
        ""
        if report.usage.trace_complete
        else (
            "<p class='warn'>The durable usage checkpoint is authoritative. "
            f"It records {report.usage.calls} call(s), while the optional trace "
            f"contains {report.usage.trace_calls} row(s).</p>"
        )
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
  <title>LHA trace · {_h(report.state.run_id)}</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#0b1020; --panel:#121a2d; --muted:#94a3b8;
      --text:#e5e7eb; --line:#26344f; --blue:#60a5fa; --green:#34d399;
      --red:#fb7185; --amber:#fbbf24; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text);
      font:14px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:40px 24px 80px; }}
    header {{ margin-bottom:28px; }}
    h1 {{ margin:0 0 8px; font-size:30px; letter-spacing:-.02em; }}
    h2 {{ margin:34px 0 12px; font-size:20px; }}
    h3 {{ margin:0 0 12px; font-size:15px; color:var(--blue); overflow-wrap:anywhere; }}
    p {{ margin:6px 0; }}
    .muted,.empty {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
    .card,.artifact {{ background:var(--panel); border:1px solid var(--line);
      border-radius:12px; padding:16px; box-shadow:0 10px 30px #0002; }}
    .artifact {{ margin:12px 0; }}
    .label {{ color:var(--muted); text-transform:uppercase; font-size:11px; letter-spacing:.08em; }}
    .value {{ font-size:20px; font-weight:700; overflow-wrap:anywhere; }}
    .pill {{ display:inline-block; border:1px solid var(--line); border-radius:999px;
      padding:2px 8px; font-size:12px; }}
    .good {{ color:var(--green); }} .bad {{ color:var(--red); }} .warn {{ color:var(--amber); }}
    .phase {{ color:var(--blue); }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel);
      border:1px solid var(--line); border-radius:12px; overflow:hidden; display:block;
      overflow-x:auto; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left;
      vertical-align:top; white-space:nowrap; }}
    th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }}
    td:last-child {{ white-space:normal; min-width:220px; }}
    pre {{ margin:0; padding:14px; border-radius:8px; background:#070b15; color:#dbeafe;
      overflow:auto; max-height:520px; white-space:pre-wrap; overflow-wrap:anywhere;
      font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
    details {{ margin-top:10px; }} summary {{ cursor:pointer; color:var(--muted); }}
    dl {{ display:grid; grid-template-columns:max-content 1fr; gap:4px 12px; }}
    dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
    @media print {{ body {{ background:white; color:#111; }} .card,.artifact,table {{ box-shadow:none; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="label">Validated run trace</div>
    <h1>{_h(report.state.task.title)}</h1>
    <p class="muted">{_h(report.state.run_id)} · updated {_h(report.updated_at.isoformat())}</p>
  </header>
  <section class="grid">
    {_metric("Status", report.state.status, status_class)}
    {_metric("Completed steps", str(len(report.state.completed_steps)))}
    {_metric("Repairs", str(sum(report.state.repairs.values())))}
    {_metric("Ledger events", str(len(report.ledger)))}
  </section>

  <h2>State</h2>
  <div class="artifact"><pre>{_h(state_json)}</pre></div>

  <h2>Ledger</h2>
  <table><thead><tr><th>Seq</th><th>Timestamp</th><th>Step</th><th>Phase</th>
    <th>Reference</th><th>Notes</th></tr></thead><tbody>{ledger_rows}</tbody></table>

  <h2>Patch</h2>
  {patch_html}

  <h2>Approval</h2>
  {approval_html}

  <h2>Verdict</h2>
  {verdict_html}

  <h2>LLM usage</h2>
  <section class="grid">
    {_metric("Calls", str(report.usage.calls))}
    {_metric("Wall time", f"{report.usage.wall_s:.3f}s")}
    {_metric("Input tokens", f"{report.usage.input_tokens:,}")}
    {_metric("Output tokens", f"{report.usage.output_tokens:,}")}
    {_metric("Reported cost", f"${report.usage.cost_usd:.4f}")}
    {_metric("Trace rows", str(report.usage.trace_calls))}
  </section>
  {usage_warning}
  <table><thead><tr><th>#</th><th>Kind</th><th>Backend</th><th>Model</th>
    <th>Status</th><th>Retries</th><th>Tokens in/out</th><th>Duration</th></tr></thead>
    <tbody>{usage_rows}</tbody></table>
</main>
</body>
</html>
"""


def _render_patch(item: NamedArtifact) -> str:
    patch = item.value
    assert isinstance(patch, Patch)
    resolved = resolve_patch(patch)
    if resolved.mode == "contents":
        diff = "\n\n".join(
            f"### {path}\n{patch.file_contents[path]}"
            for path in resolved.paths
        )
    else:
        diff = patch.unified_diff or "(empty patch)"
    return (
        "<article class='artifact'>"
        f"<h3>{_h(item.path)}</h3>"
        "<dl>"
        f"<dt>Step</dt><dd>{_h(patch.step_id)}</dd>"
        f"<dt>Files</dt><dd>{_h(', '.join(resolved.paths) or '—')}</dd>"
        f"<dt>Rationale</dt><dd>{_h(patch.rationale or '—')}</dd>"
        "</dl>"
        f"<pre>{_h(diff)}</pre>"
        f"<details><summary>Structured patch JSON</summary><pre>{_h(_json_text(patch.model_dump(mode='json')))}</pre></details>"
        "</article>"
    )


def _render_verdict(item: NamedArtifact) -> str:
    verdict = item.value
    assert isinstance(verdict, Verdict)
    rows = (
        "".join(
            "<tr>"
            f"<td>{_h(check.name)}</td>"
            f"<td class='{'good' if check.passed else 'bad'}'>{str(check.passed).lower()}</td>"
            f"<td>{_h(str(check.score) if check.score is not None else '—')}</td>"
            f"<td>{_h(str(check.threshold) if check.threshold is not None else '—')}</td>"
            f"<td>{_h(str(check.detail.get('summary') or check.detail))}</td>"
            "</tr>"
            for check in verdict.checks
        )
        or "<tr><td colspan='5' class='empty'>No checks.</td></tr>"
    )
    return (
        "<article class='artifact'>"
        f"<h3>{_h(item.path)}</h3>"
        f"<p>step={_h(verdict.step_id)} · passed="
        f"<strong class='{'good' if verdict.passed else 'bad'}'>{str(verdict.passed).lower()}</strong></p>"
        "<table><thead><tr><th>Check</th><th>Passed</th><th>Score</th>"
        f"<th>Threshold</th><th>Summary</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<details><summary>Structured verdict JSON</summary><pre>{_h(_json_text(verdict.model_dump(mode='json')))}</pre></details>"
        "</article>"
    )


def _render_usage_row(index: int, record: dict[str, Any]) -> str:
    usage = record.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = _int_field(usage.get("input_tokens"), "usage.input_tokens")
    output_tokens = _int_field(usage.get("output_tokens"), "usage.output_tokens")
    duration = _float_field(record.get("duration_s") or usage.get("duration_s"), "duration_s")
    return (
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{_h(str(record.get('kind') or '—'))}</td>"
        f"<td>{_h(str(record.get('backend') or '—'))}</td>"
        f"<td>{_h(str(usage.get('model') or '—'))}</td>"
        f"<td>{_h(str(usage.get('status') or '—'))}</td>"
        f"<td>{_h(str(usage.get('retries') if usage.get('retries') is not None else '—'))}</td>"
        f"<td>{input_tokens:,} / {output_tokens:,}</td>"
        f"<td>{duration:.3f}s</td>"
        "</tr>"
    )


def _artifact_json(path: str, value: Any) -> str:
    return (
        f"<article class='artifact'><h3>{_h(path)}</h3><pre>{_h(_json_text(value))}</pre></article>"
    )


def _metric(label: str, value: str, css_class: str = "") -> str:
    return (
        "<div class='card'>"
        f"<div class='label'>{_h(label)}</div>"
        f"<div class='value {_h(css_class)}'>{_h(value)}</div>"
        "</div>"
    )


def _empty(message: str) -> str:
    return f"<div class='artifact empty'>{_h(message)}</div>"


def _usage_summary(
    state: RunState,
    records: list[dict[str, Any]],
    run_dir: Path,
) -> UsageSummary:
    # Validate every optional trace row, but never replace the checksummed state
    # totals with a best-effort log that may be truncated or missing.
    for index, record in enumerate(records, start=1):
        _float_field(record.get("duration_s"), f"LLM record {index}.duration_s")
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        _int_field(
            usage.get("input_tokens"), f"LLM record {index}.usage.input_tokens"
        )
        _int_field(
            usage.get("output_tokens"), f"LLM record {index}.usage.output_tokens"
        )
        _float_field(usage.get("cost_usd"), f"LLM record {index}.usage.cost_usd")
    state_usage = LLMUsageTotals(
        calls=_int_field(state.llm_usage.calls, "state.llm_usage.calls"),
        wall_s=_float_field(state.llm_usage.wall_s, "state.llm_usage.wall_s"),
        input_tokens=_int_field(
            state.llm_usage.input_tokens, "state.llm_usage.input_tokens"
        ),
        output_tokens=_int_field(
            state.llm_usage.output_tokens, "state.llm_usage.output_tokens"
        ),
        cost_usd=_float_field(
            state.llm_usage.cost_usd, "state.llm_usage.cost_usd"
        ),
    )
    try:
        durable = load_usage_checkpoint(run_dir)
    except Exception as error:
        raise ReportingError(str(error)) from error
    if durable is None or state_usage.calls > durable.calls:
        usage = state_usage
    elif durable.calls > state_usage.calls:
        usage = durable
    elif durable != state_usage:
        raise ReportingError(
            "RunState and the LLM usage checkpoint disagree at the same call count"
        )
    else:
        usage = state_usage
    calls = usage.calls
    return UsageSummary(
        calls=calls,
        wall_s=usage.wall_s,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        trace_calls=len(records),
        trace_complete=len(records) == calls,
    )


def _approval_evidence(
    run_dir: Path,
) -> dict[
    tuple[str, str],
    tuple[Any, Any | None],
]:
    """Load every immutable request/decision and reject orphaned paths."""
    root = run_dir / "steps"
    if not root.exists() and not root.is_symlink():
        return {}
    if root.is_symlink() or not root.is_dir():
        raise ReportingError(f"approval artifact root is unsafe: {root}")

    paths = sorted(
        path
        for name in ("approval_request.json", "approval_decision.json")
        for path in root.rglob(name)
    )
    identities: set[tuple[str, str]] = set()
    for path in paths:
        _regular_file(path, required=True)
        relative = path.relative_to(run_dir)
        if (
            len(relative.parts) != 5
            or relative.parts[0] != "steps"
            or relative.parts[2] != "attempts"
        ):
            raise ReportingError(
                f"approval evidence is outside an attempt directory: {path}"
            )
        identities.add((relative.parts[1], relative.parts[3]))

    history: dict[tuple[str, str], tuple[Any, Any | None]] = {}
    for step_id, attempt_id in sorted(identities):
        try:
            request = read_approval_request(run_dir, step_id, attempt_id)
            decision = read_approval_decision(run_dir, step_id, attempt_id)
        except ValueError as error:
            raise ReportingError(str(error)) from error
        if request is None:
            raise ReportingError(
                f"approval decision has no immutable request: {step_id}/{attempt_id}"
            )
        if (
            request.value.step_id != step_id
            or request.value.attempt_id != attempt_id
            or approval_request_path(run_dir, step_id, attempt_id)
            not in paths
        ):
            raise ReportingError(
                f"approval request path does not match its identity: {step_id}/{attempt_id}"
            )
        if decision is not None:
            try:
                validate_decision_binding(
                    request=request,
                    decision=decision,
                    step_id=step_id,
                    attempt_id=attempt_id,
                    goal=request.value.goal,
                    artifact_sha256=request.value.artifact_sha256,
                )
            except ValueError as error:
                raise ReportingError(
                    f"approval decision is invalid for {step_id}/{attempt_id}: {error}"
                ) from error
            if approval_decision_path(
                run_dir, step_id, attempt_id
            ) not in paths:
                raise ReportingError(
                    f"approval decision path does not match its identity: "
                    f"{step_id}/{attempt_id}"
                )
        history[(step_id, attempt_id)] = (request, decision)
    return history


def _collect_approval_artifacts(
    run_dir: Path,
    state: RunState,
) -> list[NamedArtifact]:
    history = _approval_evidence(run_dir)
    artifacts: list[NamedArtifact] = []
    for request, decision in history.values():
        artifacts.append(NamedArtifact(request.reference, request.value))
        if decision is not None:
            artifacts.append(NamedArtifact(decision.reference, decision.value))

    pending_path = run_dir / "pending_approval.json"
    decision_path = run_dir / "approval.json"
    aliases_present = any(
        path.exists() or path.is_symlink()
        for path in (pending_path, decision_path)
    )
    if state.status in _TERMINAL and aliases_present:
        raise ReportingError(
            "terminal run retains transient approval aliases"
        )
    if pending_path.exists() or pending_path.is_symlink():
        pending = _load_model(pending_path, ApprovalRequest)
        pair = history.get((pending.step_id, pending.attempt_id))
        if pair is None or pair[0].value != pending:
            raise ReportingError(
                "pending approval alias does not match immutable evidence"
            )
        artifacts.append(NamedArtifact("pending_approval.json", pending))
    if decision_path.exists() or decision_path.is_symlink():
        decision = _load_model(decision_path, ApprovalDecision)
        if decision.step_id is None or decision.attempt_id is None:
            raise ReportingError("approval decision alias has no attempt identity")
        pair = history.get((decision.step_id, decision.attempt_id))
        if pair is None or pair[1] is None or pair[1].value != decision:
            raise ReportingError(
                "approval decision alias does not match immutable evidence"
            )
        artifacts.append(NamedArtifact("approval.json", decision))
    if state.status == "AWAITING_APPROVAL":
        step = state.next_step()
        if step is None:
            raise ReportingError(
                "AWAITING_APPROVAL checkpoint has no current step"
            )
        attempt_id = f"{step.step_id}-r{state.repairs_for(step)}"
        pair = history.get((step.step_id, attempt_id))
        if pair is None:
            raise ReportingError(
                "AWAITING_APPROVAL checkpoint has no immutable request"
            )
        if pair[1] is None and not pending_path.is_file():
            raise ReportingError(
                "unresolved approval has no safe pending alias"
            )
        if pair[1] is not None and not decision_path.is_file():
            raise ReportingError(
                "resolved approval has no safe decision alias before resume"
            )
    return artifacts


def validate_terminal_evidence(
    run_dir: Path,
    state: RunState,
    ledger: list[StepRecord],
) -> None:
    """Prove that a terminal label is supported before reporting or deletion."""
    if state.plan is None or not state.plan.steps:
        raise ReportingError("terminal run has no verifiable plan")
    step_ids = [step.step_id for step in state.plan.steps]
    if len(step_ids) != len(set(step_ids)) or any(
        _SAFE_STEP_ID.fullmatch(step_id) is None
        or step_id in (".", "..")
        for step_id in step_ids
    ):
        raise ReportingError("terminal run plan has unsafe or duplicate step ids")

    if not ledger:
        raise ReportingError("terminal run has no durable ledger")
    if ledger[0].seq != 1 or state.seq != ledger[-1].seq:
        raise ReportingError(
            "terminal checkpoint sequence does not match the durable ledger"
        )
    if not any(record.phase == "plan" for record in ledger):
        raise ReportingError("terminal ledger has no plan event")
    verdicts = _validate_plan_and_verdict_history(run_dir, state, ledger)

    completed = state.completed_steps
    if len(completed) != len(set(completed)):
        raise ReportingError("terminal checkpoint repeats a completed step")
    if completed != step_ids[: len(completed)] or state.cursor != len(completed):
        raise ReportingError(
            "terminal cursor/completed steps do not match the plan prefix"
        )
    steps_by_id = {step.step_id: step for step in state.plan.steps}
    for step_id in completed:
        verify_records = [
            record
            for record in ledger
            if record.step_id == step_id and record.phase == "verify"
        ]
        complete_records = [
            record
            for record in ledger
            if record.step_id == step_id and record.phase == "complete"
        ]
        if not verify_records or not complete_records:
            raise ReportingError(
                f"completed step lacks durable verify/complete events: {step_id}"
            )
        verify_record = verify_records[-1]
        complete_record = complete_records[-1]
        expected_attempt = state.attempt_ids.get(step_id)
        verdict = verdicts.get(expected_attempt or "")
        if verdict is None:
            raise ReportingError(
                f"completed step lacks a ledger-bound verdict: {step_id}"
            )
        verdict_ref = (
            Path("attempts") / (expected_attempt or "") / "verify.json"
        ).as_posix()
        verdict_path = _regular_file(
            run_dir / "steps" / step_id / verdict_ref,
            required=True,
        )
        verdict_bytes = verdict_path.read_bytes()
        verdict_sha256 = sha256_bytes(verdict_bytes)
        alias_path = _regular_file(
            run_dir / "steps" / step_id / "verify.json",
            required=True,
        )
        if alias_path.read_bytes() != verdict_bytes:
            raise ReportingError(
                "verdict is not bound to the completed attempt alias: "
                f"{step_id}"
            )
        if verdict.step_id != step_id or not verdict.passed:
            raise ReportingError(
                f"completed step lacks a passing matching verdict: {step_id}"
            )
        if [check.name for check in verdict.checks] != steps_by_id[step_id].verifiers:
            raise ReportingError(
                f"verdict checks do not match the plan for completed step: {step_id}"
            )
        if (
            expected_attempt is None
            or verdict.attempt_id != expected_attempt
            or verify_record.attempt_id != expected_attempt
            or complete_record.attempt_id != expected_attempt
            or verify_record.verdict_ref != verdict_ref
            or verify_record.evidence_sha256 != verdict_sha256
            or complete_record.evidence_sha256 != verdict_sha256
        ):
            raise ReportingError(
                f"verdict is not bound to the completed attempt: {step_id}"
            )
        artifact_name = {
            "edit_code": "patch.json",
            "run_experiment": "experiment_evidence.json",
            "repo_stage": "repo_stage_evidence.json",
            "gather_context": "context_bundle.json",
            "answer_query": "context_bundle.json",
            "repo_integrity": "repo_integrity.json",
        }[steps_by_id[step_id].action]
        artifact_ref = (
            Path("attempts") / expected_attempt / artifact_name
        ).as_posix()
        if verdict.artifact_ref != artifact_ref:
            raise ReportingError(
                f"verdict has an unsafe artifact reference: {step_id}"
            )
        artifact_path = _regular_file(
            _safe_run_relative(
                run_dir,
                (Path("steps") / step_id / artifact_ref).as_posix(),
            ),
            required=True,
        )
        if (
            verdict.artifact_sha256 is None
            or sha256_bytes(artifact_path.read_bytes())
            != verdict.artifact_sha256
        ):
            raise ReportingError(
                f"verdict input artifact does not match: {step_id}"
            )

    transactions = [
        transaction
        for step_id in step_ids
        for transaction in list_transactions(run_dir, step_id)
    ]
    if state.status == "DONE":
        if (
            state.cursor != len(step_ids)
            or completed != step_ids
            or state.failed_steps
        ):
            raise ReportingError("DONE checkpoint does not complete the full plan")
        unfinished = [
            transaction
            for transaction in transactions
            if transaction.status in ("PREPARED", "APPLIED")
        ]
        if unfinished:
            raise ReportingError(
                "DONE run contains an unresolved patch transaction"
            )
        # A verdict proves what was checked at completion time. Retention must
        # also prove those bytes still exist now: for a path touched by several
        # attempts or steps, the last VERIFIED transaction is authoritative.
        expected_worktree = {}
        for transaction in transactions:
            if transaction.status != "VERIFIED":
                continue
            for relative in transaction.resolved_paths:
                expected_worktree[relative] = transaction.applied_state[relative]
        if expected_worktree:
            workdir = run_dir / "workdir"
            if workdir.is_symlink() or not workdir.is_dir():
                raise ReportingError(f"DONE worktree is unsafe: {workdir}")
            try:
                actual_worktree = state_for_paths(
                    workdir, sorted(expected_worktree)
                )
            except Exception as error:
                raise ReportingError(
                    f"DONE worktree could not be validated: {error}"
                ) from error
            mismatches = [
                relative
                for relative, expected in expected_worktree.items()
                if actual_worktree.get(relative) != expected
            ]
            if mismatches:
                raise ReportingError(
                    "DONE worktree does not match the last VERIFIED transaction: "
                    + ", ".join(sorted(mismatches))
                )
    else:
        unsafe = [
            transaction
            for transaction in transactions
            if transaction.status in ("PREPARED", "APPLIED")
        ]
        if unsafe:
            raise ReportingError(
                "FAILED run contains an applied or prepared patch transaction"
            )
    terminal_status: Literal["DONE", "FAILED"] = (
        "DONE" if state.status == "DONE" else "FAILED"
    )
    try:
        validate_terminal_transaction_state(
            run_dir,
            run_dir / "workdir",
            terminal_status,
        )
    except Exception as error:
        raise ReportingError(str(error)) from error


def _validate_approval_record(
    *,
    run_dir: Path,
    step,
    attempt_id: str,
    record: StepRecord,
) -> tuple[ApprovalDecision, set[Path]]:
    try:
        request = read_approval_request(
            run_dir, step.step_id, attempt_id
        )
        decision = read_approval_decision(
            run_dir, step.step_id, attempt_id
        )
    except ValueError as error:
        raise ReportingError(str(error)) from error
    if request is None or decision is None:
        raise ReportingError(
            f"approval evidence is incomplete for {attempt_id}"
        )

    artifact_sha256 = None
    if step.action == "edit_code":
        patch_path = _regular_file(
            approval_request_path(
                run_dir, step.step_id, attempt_id
            ).with_name("patch.json"),
            required=True,
        )
        artifact_sha256 = sha256_bytes(patch_path.read_bytes())
    try:
        validate_decision_binding(
            request=request,
            decision=decision,
            step_id=step.step_id,
            attempt_id=attempt_id,
            goal=step.goal,
            artifact_sha256=artifact_sha256,
        )
    except ValueError as error:
        raise ReportingError(
            f"approval evidence is misbound for {attempt_id}: {error}"
        ) from error
    if (
        record.artifact_ref != approval_decision_ref(attempt_id)
        or record.evidence_sha256 != decision.sha256
        or record.idempotency_key != f"{attempt_id}:approval"
    ):
        raise ReportingError(
            f"approval ledger event is not bound to {attempt_id}"
        )
    return decision.value, {
        approval_request_path(run_dir, step.step_id, attempt_id),
        approval_decision_path(run_dir, step.step_id, attempt_id),
    }


def _validate_plan_and_verdict_history(
    run_dir: Path,
    state: RunState,
    ledger: list[StepRecord],
) -> dict[str, Verdict]:
    """Replay immutable plan/verdict evidence without trusting mutable aliases."""
    plan_records = [record for record in ledger if record.phase == "plan"]
    if len(plan_records) != 1:
        raise ReportingError("terminal ledger must contain exactly one plan event")
    plan_record = plan_records[0]
    initial_ref = "plans/initial.json"
    initial_path = _regular_file(
        _safe_run_relative(run_dir, initial_ref), required=True
    )
    initial_bytes = initial_path.read_bytes()
    if (
        plan_record.step_id != "-"
        or plan_record.artifact_ref != initial_ref
        or plan_record.evidence_sha256 != sha256_bytes(initial_bytes)
    ):
        raise ReportingError(
            "plan event is not bound to the immutable initial plan"
        )
    initial = _load_model(initial_path, Plan)
    plan = initial.model_copy(deep=True)
    cursor = 0
    repairs: dict[str, int] = {}
    completed: list[str] = []
    failed: list[str] = []
    verdicts: dict[str, Verdict] = {}
    expected_plan_paths = {initial_path}
    expected_verdict_paths: set[Path] = set()
    expected_action_paths: set[Path] = set()
    expected_context_paths: set[Path] = set()
    expected_approval_paths: set[Path] = set()
    executed_attempts: set[str] = set()
    gated_attempts: set[str] = set()
    approvals: dict[str, ApprovalDecision] = {}

    for record in ledger:
        if record.phase == "plan":
            continue
        current = plan.steps[cursor] if cursor < len(plan.steps) else None
        if record.phase in ("verify", "repair", "complete") and current is None:
            raise ReportingError(
                f"ledger has {record.phase} after the plan completed"
            )
        if record.phase == "context":
            if current is None:
                raise ReportingError("ledger has context after the plan completed")
            attempt_id = (
                f"{current.step_id}-r{repairs.get(current.step_id, 0)}"
            )
            expected_ref = (
                Path("attempts") / attempt_id / "context_bundle.json"
            ).as_posix()
            context_path = _regular_file(
                _safe_run_relative(
                    run_dir,
                    (
                        Path("steps")
                        / current.step_id
                        / expected_ref
                    ).as_posix(),
                ),
                required=True,
            )
            if (
                record.step_id != current.step_id
                or record.attempt_id != attempt_id
                or record.artifact_ref != expected_ref
                or record.evidence_sha256
                != sha256_bytes(context_path.read_bytes())
            ):
                raise ReportingError(
                    f"context event is not bound to {attempt_id}"
                )
            _load_model(context_path, ContextBundle)
            expected_context_paths.add(context_path)
            continue
        if record.phase == "execute":
            if current is None:
                raise ReportingError("ledger has execute after the plan completed")
            attempt_id = (
                f"{current.step_id}-r{repairs.get(current.step_id, 0)}"
            )
            if (
                record.step_id != current.step_id
                or record.attempt_id != attempt_id
            ):
                raise ReportingError(
                    f"execute event identity is invalid for {attempt_id}"
                )
            action_paths = _validate_action_attempt_evidence(
                run_dir=run_dir,
                state=state,
                step=current,
                attempt_id=attempt_id,
                record=record,
            )
            expected_action_paths.update(action_paths)
            executed_attempts.add(attempt_id)
            if current.requires_approval:
                gated_attempts.add(attempt_id)
            continue
        if record.phase == "approval":
            if current is None:
                raise ReportingError(
                    "ledger has approval after the plan completed"
                )
            attempt_id = (
                f"{current.step_id}-r{repairs.get(current.step_id, 0)}"
            )
            if (
                not current.requires_approval
                or attempt_id not in executed_attempts
                or attempt_id in approvals
                or record.step_id != current.step_id
                or record.attempt_id != attempt_id
            ):
                raise ReportingError(
                    f"approval event identity is invalid for {attempt_id}"
                )
            decision, evidence_paths = _validate_approval_record(
                run_dir=run_dir,
                step=current,
                attempt_id=attempt_id,
                record=record,
            )
            approvals[attempt_id] = decision
            expected_approval_paths.update(evidence_paths)
            continue
        if record.phase == "verify":
            assert current is not None
            expected_attempt = (
                f"{current.step_id}-r{repairs.get(current.step_id, 0)}"
            )
            expected_ref = (
                Path("attempts") / expected_attempt / "verify.json"
            ).as_posix()
            if (
                record.step_id != current.step_id
                or record.attempt_id != expected_attempt
                or record.verdict_ref != expected_ref
                or record.evidence_sha256 is None
            ):
                raise ReportingError(
                    f"verify event identity is invalid for {expected_attempt}"
                )
            verdict_path = _regular_file(
                _safe_run_relative(
                    run_dir,
                    (
                        Path("steps")
                        / current.step_id
                        / expected_ref
                    ).as_posix(),
                ),
                required=True,
            )
            verdict_bytes = verdict_path.read_bytes()
            if sha256_bytes(verdict_bytes) != record.evidence_sha256:
                raise ReportingError(
                    f"immutable verdict changed for {expected_attempt}"
                )
            verdict = _load_model(verdict_path, Verdict)
            names = [check.name for check in verdict.checks]
            policy_names = (
                ["oracle-policy"]
                if record.idempotency_key == f"{expected_attempt}:policy"
                else list(current.verifiers)
            )
            if (
                current.requires_approval
                and policy_names != ["oracle-policy"]
                and (
                    expected_attempt not in approvals
                    or not approvals[expected_attempt].approved
                )
            ):
                raise ReportingError(
                    f"attempt was verified without approval: {expected_attempt}"
                )
            if (
                verdict.step_id != current.step_id
                or verdict.attempt_id != expected_attempt
                or names != policy_names
            ):
                raise ReportingError(
                    f"immutable verdict identity is invalid for {expected_attempt}"
                )
            artifact_name = {
                "edit_code": "patch.json",
                "run_experiment": "experiment_evidence.json",
                "repo_stage": "repo_stage_evidence.json",
                "gather_context": "context_bundle.json",
                "answer_query": "context_bundle.json",
                "repo_integrity": "repo_integrity.json",
            }[current.action]
            artifact_ref = (
                Path("attempts") / expected_attempt / artifact_name
            ).as_posix()
            artifact_path = _regular_file(
                _safe_run_relative(
                    run_dir,
                    (
                        Path("steps")
                        / current.step_id
                        / artifact_ref
                    ).as_posix(),
                ),
                required=True,
            )
            if (
                verdict.artifact_ref != artifact_ref
                or verdict.artifact_sha256
                != sha256_bytes(artifact_path.read_bytes())
            ):
                raise ReportingError(
                    f"verdict input artifact changed for {expected_attempt}"
                )
            verifier_stage_paths = _validate_verifier_stage_evidence(
                run_dir=run_dir,
                step=current,
                attempt_id=expected_attempt,
                verdict=verdict,
                validate_alias=not any(
                    candidate.phase == "verify"
                    and candidate.step_id == current.step_id
                    and candidate.seq > record.seq
                    for candidate in ledger
                ),
            )
            expected_action_paths.update(verifier_stage_paths)
            verdicts[expected_attempt] = verdict
            expected_verdict_paths.add(verdict_path)
            continue
        if record.phase == "repair":
            assert current is not None
            attempt_id = (
                f"{current.step_id}-r{repairs.get(current.step_id, 0)}"
            )
            verdict = verdicts.get(attempt_id)
            expected_ref = f"plans/{attempt_id}-repair.json"
            repair_path = _regular_file(
                _safe_run_relative(run_dir, expected_ref), required=True
            )
            repair_bytes = repair_path.read_bytes()
            if (
                record.step_id != current.step_id
                or record.attempt_id != attempt_id
                or record.artifact_ref != expected_ref
                or record.evidence_sha256 != sha256_bytes(repair_bytes)
                or verdict is None
                or verdict.passed
            ):
                raise ReportingError(
                    f"repair event is not bound to a failing verdict: {attempt_id}"
                )
            expected = plan.model_copy(deep=True)
            expected.steps[cursor] = current.as_repair(verdict.failures)
            repaired = _load_model(repair_path, Plan)
            if repaired != expected:
                raise ReportingError(
                    f"immutable repair plan is invalid for {attempt_id}"
                )
            plan = repaired
            repairs[current.step_id] = repairs.get(current.step_id, 0) + 1
            expected_plan_paths.add(repair_path)
            continue
        if record.phase == "complete":
            assert current is not None
            attempt_id = (
                f"{current.step_id}-r{repairs.get(current.step_id, 0)}"
            )
            verdict = verdicts.get(attempt_id)
            if (
                record.step_id != current.step_id
                or record.attempt_id != attempt_id
                or verdict is None
                or not verdict.passed
                or record.evidence_sha256
                != next(
                    (
                        candidate.evidence_sha256
                        for candidate in ledger
                        if candidate.phase == "verify"
                        and candidate.attempt_id == attempt_id
                    ),
                    None,
                )
            ):
                raise ReportingError(
                    f"complete event is not bound to a passing verdict: {attempt_id}"
                )
            completed.append(current.step_id)
            cursor += 1
            continue
        if record.phase == "fail" and current is not None:
            if record.step_id != current.step_id:
                raise ReportingError("fail event does not match the current plan step")
            if failed:
                raise ReportingError("ledger contains multiple run failures")
            if record.attempt_id:
                verdict = verdicts.get(record.attempt_id)
                if verdict is not None and verdict.passed:
                    raise ReportingError("passing verdict is followed by fail")
                decision = approvals.get(record.attempt_id)
                if verdict is None and decision is not None and decision.approved:
                    raise ReportingError(
                        "approved attempt failed without a verifier verdict"
                    )
            failed.append(current.step_id)

    plan_alias = _regular_file(run_dir / "plan.json", required=True)
    if _load_model(plan_alias, Plan) != plan or state.plan != plan:
        raise ReportingError(
            "plan alias or RunState does not match immutable plan history"
        )
    if (
        state.cursor != cursor
        or state.completed_steps != completed
        or state.repairs != repairs
        or state.failed_steps != failed
    ):
        raise ReportingError(
            "terminal checkpoint progress does not match immutable history"
        )

    plans_root = run_dir / "plans"
    actual_plan_paths = (
        set(plans_root.rglob("*.json")) if plans_root.is_dir() else set()
    )
    if actual_plan_paths != expected_plan_paths:
        raise ReportingError("plan snapshot set does not match the ledger")
    attempts_root = run_dir / "steps"
    actual_verdict_paths = {
        path
        for path in attempts_root.rglob("verify.json")
        if "attempts" in path.parts
    }
    if actual_verdict_paths != expected_verdict_paths:
        raise ReportingError("immutable verdict set does not match the ledger")
    actual_action_paths = {
        path
        for name in (
            "experiment_intent.json",
            "experiment_evidence.json",
            "repo_stage_intent.json",
            "repo_stage_evidence.json",
        )
        for path in attempts_root.rglob(name)
    }
    if actual_action_paths != expected_action_paths:
        raise ReportingError("action attempt evidence set does not match the ledger")
    actual_context_paths = {
        path
        for path in attempts_root.rglob("context_bundle.json")
        if "attempts" in path.parts
    }
    if actual_context_paths != expected_context_paths:
        raise ReportingError("context snapshot set does not match the ledger")
    if set(approvals) != gated_attempts:
        raise ReportingError(
            "approval decisions do not match approval-gated executions"
        )
    actual_approval_paths = {
        path
        for name in ("approval_request.json", "approval_decision.json")
        for path in attempts_root.rglob(name)
    }
    if actual_approval_paths != expected_approval_paths:
        raise ReportingError(
            "approval evidence set does not match the ledger"
        )
    return verdicts


def validate_llm_attempt_evidence(
    run_dir: Path,
    state: RunState,
    ledger: list[StepRecord],
    trace: list[dict[str, Any]],
) -> None:
    """Validate every paid plan/patch call journal and reject orphaned entries."""
    root = run_dir / "llm_attempts"
    durable_usage = load_usage_checkpoint(run_dir)
    expected_calls = max(
        state.llm_usage.calls,
        durable_usage.calls if durable_usage is not None else 0,
    )
    attempt_patch_paths = {
        path
        for path in (run_dir / "steps").rglob("patch.json")
        if "attempts" in path.parts
    }
    trace_kinds = [
        record.get("kind")
        for record in trace
        if record.get("kind") in {"plan", "propose_patch"}
    ]
    if not root.exists() and not root.is_symlink():
        if attempt_patch_paths or trace_kinds:
            raise ReportingError(
                "model-authored evidence exists but the LLM attempt journal is missing"
            )
        return
    if root.is_symlink() or not root.is_dir():
        raise ReportingError(f"LLM attempt journal root is unsafe: {root}")
    kind_dirs = sorted(root.iterdir())
    if any(
        path.is_symlink()
        or not path.is_dir()
        or path.name not in {"plan", "propose_patch"}
        for path in kind_dirs
    ):
        raise ReportingError(
            "LLM attempt journal contains an unknown call-kind path"
        )
    attempt_dirs = sorted(
        attempt
        for kind_dir in kind_dirs
        for attempt in kind_dir.iterdir()
    )
    if len(attempt_dirs) > expected_calls:
        raise ReportingError(
            "LLM attempt journal exceeds the recorded call count"
        )
    completed_kinds: list[str] = []
    journaled_patch_paths: set[Path] = set()
    seen_plan = 0
    ledger_attempts = {
        record.attempt_id for record in ledger if record.attempt_id
    }
    for directory in attempt_dirs:
        if directory.is_symlink() or not directory.is_dir():
            raise ReportingError(
                f"LLM attempt journal entry is unsafe: {directory}"
            )
        files = {path.name for path in directory.iterdir()}
        if not files <= {"intent.json", "result.json"} or "intent.json" not in files:
            raise ReportingError(
                f"LLM attempt journal has unknown or missing files: {directory}"
            )
        intent_path = _regular_file(directory / "intent.json", required=True)
        try:
            intent = json.loads(intent_path.read_text())
            input_payload = intent["input"]
            input_sha256 = hashlib.sha256(
                json.dumps(
                    input_payload, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        except Exception as error:
            raise ReportingError(
                f"invalid LLM intent {intent_path}: {error}"
            ) from error
        kind = intent.get("kind")
        logical_attempt = intent.get("logical_attempt_id")
        if (
            intent.get("schema_version") != 1
            or kind not in {"plan", "propose_patch"}
            or intent.get("input_sha256") != input_sha256
            or directory.parent.name != kind
            or logical_attempt != directory.name
            or not isinstance(input_payload, dict)
        ):
            raise ReportingError(f"LLM intent identity is invalid: {intent_path}")
        context = input_payload.get("context")
        if (
            not isinstance(context, dict)
            or context.get("run_id") != state.run_id
            or context.get("task") != state.task.model_dump(mode="json")
            or not isinstance(context.get("config"), dict)
            or context.get("attempt_id") != logical_attempt
        ):
            raise ReportingError(
                f"LLM intent is not bound to this run: {intent_path}"
            )

        result_path = directory / "result.json"
        if not result_path.exists() and not result_path.is_symlink():
            if state.status != "FAILED":
                raise ReportingError(
                    f"non-failed run has an ambiguous LLM call: {directory}"
                )
            continue
        result_path = _regular_file(result_path, required=True)
        try:
            result_envelope = json.loads(result_path.read_text())
            result = result_envelope["result"]
            result_sha256 = hashlib.sha256(
                json.dumps(
                    result, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        except Exception as error:
            raise ReportingError(
                f"invalid LLM result {result_path}: {error}"
            ) from error
        if (
            result_envelope.get("schema_version") != 1
            or result_envelope.get("kind") != kind
            or result_envelope.get("input_sha256") != input_sha256
            or result_envelope.get("result_sha256") != result_sha256
            or not isinstance(result, dict)
        ):
            raise ReportingError(
                f"LLM result identity or checksum is invalid: {result_path}"
            )
        completed_kinds.append(kind)

        if kind == "plan":
            seen_plan += 1
            if seen_plan > 1:
                raise ReportingError("run contains multiple LLM plan attempts")
            try:
                from .agents.supervisor import Supervisor
                from .tasks.spec import TaskSpec

                task = TaskSpec.model_validate(input_payload["task"])
                template = Plan.model_validate(input_payload["template"])
                if task != state.task:
                    raise ValueError("task mismatch")
                if result.get("type") == "None" and result.get("value") is None:
                    selected = template
                elif result.get("type") == "Plan":
                    candidate = Plan.model_validate(result.get("value"))
                    selected = (
                        candidate
                        if Supervisor._valid(
                            candidate, task=task, template=template
                        )
                        else template
                    )
                else:
                    raise ValueError("unexpected result type")
                initial = _load_model(run_dir / "plans" / "initial.json", Plan)
            except Exception as error:
                raise ReportingError(
                    f"invalid journaled plan result: {error}"
                ) from error
            if selected != initial:
                raise ReportingError(
                    "journaled plan result does not match the initial plan"
                )
            continue

        try:
            step_payload = input_payload["step"]
            step_id = str(step_payload["step_id"])
            attempt_id = str(context["attempt_id"])
            patch = Patch.model_validate(result.get("value"))
        except Exception as error:
            raise ReportingError(
                f"invalid journaled patch result: {error}"
            ) from error
        if (
            result.get("type") != "Patch"
            or attempt_id not in ledger_attempts
            or patch.step_id != step_id
        ):
            raise ReportingError(
                f"journaled patch call is orphaned: {directory}"
            )
        proposal_path = _regular_file(
            run_dir
            / "steps"
            / step_id
            / "attempts"
            / attempt_id
            / "patch.json",
            required=True,
        )
        if _load_model(proposal_path, Patch) != patch:
            raise ReportingError(
                f"journaled patch does not match the attempt artifact: {attempt_id}"
            )
        journaled_patch_paths.add(proposal_path)

    # Tracing is intentionally best-effort, but any rows that did persist must
    # not claim more completed plan/patch calls than the durable journals.
    remaining = list(completed_kinds)
    for kind in trace_kinds:
        if kind not in remaining:
            raise ReportingError(
                "LLM trace contains a call absent from the durable journal"
            )
        remaining.remove(kind)
    if journaled_patch_paths != attempt_patch_paths:
        raise ReportingError(
            "model-authored patch attempts do not match the LLM journal"
        )


def _validate_verifier_stage_evidence(
    *,
    run_dir: Path,
    step,
    attempt_id: str,
    verdict: Verdict,
    validate_alias: bool,
) -> set[Path]:
    attempt_dir = (
        run_dir / "steps" / step.step_id / "attempts" / attempt_id
    )
    intent_path = attempt_dir / "repo_stage_intent.json"
    evidence_path = attempt_dir / "repo_stage_evidence.json"
    present = any(
        path.exists() or path.is_symlink()
        for path in (intent_path, evidence_path)
    )
    if not present or step.action == "repo_stage":
        return set()
    intent_path = _regular_file(intent_path, required=True)
    evidence_path = _regular_file(evidence_path, required=True)
    intent = _read_enveloped_model(intent_path, RepoStageIntent)
    evidence = _read_enveloped_model(evidence_path, RepoStageEvidence)
    try:
        spec = RepoAdapterSpec.model_validate(step.params["repo_adapter_spec"])
        spec_sha256 = hashlib.sha256(
            json.dumps(
                spec.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        stage = step.params["repo_stage"]
        check = next(
            item for item in verdict.checks if item.name == "repo-targeted"
        )
        recorded_result = check.detail["result"]
    except Exception as error:
        raise ReportingError(
            f"repository verifier evidence is orphaned for {attempt_id}: {error}"
        ) from error
    if (
        evidence.intent != intent
        or intent.step_id != step.step_id
        or intent.attempt_id != attempt_id
        or intent.stage != stage
        or intent.spec_sha256 != spec_sha256
        or evidence.result.model_dump(mode="json") != recorded_result
    ):
        raise ReportingError(
            f"repository verifier evidence is invalid for {attempt_id}"
        )
    from .repo_adapter import RepoStageResult

    if validate_alias:
        alias = _load_model(
            run_dir / "steps" / step.step_id / "repo_stage.json",
            RepoStageResult,
        )
        if alias != evidence.result:
            raise ReportingError(
                f"repository verifier alias does not match {attempt_id}"
            )
    return {intent_path, evidence_path}


def _validate_action_attempt_evidence(
    *,
    run_dir: Path,
    state: RunState,
    step,
    attempt_id: str,
    record: StepRecord,
) -> set[Path]:
    attempt_dir = (
        run_dir / "steps" / step.step_id / "attempts" / attempt_id
    )
    if step.action == "run_experiment":
        intent_path = _regular_file(
            attempt_dir / "experiment_intent.json", required=True
        )
        evidence_path = _regular_file(
            attempt_dir / "experiment_evidence.json", required=True
        )
        expected_ref = (
            Path("attempts") / attempt_id / "experiment_evidence.json"
        ).as_posix()
        if (
            record.artifact_ref != expected_ref
            or record.evidence_sha256
            != sha256_bytes(evidence_path.read_bytes())
        ):
            raise ReportingError(
                f"execute event is not bound to experiment evidence: {attempt_id}"
            )
        intent = _read_enveloped_model(intent_path, ExperimentIntent)
        evidence = _read_enveloped_model(evidence_path, ExperimentEvidence)
        params_sha256 = hashlib.sha256(
            json.dumps(
                step.params, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if (
            evidence.intent != intent
            or intent.step_id != step.step_id
            or intent.attempt_id != attempt_id
            or intent.command != tuple(build_cmd(step))
            or intent.params_sha256 != params_sha256
        ):
            raise ReportingError(
                f"experiment evidence identity is invalid: {attempt_id}"
            )
        if state.attempt_ids.get(step.step_id) == attempt_id:
            alias = _load_model(
                run_dir / "steps" / step.step_id / "experiment.json",
                ExperimentResult,
            )
            if alias != evidence.result:
                raise ReportingError(
                    f"experiment alias does not match evidence: {attempt_id}"
                )
            if state.status == "DONE" and step.step_id in state.completed_steps:
                try:
                    validate_experiment_result(
                        run_dir / "workdir", evidence.result
                    )
                except Exception as error:
                    raise ReportingError(
                        f"experiment outputs changed for {attempt_id}: {error}"
                    ) from error
        return {intent_path, evidence_path}

    if step.action == "repo_stage":
        intent_path = _regular_file(
            attempt_dir / "repo_stage_intent.json", required=True
        )
        evidence_path = _regular_file(
            attempt_dir / "repo_stage_evidence.json", required=True
        )
        expected_ref = (
            Path("attempts") / attempt_id / "repo_stage_evidence.json"
        ).as_posix()
        if (
            record.artifact_ref != expected_ref
            or record.evidence_sha256
            != sha256_bytes(evidence_path.read_bytes())
        ):
            raise ReportingError(
                f"execute event is not bound to repository stage evidence: {attempt_id}"
            )
        intent = _read_enveloped_model(intent_path, RepoStageIntent)
        evidence = _read_enveloped_model(evidence_path, RepoStageEvidence)
        try:
            spec = RepoAdapterSpec.model_validate(
                step.params["repo_adapter_spec"]
            )
            spec_sha256 = hashlib.sha256(
                json.dumps(
                    spec.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            stage = step.params["repo_stage"]
        except Exception as error:
            raise ReportingError(
                f"invalid repository stage plan for {attempt_id}: {error}"
            ) from error
        if (
            evidence.intent != intent
            or intent.step_id != step.step_id
            or intent.attempt_id != attempt_id
            or intent.stage != stage
            or intent.spec_sha256 != spec_sha256
        ):
            raise ReportingError(
                f"repository stage evidence identity is invalid: {attempt_id}"
            )
        if state.attempt_ids.get(step.step_id) == attempt_id:
            from .repo_adapter import RepoStageResult

            alias = _load_model(
                run_dir / "steps" / step.step_id / "repo_stage.json",
                RepoStageResult,
            )
            if alias != evidence.result:
                raise ReportingError(
                    f"repository stage alias does not match evidence: {attempt_id}"
                )
        return {intent_path, evidence_path}

    if record.evidence_sha256 is not None:
        artifact = _regular_file(
            run_dir / "steps" / step.step_id / (record.artifact_ref or ""),
            required=True,
        )
        if sha256_bytes(artifact.read_bytes()) != record.evidence_sha256:
            raise ReportingError(
                f"execute artifact changed for {attempt_id}"
            )
    return set()


def _read_enveloped_model(path: Path, model_type):
    path = _regular_file(path, required=True)
    try:
        raw = json.loads(path.read_text())
        payload = raw["payload"]
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if raw.get("schema_version") != 1 or raw.get("sha256") != digest:
            raise ValueError("checksum mismatch")
        return model_type.model_validate(payload)
    except Exception as error:
        raise ReportingError(
            f"invalid checksummed evidence {path}: {error}"
        ) from error


def validate_recovery_evidence(run_dir: Path) -> None:
    """Validate transaction, backup, and manifest evidence before retention."""
    state = load_state(run_dir)
    try:
        validate_transaction_journals(run_dir)
    except Exception as error:
        raise ReportingError(str(error)) from error
    transaction_root = run_dir / "transactions"
    expected_logs: set[Path] = set()
    latest_reviews: dict[str, tuple[int, bytes]] = {}
    if transaction_root.exists() or transaction_root.is_symlink():
        if transaction_root.is_symlink() or not transaction_root.is_dir():
            raise ReportingError(f"transaction directory is unsafe: {transaction_root}")
        for descendant in transaction_root.rglob("*"):
            if descendant.is_symlink():
                raise ReportingError(f"refusing symlink transaction evidence: {descendant}")
            if descendant.is_dir():
                continue
            if descendant.name.endswith(".events.jsonl"):
                continue
            if descendant.suffix != ".json":
                raise ReportingError(f"unknown transaction evidence: {descendant}")
            try:
                envelope = json.loads(descendant.read_text())
                payload = envelope["payload"]
                canonical = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode()
                digest = sha256_bytes(canonical)
                if digest != envelope["sha256"]:
                    raise ValueError("checksum mismatch")
                transaction = PatchTransaction.model_validate(payload)
            except Exception as error:
                raise ReportingError(
                    f"invalid patch transaction {descendant}: {error}"
                ) from error
            expected_path = transaction_path(
                run_dir, transaction.step_id, transaction.attempt_id
            )
            if descendant != expected_path:
                raise ReportingError(
                    f"transaction path does not match its identity: {descendant}"
                )
            log_path = transaction_log_path(
                run_dir, transaction.step_id, transaction.attempt_id
            )
            expected_logs.add(log_path)
            try:
                events = read_transaction_events(
                    run_dir, transaction.step_id, transaction.attempt_id
                )
            except Exception as error:
                raise ReportingError(str(error)) from error
            if (
                not events
                or events[-1].status != transaction.status
                or events[-1].transaction_sha256 != digest
            ):
                raise ReportingError(
                    f"transaction log does not end at the persisted state: {descendant}"
                )
            backups = []
            for reference in (transaction.backup_ref, transaction.backup_mirror_ref):
                backup_path = _safe_run_relative(run_dir, reference)
                try:
                    backup = load_backup(backup_path, required=True)
                    assert backup is not None
                except Exception as error:
                    raise ReportingError(str(error)) from error
                if backup_sha256(backup) != transaction.backup_sha256:
                    raise ReportingError(
                        f"backup digest does not match transaction: {backup_path}"
                    )
                backups.append(backup)
            attempt_dir = attempt_artifact_dir(
                run_dir, transaction.step_id, transaction.attempt_id
            )
            patch_path = _regular_file(attempt_dir / "patch.json", required=True)
            manifest_path = _regular_file(
                attempt_dir / "manifest.json", required=True
            )
            patch_bytes = patch_path.read_bytes()
            if sha256_bytes(patch_bytes) != transaction.patch_sha256:
                raise ReportingError(
                    f"transaction patch digest does not match {patch_path}"
                )
            try:
                patch = Patch.model_validate_json(patch_bytes)
                if patch.step_id != transaction.step_id:
                    raise ValueError(
                        "patch step identity does not match its transaction"
                    )
                resolved = resolve_patch(patch, patch_bytes=patch_bytes)
                manifest = ArtifactManifest.model_validate_json(
                    manifest_path.read_bytes()
                )
                backup = backups[0]
                expected_review = render_review_diff(
                    patch, resolved, backup
                ).encode("utf-8")
                expected_base = {
                    rel: saved_file_state(
                        backup.originals[rel], backup.modes[rel]
                    )
                    for rel in transaction.resolved_paths
                }
            except Exception as error:
                raise ReportingError(
                    f"invalid transaction artifacts for {transaction.step_id}/"
                    f"{transaction.attempt_id}: {error}"
                ) from error
            if resolved.paths != transaction.resolved_paths:
                raise ReportingError(
                    f"transaction write set does not match {patch_path}"
                )
            review_path = _regular_file(
                attempt_dir / "review.diff", required=True
            )
            if review_path.read_bytes() != expected_review:
                raise ReportingError(
                    f"review diff does not match transaction: {review_path}"
                )
            current_review = latest_reviews.get(transaction.step_id)
            transaction_sequence = int(transaction.sequence or 0)
            if (
                current_review is None
                or transaction_sequence > current_review[0]
            ):
                latest_reviews[transaction.step_id] = (
                    transaction_sequence,
                    expected_review,
                )
            if (
                manifest.step_id != transaction.step_id
                or manifest.artifact_sha256 != transaction.patch_sha256
                or manifest.touched_files != transaction.resolved_paths
                or manifest.base_state != expected_base
            ):
                raise ReportingError(
                    f"manifest does not match transaction: {manifest_path}"
                )
            step = next(
                (
                    candidate
                    for candidate in (state.plan.steps if state.plan else [])
                    if candidate.step_id == transaction.step_id
                ),
                None,
            )
            if step is None:
                raise ReportingError(
                    f"transaction step is absent from the run plan: "
                    f"{transaction.step_id}"
                )
            if (
                manifest.verifiers != list(step.verifiers)
                or manifest.policy_overrides
                != list(state.task.allowed_protected_files)
            ):
                raise ReportingError(
                    f"manifest policy does not match the run: {manifest_path}"
                )
        actual_logs = set(transaction_root.rglob("*.events.jsonl"))
        orphaned = actual_logs - expected_logs
        if orphaned:
            raise ReportingError(
                f"orphaned transaction log: {sorted(orphaned)[0]}"
            )
        for step_id, (_sequence, expected_review) in latest_reviews.items():
            alias = _regular_file(
                run_dir / "steps" / step_id / "patch.diff",
                required=True,
            )
            if alias.read_bytes() != expected_review:
                raise ReportingError(
                    f"review diff alias does not match the latest attempt: {alias}"
                )

    backup_root = run_dir / "backups"
    if backup_root.exists() or backup_root.is_symlink():
        if backup_root.is_symlink() or not backup_root.is_dir():
            raise ReportingError(f"backup directory is unsafe: {backup_root}")
        for descendant in backup_root.rglob("*"):
            if descendant.is_symlink():
                raise ReportingError(f"refusing symlink backup evidence: {descendant}")
            if descendant.is_dir():
                continue
            if descendant.suffix != ".json":
                raise ReportingError(f"unknown backup evidence: {descendant}")
            try:
                load_backup(descendant, required=True)
            except Exception as error:
                raise ReportingError(str(error)) from error

    for manifest_path in _artifact_paths(run_dir, "manifest.json"):
        manifest = _load_model(manifest_path, ArtifactManifest)
        patch_path = manifest_path.with_name("patch.json")
        if not patch_path.exists():
            raise ReportingError(
                f"manifest has no sibling patch artifact: {manifest_path}"
            )
        patch_bytes = _regular_file(patch_path, required=True).read_bytes()
        if sha256_bytes(patch_bytes) != manifest.artifact_sha256:
            raise ReportingError(
                f"manifest artifact hash does not match {patch_path}"
            )
        try:
            patch = Patch.model_validate_json(patch_bytes)
            if patch.step_id != manifest.step_id:
                raise ValueError(
                    "patch step identity does not match its manifest"
                )
            resolved = resolve_patch(patch, patch_bytes=patch_bytes)
        except Exception as error:
            raise ReportingError(f"invalid patch beside {manifest_path}: {error}") from error
        if resolved.paths != manifest.touched_files:
            raise ReportingError(
                f"manifest write set does not match {patch_path}"
            )


def _safe_run_relative(run_dir: Path, value: str) -> Path:
    relative = Path(value)
    if (
        not value
        or "\\" in value
        or relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ReportingError(f"unsafe run evidence path: {value!r}")
    path = run_dir / relative
    try:
        path.relative_to(run_dir)
    except ValueError as error:
        raise ReportingError(f"run evidence path escaped: {value!r}") from error
    probe = run_dir
    for part in relative.parts:
        probe /= part
        if probe.is_symlink():
            raise ReportingError(
                f"run evidence path contains a symlink: {value!r}"
            )
    return path


def _artifact_paths(run_dir: Path, name: str) -> list[Path]:
    paths: list[Path] = []
    flat = run_dir / name
    if flat.exists() or flat.is_symlink():
        paths.append(_regular_file(flat, required=True))
    steps = run_dir / "steps"
    if steps.exists():
        if steps.is_symlink() or not steps.is_dir():
            raise ReportingError(f"artifact directory is unsafe: {steps}")
        try:
            descendants = sorted(steps.rglob("*"))
        except OSError as e:
            raise ReportingError(f"cannot inspect artifact directory {steps}: {e}") from e
        for descendant in descendants:
            if descendant.is_symlink():
                raise ReportingError(f"refusing symlink in artifact directory: {descendant}")
        for path in (path for path in descendants if path.name == name):
            paths.append(_regular_file(path, required=True))
    return paths


def _load_model(path: Path, model_type):
    path = _regular_file(path, required=True)
    try:
        return model_type.model_validate_json(path.read_text())
    except Exception as e:
        raise ReportingError(f"invalid {path.name} artifact {path}: {e}") from e


def _load_json_object(path: Path) -> dict[str, Any]:
    path = _regular_file(path, required=True)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ReportingError(f"invalid JSON artifact {path}: {e}") from e
    if not isinstance(value, dict):
        raise ReportingError(f"JSON artifact {path} must contain an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() and not path.is_symlink():
        return []
    path = _regular_file(path, required=True)
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        raise ReportingError(f"cannot read LLM trace {path}: {e}") from e
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as e:
            raise ReportingError(f"LLM trace {path} line {line_number} is corrupt: {e}") from e
        if not isinstance(value, dict):
            raise ReportingError(f"LLM trace {path} line {line_number} is not an object")
        records.append(value)
    return records


def _regular_file(path: Path, *, required: bool) -> Path:
    if path.is_symlink():
        raise ReportingError(f"refusing symlink artifact {path}")
    if not path.exists():
        if required:
            raise ReportingError(f"required run artifact is missing: {path}")
        return path
    if not path.is_file():
        raise ReportingError(f"run artifact is not a regular file: {path}")
    return path


def _resolve_run_dir(runs_dir: str | Path, run_id: str) -> Path:
    if not run_id or "\x00" in run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ReportingError(f"invalid run id {run_id!r}; expected one path segment")
    root = _runs_root(runs_dir)
    run_dir = root / run_id
    if run_dir.is_symlink():
        raise ReportingError(f"run directory is a symlink: {run_dir}")
    if not run_dir.exists() or not run_dir.is_dir():
        raise ReportingError(f"run not found: {run_id}")
    return run_dir


def _runs_root(runs_dir: str | Path) -> Path:
    return Path(runs_dir).expanduser().resolve()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as e:  # defensive: rglob must never escape the run
        raise ReportingError(f"artifact escaped run directory: {path}") from e


def _mtime(path: Path) -> datetime:
    try:
        return datetime.fromtimestamp(path.lstat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _model_json(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else value


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _int_field(value: Any, label: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportingError(f"{label} must be a non-negative integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ReportingError(f"{label} must be a non-negative integer")
    return int(numeric)


def _float_field(value: Any, label: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportingError(f"{label} must be a finite non-negative number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ReportingError(f"{label} must be a finite non-negative number")
    return numeric
