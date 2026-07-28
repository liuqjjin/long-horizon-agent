"""Command-line interface for running and checking LHA tasks.

lha run <task.yaml>      run a task and its configured checks
lha resume <run_id>      resume a paused/awaiting run
lha eval [--quick]       run the six repository regression workflows
lha ablate [tasks...]    compare direct acceptance, checks, and repair
lha ablation-attempt ... register or close a one-shot formal ablation
lha horizon              compose measured task outcomes across task counts
lha batch <task>...      run multiple tasks in parallel (process-isolated)
lha trace <run_id>       render a run's ledger timeline
lha index <path>         (re)build the code index for a repo
lha index-docs           (re)build paper/experiment/skill indexes via CocoIndex
lha ask <query...>       retrieve fresh context with source locations
lha runs list|show|prune inspect and safely retain persisted runs
lha approve|reject <run_id>   resolve a pending human-approval gate

Global: --llm {stub,claude_cli,codex_cli,anthropic}, -v/--verbose, --version.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from . import __version__, live_context
from .config import Config
from .harness import Harness, HumanApprovalGate, load_state_by_id
from .harness.checkpoint import run_lock
from .harness.errors import RunLocked
from .tasks.spec import TaskSpec

_EPILOG = """\
examples:
  lha run data/tasks/fix_average.yaml            fix a bug, verified by real pytest
  lha eval                                       run the six repository workflows
  lha run --runtime langgraph <task>             durable run with an approval gate
  lha ask "how is average computed" --kinds code retrieve context with source locations
  lha trace <run_id> --html                      write a self-contained run report
  lha runs prune --older-than-days 30            dry-run terminal-run retention
"""


def _config(args) -> Config:
    cfg = Config.from_env()
    if getattr(args, "llm", None):
        cfg.llm_backend = args.llm
    return cfg


def _make_runner(args, cfg: Config):
    if getattr(args, "runtime", "loop") == "langgraph":
        from .runtime.langgraph_runner import LangGraphHarness

        return LangGraphHarness(cfg, auto_approve=args.auto_approve)
    return Harness(cfg, auto_approve=args.auto_approve)


def _cmd_run(args) -> int:
    cfg = _config(args)
    task = TaskSpec.from_file(args.task)
    result = _make_runner(args, cfg).run(task)
    return _emit(result, args)


def _cmd_resume(args) -> int:
    cfg = _config(args)
    result = _make_runner(args, cfg).resume(args.run_id)
    return _emit(result, args)


def _cmd_batch(args) -> int:
    import json

    from .orchestrator import run_tasks

    outcomes = run_tasks(args.tasks, llm=getattr(args, "llm", None), max_workers=args.workers)
    print(f"ran {len(outcomes)} task(s) across {args.workers} worker(s):")
    for o in outcomes:
        print(f"  - {o.task}: status={o.status} verified={o.verified} run_id={o.run_id}")
    Path(_config(args).runs_dir).mkdir(parents=True, exist_ok=True)
    report = Path(_config(args).runs_dir) / "batch_report.json"
    report.write_text(json.dumps([o.__dict__ for o in outcomes], indent=2))
    print(f"report: {report}")
    return 0 if all(o.status == "DONE" for o in outcomes) else 1


def _cmd_eval(args) -> int:
    import json

    from .eval import run_eval

    cfg = _config(args)
    report = run_eval(cfg, quick=args.quick)
    print(report.to_markdown())
    print(f"score: {report.score}")
    out = Path(cfg.runs_dir) / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"score": report.score, "results": [r.__dict__ for r in report.results]}, indent=2
        )
    )
    print(f"report: {out}")
    return 0 if report.all_passed else 1


def _cmd_ablate(args) -> int:
    import glob

    from .ablation import run_ablation

    cfg = _config(args)
    tasks = args.tasks or sorted(glob.glob("data/tasks/bench_*.yaml"))
    if not tasks:
        print("no tasks (pass paths or add data/tasks/bench_*.yaml)")
        return 1
    llm = getattr(args, "llm", None) or "codex_cli"
    out = Path(args.out) if args.out else Path(cfg.runs_dir) / "ablation"
    model = args.model or None
    print(
        f"verification ablation: {len(tasks)} task(s) x 3 conditions x {args.reps} rep(s), "
        f"llm={llm} model={model or '(default)'}"
    )
    report = run_ablation(
        cfg,
        tasks,
        llm=llm,
        model=model,
        reps=args.reps,
        out_dir=out,
        scorer_backend=args.scorer_backend,
    )
    print()
    print(report.to_markdown())
    print(f"report: {out / 'ablation_report.md'}")
    return 1 if any(record.status == "ERROR" for record in report.records) else 0


def _emit_attempt_result(result: dict, *, as_json: bool) -> int:
    import json

    if as_json:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    for key, value in result.items():
        if value is not None:
            print(f"{key}: {value}")
    return 0


def _cmd_ablation_attempt_register(args) -> int:
    from .formal_attempt_cli import register_formal_attempt

    result = register_formal_attempt(
        repo_root=Path.cwd(),
        config=_config(args),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        docker_image_id=args.docker_image_id,
        witness_remote_name=args.witness_remote,
    )
    return _emit_attempt_result(result, as_json=args.json)


def _cmd_ablation_attempt_status(args) -> int:
    from .formal_attempt_cli import formal_attempt_status

    return _emit_attempt_result(
        formal_attempt_status(repo_root=Path.cwd()),
        as_json=args.json,
    )


def _cmd_ablation_attempt_complete(args) -> int:
    from .formal_attempt_cli import complete_formal_attempt

    return _emit_attempt_result(
        complete_formal_attempt(repo_root=Path.cwd()),
        as_json=args.json,
    )


def _cmd_ablation_attempt_abandon(args) -> int:
    from .formal_attempt_cli import abandon_formal_attempt

    result = abandon_formal_attempt(
        repo_root=Path.cwd(),
        reason_code=args.reason_code,
        reason=args.reason,
    )
    return _emit_attempt_result(result, as_json=args.json)


def _cmd_horizon(args) -> int:
    from .horizon import HorizonDataError, run_horizon

    cfg = _config(args)
    out = Path(args.out) if args.out else Path(cfg.runs_dir) / "horizon"
    try:
        report = run_horizon(args.from_report, out, seed=args.seed)
    except HorizonDataError as e:
        print(f"cannot build the horizon analysis: {e}")
        return 1
    print(report.to_markdown())
    print(f"report: {out / 'horizon_report.md'}")
    return 0


def _cmd_index(args) -> int:
    live_context.configure(code_root=args.path, config=_config(args))
    result = live_context.index_code(args.path)
    if not result.ok:
        print(f"index FAILED for {args.path}: {result.detail}")
        return 1
    try:
        hits = live_context.search_code("function", k=1)
    except live_context.BackendUnavailable as e:
        print(f"indexed {args.path}, but the smoke search failed: {e}")
        return 1
    print(f"indexed {args.path} (smoke search returned {len(hits)} hit(s))")
    return 0


def _cmd_index_docs(args) -> int:
    live_context.configure(config=_config(args))
    try:
        results = live_context.index_docs()
    except live_context.BackendUnavailable as e:
        print("index status: backend_unavailable")
        print(f"detail: {e}")
        return 1
    except Exception as e:  # CLI boundary: an unstructured build crash is still a failed index
        print("index status: index_failed")
        print(f"detail: {type(e).__name__}: {e}")
        return 1
    if not results:
        print("index status: empty (no paper/experiment/skill source directories)")
        return 0
    failed = False
    for result in results:
        label = "ok" if result.ok else "index_failed"
        detail = result.detail or result.version_after or "(no detail)"
        print(f"{result.kind}: {label} — {detail}")
        failed = failed or not result.ok
    print(f"index status: {'index_failed' if failed else 'ok'}")
    return 1 if failed else 0


def _cmd_ask(args) -> int:
    cfg = _config(args)
    live_context.configure(code_root=args.root, config=cfg)
    query = " ".join(args.query)
    kinds = (
        tuple(dict.fromkeys(part.strip() for part in args.kinds.split(",") if part.strip()))
        if args.kinds
        else ("code", "paper", "experiment")
    )
    invalid = [kind for kind in kinds if kind not in {"code", "paper", "experiment", "skill"}]
    if invalid or not kinds:
        print(
            "invalid --kinds; expected a comma list from code,paper,experiment,skill",
            file=sys.stderr,
        )
        return 2
    print(f"Q: {query}")
    try:
        bundle = live_context.get_fresh_context(
            query, kinds=kinds, k=args.k, max_age_s=cfg.freshness_max_age_s
        )
    except live_context.BackendUnavailable as e:
        print("context status: backend_unavailable")
        print(f"detail: {e}")
        return 1
    except Exception as e:
        print("context status: index_failed")
        print(f"detail: {type(e).__name__}: {e}")
        return 1
    print(
        f"freshness: stale={bundle.freshness.is_stale()} "
        f"indexed_at={bundle.freshness.indexed_at:%Y-%m-%d %H:%M:%S} "
        f"reasons={bundle.freshness.reasons or '-'}"
    )
    if bundle.freshness.is_stale():
        print("-> context is stale; reject_stale() reindexing incrementally...")
        try:
            bundle = live_context.reject_stale(bundle)
        except live_context.StaleContextError as e:
            print("context status: index_failed")
            print(f"detail: {e}")
            return 1
        except Exception as e:  # keep unexpected reindex failures on the stable CLI contract
            print("context status: index_failed")
            print(f"detail: {type(e).__name__}: {e}")
            return 1
        print(
            f"   refreshed: stale={bundle.freshness.is_stale()} "
            f"indexed_at={bundle.freshness.indexed_at:%Y-%m-%d %H:%M:%S}"
        )
    effective_status = bundle.status
    if bundle.status == "index_failed" or bundle.freshness.is_stale():
        effective_status = "index_failed"
    elif bundle.unavailable_kinds:
        # A partial hit must not hide a requested source that could not be searched.
        effective_status = "backend_unavailable"
    elif bundle.status == "ok" and not bundle.items:
        effective_status = "empty"
    if effective_status not in {"ok", "empty", "backend_unavailable", "index_failed"}:
        effective_status = "index_failed"
    print(f"context status: {effective_status}")
    if bundle.unavailable_kinds:
        print("unavailable kinds: " + ", ".join(bundle.unavailable_kinds))
    for note in bundle.status_notes:
        print(f"status detail: {note}")
    print(f"{len(bundle.items)} context item(s):")
    for item in bundle.items:
        snippet = " ".join(item.text.split())[:120]
        print(f"  - [{item.provenance.locator}] (score={item.provenance.score:.3f}) {snippet}")
    if bundle.answer:
        print("\nAnswer:\n" + bundle.answer)
    return 0 if effective_status == "ok" else 1


def _cmd_trace(args) -> int:
    from .reporting import ReportingError, collect_run, write_html_trace

    cfg = _config(args)
    if args.out and not args.html:
        print("--out requires --html", file=sys.stderr)
        return 2
    try:
        if args.html:
            output = write_html_trace(cfg.runs_dir, args.run_id, out=args.out or None)
            print(f"HTML trace: {output}")
            return 0
        report = collect_run(cfg.runs_dir, args.run_id)
    except ReportingError as e:
        print(f"cannot render run {args.run_id}: {e}", file=sys.stderr)
        return 1
    state = report.state
    print(f"run_id : {state.run_id}")
    print(f"status : {state.status}")
    print(f"task   : {state.task.title} ({state.task.kind})")
    if not report.ledger:
        print("(no ledger)")
        return 0
    print(f"\nledger: {report.run_dir / 'ledger.jsonl'}")
    for event in report.ledger:
        ts = event.timestamp.isoformat()[:19].replace("T", " ")
        ref = event.verdict_ref or event.artifact_ref or ""
        extra = f"  {ref}" if ref else ""
        if event.notes:
            extra += f"  — {event.notes}"
        print(f"  [{event.seq:>3}] {ts}  {event.step_id:<12} {event.phase:<9}{extra}")
    return 0


def _cmd_runs_list(args) -> int:
    from .reporting import ReportingError, discover_runs

    try:
        runs = discover_runs(_config(args).runs_dir)
    except ReportingError as e:
        print(f"cannot list runs: {e}", file=sys.stderr)
        return 1
    if not runs:
        print("no persisted runs")
        return 0
    print(f"{'RUN ID':<42} {'STATUS':<18} {'UPDATED (UTC)':<25} TASK")
    for run in runs:
        print(
            f"{run.run_id:<42} {run.status:<18} "
            f"{run.updated_at.isoformat(timespec='seconds'):<25} {run.task}"
        )
        if run.error:
            print(f"  detail: {run.error}")
    return 1 if any(run.status == "CORRUPT" for run in runs) else 0


def _cmd_runs_show(args) -> int:
    from .reporting import ReportingError, collect_run

    try:
        report = collect_run(_config(args).runs_dir, args.run_id)
    except ReportingError as e:
        print(f"cannot show run {args.run_id}: {e}", file=sys.stderr)
        return 1
    state = report.state
    print(f"run_id        : {state.run_id}")
    print(f"status        : {state.status}")
    print(f"task          : {state.task.title} ({state.task.kind})")
    print(f"updated       : {report.updated_at.isoformat()}")
    print(f"run_dir       : {report.run_dir}")
    print(f"cursor        : {state.cursor}")
    print(f"completed     : {len(state.completed_steps)}")
    print(f"failed        : {len(state.failed_steps)}")
    print(f"ledger events : {len(report.ledger)}")
    print(f"patches       : {len(report.patches)}")
    print(f"approvals     : {len(report.approvals)}")
    print(f"verdicts      : {len(report.verdicts)}")
    print(f"LLM calls     : {report.usage.calls}")
    print(f"input tokens  : {report.usage.input_tokens}")
    print(f"output tokens : {report.usage.output_tokens}")
    print(f"reported cost : ${report.usage.cost_usd:.4f}")
    return 0


def _cmd_runs_prune(args) -> int:
    from .reporting import ReportingError, prune_runs

    try:
        result = prune_runs(
            _config(args).runs_dir,
            older_than_days=args.older_than_days,
            apply=args.apply,
        )
    except ReportingError as e:
        print(f"cannot prune runs: {e}", file=sys.stderr)
        return 1
    if not result.entries:
        print("no runs matched the retention cutoff")
        return 0
    for entry in result.entries:
        print(f"{entry.action:<12} {entry.run_id} status={entry.status} — {entry.detail}")
    if not args.apply:
        print("dry run only; pass --apply to permanently delete eligible terminal runs")
    return 1 if result.refused else 0


def _cmd_approve(args, approved: bool) -> int:
    cfg = _config(args)
    state = load_state_by_id(cfg.runs_dir, args.run_id)
    try:
        with run_lock(state.run_dir):
            state = load_state_by_id(cfg.runs_dir, args.run_id)
            if state.status != "AWAITING_APPROVAL":
                print(
                    f"cannot resolve approval: run status is {state.status}, "
                    "not AWAITING_APPROVAL",
                    file=sys.stderr,
                )
                return 1
            step = state.next_step()
            if step is None:
                print(
                    "cannot resolve approval: checkpoint has no pending plan step",
                    file=sys.stderr,
                )
                return 1
            gate = HumanApprovalGate(state.run_dir)
            try:
                request = gate.pending_request()
            except ValueError as error:
                print(f"cannot resolve approval: {error}", file=sys.stderr)
                return 1
            attempt_id = state.attempt_id(step)
            expected_hash = None
            if step.action == "edit_code":
                patch_path = (
                    Path(state.run_dir)
                    / "steps"
                    / step.step_id
                    / "patch.json"
                )
                if patch_path.is_symlink() or not patch_path.is_file():
                    print(
                        "cannot resolve approval: reviewed patch artifact is missing",
                        file=sys.stderr,
                    )
                    return 1
                expected_hash = hashlib.sha256(patch_path.read_bytes()).hexdigest()
            if (
                request.step_id != step.step_id
                or request.attempt_id != attempt_id
                or request.goal != step.goal
                or request.artifact_sha256 != expected_hash
            ):
                print(
                    "cannot resolve approval: pending request does not match "
                    "the current step and artifact",
                    file=sys.stderr,
                )
                return 1
            try:
                gate.resolve(approved=approved, note=args.note or "")
            except ValueError as error:
                print(f"cannot resolve approval: {error}", file=sys.stderr)
                return 1
    except RunLocked as error:
        print(f"cannot resolve approval: {error}", file=sys.stderr)
        return 1
    print(
        f"{'approved' if approved else 'rejected'} run {args.run_id}; run `lha resume {args.run_id}`"
    )
    return 0


def _result_dict(result) -> dict:
    run_dir = Path(result.state.run_dir)
    verified = None
    vj = run_dir / "verify.json"
    if vj.exists():
        from .verifiers.verdict import Verdict

        verified = Verdict.model_validate_json(vj.read_text()).passed
    return {
        "run_id": result.state.run_id,
        "status": result.status,
        "run_dir": str(run_dir),
        "verified": verified,
        "summary": result.state.pr_summary_path,
        "message": result.message,
        "llm_usage": result.state.llm_usage.model_dump(mode="json"),
    }


def _usage_totals(run_dir: Path) -> dict | None:
    """Sum the run's llm_trace.jsonl into {calls, input_tokens, output_tokens, cost_usd}."""
    import json

    trace = run_dir / "llm_trace.jsonl"
    if not trace.exists():
        return None
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    for line in trace.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn tail must not hide the rest of the trace
        totals["calls"] += 1
        usage = rec.get("usage") or {}
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["cost_usd"] += float(usage.get("cost_usd") or 0.0)
    return totals


def _emit(result, args) -> int:
    if getattr(args, "json", False):
        import json

        print("__LHA_RESULT__ " + json.dumps(_result_dict(result)))
    else:
        _print_result(result)
    return 0 if result.status in ("DONE", "AWAITING_APPROVAL", "PAUSED") else 1


def _print_result(result) -> None:
    s = result.state
    print(f"run_id : {s.run_id}")
    print(f"status : {result.status}" + (f"  ({result.message})" if result.message else ""))
    print(f"run_dir: {s.run_dir}")
    verify = Path(s.run_dir) / "verify.json"
    if verify.exists():
        from .verifiers.verdict import Verdict

        v = Verdict.model_validate_json(verify.read_text())
        print(f"verify : passed={v.passed}")
        for c in v.checks:
            print(f"   - {c.name}: passed={c.passed} ({c.detail.get('summary', '')})")
    if s.pr_summary_path:
        print(f"pr     : {s.pr_summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lha",
        description="run task steps with executable checks and resumable state",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--llm",
        choices=["stub", "claude_cli", "codex_cli", "anthropic"],
        help="LLM backend override",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (-v info, -vv debug)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a task")
    pr.add_argument("task")
    pr.add_argument("--auto-approve", action="store_true")
    pr.add_argument("--runtime", choices=["loop", "langgraph"], default="loop")
    pr.add_argument("--json", action="store_true", help="emit a machine-readable result line")
    pr.set_defaults(func=_cmd_run)

    pres = sub.add_parser("resume", help="resume a run")
    pres.add_argument("run_id")
    pres.add_argument("--auto-approve", action="store_true")
    pres.add_argument("--runtime", choices=["loop", "langgraph"], default="loop")
    pres.add_argument("--json", action="store_true")
    pres.set_defaults(func=_cmd_resume)

    pb = sub.add_parser("batch", help="run multiple tasks concurrently")
    pb.add_argument("tasks", nargs="+")
    pb.add_argument("--workers", type=int, default=4)
    pb.set_defaults(func=_cmd_batch)

    pev = sub.add_parser("eval", help="run the six repository regression workflows")
    pev.add_argument("--quick", action="store_true", help="skip the slow experiment cases")
    pev.set_defaults(func=_cmd_eval)

    pab = sub.add_parser("ablate", help="verification ablation (trust vs gate vs verify)")
    pab.add_argument("tasks", nargs="*", help="task yamls (default: data/tasks/bench_*.yaml)")
    pab.add_argument("--reps", type=int, default=1, help="repetitions per condition")
    pab.add_argument(
        "--model",
        default="",
        help="implementer model (for example gpt-5.4-mini)",
    )
    pab.add_argument("--out", default="", help="output dir (default: <runs>/ablation)")
    pab.add_argument(
        "--scorer-backend",
        choices=["trusted-local", "docker"],
        default="trusted-local",
        help="where the independent final scorer runs (docker for untrusted repos)",
    )
    pab.set_defaults(func=_cmd_ablate)

    pat = sub.add_parser(
        "ablation-attempt",
        help="register, inspect, complete, or abandon a one-shot formal ablation",
    )
    attempt_sub = pat.add_subparsers(dest="attempt_cmd", required=True)
    patr = attempt_sub.add_parser(
        "register",
        help="write a REGISTERED event after resolving every formal input",
    )
    patr.add_argument("--model", required=True, help="exact Codex model")
    patr.add_argument(
        "--reasoning-effort",
        required=True,
        help="exact Codex reasoning effort",
    )
    patr.add_argument(
        "--docker-image-id",
        required=True,
        help="immutable Docker image ID (sha256:..., tags are rejected)",
    )
    patr.add_argument(
        "--witness-remote",
        default="formal-witness",
        help="repository-local public HTTPS witness remote",
    )
    patr.add_argument("--json", action="store_true")
    patr.set_defaults(func=_cmd_ablation_attempt_register)

    pats = attempt_sub.add_parser("status", help="read the current attempt state")
    pats.add_argument("--json", action="store_true")
    pats.set_defaults(func=_cmd_ablation_attempt_status)

    patc = attempt_sub.add_parser(
        "complete",
        help="validate all formal evidence and write a COMPLETED event",
    )
    patc.add_argument("--json", action="store_true")
    patc.set_defaults(func=_cmd_ablation_attempt_complete)

    pata = attempt_sub.add_parser(
        "abandon",
        help="write an explicit ABANDONED event for the open attempt",
    )
    pata.add_argument("--reason-code", required=True)
    pata.add_argument("--reason", required=True)
    pata.add_argument("--json", action="store_true")
    pata.set_defaults(func=_cmd_ablation_attempt_abandon)

    ph = sub.add_parser("horizon", help="compose measured outcomes across task counts")
    ph.add_argument(
        "--from-report",
        default="benchmarks/ablation_report.json",
        help="measured ablation report to compose from (default: the committed snapshot)",
    )
    ph.add_argument("--out", default="", help="output dir (default: <runs>/horizon)")
    ph.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ph.set_defaults(func=_cmd_horizon)

    pi = sub.add_parser("index", help="build the code index for a path")
    pi.add_argument("path")
    pi.set_defaults(func=_cmd_index)

    pid = sub.add_parser(
        "index-docs", help="(re)build paper/experiment/skill indexes via CocoIndex"
    )
    pid.set_defaults(func=_cmd_index_docs)

    pa = sub.add_parser(
        "ask",
        help="retrieve indexed context and print source locations",
    )
    pa.add_argument("query", nargs="+")
    pa.add_argument("--root", default=".", help="code root to search")
    pa.add_argument("--kinds", default="", help="comma list: code,paper,experiment,skill")
    pa.add_argument("--k", type=int, default=8)
    pa.set_defaults(func=_cmd_ask)

    ptr = sub.add_parser("trace", help="render a run's ledger timeline")
    ptr.add_argument("run_id")
    ptr.add_argument("--html", action="store_true", help="write a self-contained static HTML trace")
    ptr.add_argument("--out", default="", help="HTML output path (default: <run>/trace.html)")
    ptr.set_defaults(func=_cmd_trace)

    pruns = sub.add_parser("runs", help="inspect and safely prune persisted runs")
    run_sub = pruns.add_subparsers(dest="runs_cmd", required=True)
    prl = run_sub.add_parser("list", help="list validated run checkpoints")
    prl.set_defaults(func=_cmd_runs_list)
    prs = run_sub.add_parser("show", help="show one run's status and evidence summary")
    prs.add_argument("run_id")
    prs.set_defaults(func=_cmd_runs_show)
    prp = run_sub.add_parser("prune", help="list or delete old terminal runs")
    prp.add_argument("--older-than-days", type=int, required=True)
    prp.add_argument(
        "--apply",
        action="store_true",
        help="permanently delete eligible DONE/FAILED runs (default is dry-run)",
    )
    prp.set_defaults(func=_cmd_runs_prune)

    pap = sub.add_parser("approve", help="approve a pending gate")
    pap.add_argument("run_id")
    pap.add_argument("--note", default="")
    pap.set_defaults(func=lambda a: _cmd_approve(a, True))

    prj = sub.add_parser("reject", help="reject a pending gate")
    prj.add_argument("run_id")
    prj.add_argument("--note", default="")
    prj.set_defaults(func=lambda a: _cmd_approve(a, False))

    return p


def _setup_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", 0))
    try:
        exit_code = args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        exit_code = 130
    except Exception as error:
        if getattr(args, "verbose", 0) >= 2:
            logging.getLogger(__name__).exception("command failed")
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
