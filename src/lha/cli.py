"""``lha`` command-line interface.

lha run <task.yaml>      run a task through the verification loop
lha resume <run_id>      resume a paused/awaiting run
lha eval [--quick]       self-evaluate across the five workflows
lha ablate [tasks...]    verification ablation: trust vs gate vs verify (real LLM)
lha horizon              error compounding: the per-step effect across n steps
lha batch <task>...      run multiple tasks in parallel (process-isolated)
lha trace <run_id>       render a run's ledger timeline
lha index <path>         (re)build the code index for a repo
lha index-docs           (re)build paper/experiment indexes via CocoIndex
lha ask <query...>       answer a query with fresh, cited context
lha approve|reject <run_id>   resolve a pending human-approval gate

Global: --llm {stub,claude_cli,anthropic}, -v/--verbose, --version.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, live_context
from .config import Config
from .harness import Harness, HumanApprovalGate, load_state_by_id
from .tasks.spec import TaskSpec

_EPILOG = """\
examples:
  lha run data/tasks/fix_average.yaml            fix a bug, verified by real pytest
  lha eval                                       self-evaluate across the five workflows
  lha run --runtime langgraph <task>             durable run with an approval gate
  lha ask "how is average computed" --kinds code answer with fresh, cited context
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
    llm = getattr(args, "llm", None) or "claude_cli"
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
    return 0


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
    live_context.index_docs()
    print("indexed paper + experiment notes (CocoIndex flows)")
    return 0


def _cmd_ask(args) -> int:
    cfg = _config(args)
    live_context.configure(code_root=args.root, config=cfg)
    query = " ".join(args.query)
    kinds = tuple(args.kinds.split(",")) if args.kinds else ("code", "paper", "experiment")
    bundle = live_context.get_fresh_context(
        query, kinds=kinds, k=args.k, max_age_s=cfg.freshness_max_age_s
    )
    print(f"Q: {query}")
    print(
        f"freshness: stale={bundle.freshness.is_stale()} "
        f"indexed_at={bundle.freshness.indexed_at:%Y-%m-%d %H:%M:%S} "
        f"reasons={bundle.freshness.reasons or '-'}"
    )
    if bundle.freshness.is_stale():
        print("-> context is stale; reject_stale() reindexing incrementally...")
        bundle = live_context.reject_stale(bundle)
        print(
            f"   refreshed: stale={bundle.freshness.is_stale()} "
            f"indexed_at={bundle.freshness.indexed_at:%Y-%m-%d %H:%M:%S}"
        )
    print(f"{len(bundle.items)} context item(s):")
    for item in bundle.items:
        snippet = " ".join(item.text.split())[:120]
        print(f"  - [{item.provenance.locator}] (score={item.provenance.score:.3f}) {snippet}")
    if bundle.answer:
        print("\nAnswer:\n" + bundle.answer)
    return 0


def _cmd_trace(args) -> int:
    import json

    cfg = _config(args)
    state = load_state_by_id(cfg.runs_dir, args.run_id)
    run_dir = Path(state.run_dir)
    print(f"run_id : {state.run_id}")
    print(f"status : {state.status}")
    print(f"task   : {state.task.title} ({state.task.kind})")
    ledger = run_dir / "ledger.jsonl"
    if not ledger.exists():
        print("(no ledger)")
        return 0
    print(f"\nledger: {ledger}")
    for line in ledger.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = str(e.get("timestamp", ""))[:19].replace("T", " ")
        ref = e.get("verdict_ref") or e.get("artifact_ref") or ""
        extra = f"  {ref}" if ref else ""
        if e.get("notes"):
            extra += f"  — {e['notes']}"
        print(
            f"  [{e.get('seq', '-'):>3}] {ts}  {e.get('step_id', '-'):<12} {e.get('phase', ''):<9}{extra}"
        )
    return 0


def _cmd_approve(args, approved: bool) -> int:
    cfg = _config(args)
    state = load_state_by_id(cfg.runs_dir, args.run_id)
    gate = HumanApprovalGate(state.run_dir)
    gate.resolve(approved=approved, note=args.note or "")
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
        "llm_usage": _usage_totals(run_dir),
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
        description="verification-first long-horizon agent harness",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--llm", choices=["stub", "claude_cli", "anthropic"], help="LLM backend override"
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

    pb = sub.add_parser("batch", help="run multiple tasks in parallel (orchestrator-worker)")
    pb.add_argument("tasks", nargs="+")
    pb.add_argument("--workers", type=int, default=4)
    pb.set_defaults(func=_cmd_batch)

    pev = sub.add_parser("eval", help="self-evaluate across the five workflows")
    pev.add_argument("--quick", action="store_true", help="skip the slow experiment cases")
    pev.set_defaults(func=_cmd_eval)

    pab = sub.add_parser("ablate", help="verification ablation (trust vs gate vs verify)")
    pab.add_argument("tasks", nargs="*", help="task yamls (default: data/tasks/bench_*.yaml)")
    pab.add_argument("--reps", type=int, default=1, help="repetitions per condition")
    pab.add_argument("--model", default="", help="implementer model (e.g. haiku) to calibrate difficulty")
    pab.add_argument("--out", default="", help="output dir (default: <runs>/ablation)")
    pab.add_argument(
        "--scorer-backend",
        choices=["trusted-local", "docker"],
        default="trusted-local",
        help="where the independent final scorer runs (docker for untrusted repos)",
    )
    pab.set_defaults(func=_cmd_ablate)

    ph = sub.add_parser(
        "horizon", help="error compounding: what the per-step gate buys over n steps"
    )
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

    pid = sub.add_parser("index-docs", help="(re)build paper/experiment indexes via CocoIndex")
    pid.set_defaults(func=_cmd_index_docs)

    pa = sub.add_parser("ask", help="answer a query with fresh, cited context")
    pa.add_argument("query", nargs="+")
    pa.add_argument("--root", default=".", help="code root to search")
    pa.add_argument("--kinds", default="", help="comma list: code,paper,experiment")
    pa.add_argument("--k", type=int, default=8)
    pa.set_defaults(func=_cmd_ask)

    ptr = sub.add_parser("trace", help="render a run's ledger timeline")
    ptr.add_argument("run_id")
    ptr.set_defaults(func=_cmd_trace)

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
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
