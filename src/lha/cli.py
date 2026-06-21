"""``lha`` command-line interface.

    lha run <task.yaml>      run a task through the verification loop
    lha resume <run_id>      resume a paused/awaiting run
    lha index <path>         (re)build the code index for a repo
    lha ask <query...>       answer a query with fresh, cited context
    lha approve <run_id>     approve a pending human-approval gate
    lha reject <run_id>      reject a pending human-approval gate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import live_context
from .config import Config
from .harness import Harness, HumanApprovalGate, load_state_by_id
from .tasks.spec import TaskSpec


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


def _cmd_index(args) -> int:
    live_context.configure(code_root=args.path, config=_config(args))
    live_context.index_code(args.path)
    hits = live_context.search_code("function", k=1)
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
    print(f"freshness: stale={bundle.freshness.is_stale()} "
          f"indexed_at={bundle.freshness.indexed_at:%Y-%m-%d %H:%M:%S} "
          f"reasons={bundle.freshness.reasons or '-'}")
    if bundle.freshness.is_stale():
        print("-> context is stale; reject_stale() reindexing incrementally...")
        bundle = live_context.reject_stale(bundle)
        print(f"   refreshed: stale={bundle.freshness.is_stale()} "
              f"indexed_at={bundle.freshness.indexed_at:%Y-%m-%d %H:%M:%S}")
    print(f"{len(bundle.items)} context item(s):")
    for item in bundle.items:
        snippet = " ".join(item.text.split())[:120]
        print(f"  - [{item.provenance.locator}] (score={item.provenance.score:.3f}) {snippet}")
    if bundle.answer:
        print("\nAnswer:\n" + bundle.answer)
    return 0


def _cmd_approve(args, approved: bool) -> int:
    cfg = _config(args)
    state = load_state_by_id(cfg.runs_dir, args.run_id)
    gate = HumanApprovalGate(state.run_dir)
    gate.resolve(approved=approved, note=args.note or "")
    print(f"{'approved' if approved else 'rejected'} run {args.run_id}; run `lha resume {args.run_id}`")
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
    }


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
    p = argparse.ArgumentParser(prog="lha", description="verification-first agent harness")
    p.add_argument("--llm", choices=["stub", "claude_cli", "anthropic"], help="LLM backend override")
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

    pev = sub.add_parser("eval", help="run ResearchAgentBench-Lite (self-evaluation)")
    pev.add_argument("--quick", action="store_true", help="skip the slow experiment cases")
    pev.set_defaults(func=_cmd_eval)

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

    pap = sub.add_parser("approve", help="approve a pending gate")
    pap.add_argument("run_id")
    pap.add_argument("--note", default="")
    pap.set_defaults(func=lambda a: _cmd_approve(a, True))

    prj = sub.add_parser("reject", help="reject a pending gate")
    prj.add_argument("run_id")
    prj.add_argument("--note", default="")
    prj.set_defaults(func=lambda a: _cmd_approve(a, False))

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
