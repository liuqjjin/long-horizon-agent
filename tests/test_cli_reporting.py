"""Hermetic contracts for CLI reporting, run pruning, and context exit status."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from conftest import hermetic_task

import lha.cli as cli
from lha.artifacts import Plan, Step
from lha.clock import now
from lha.config import Config
from lha.harness import Harness
from lha.harness.approval import ApprovalDecision, HumanApprovalGate
from lha.harness.checkpoint import append_ledger, load_state, run_lock, save_state
from lha.harness.state import LLMUsageState, RunState, RunStatus, StepRecord
from lha.harness.transaction import list_transactions
from lha.live_context import (
    ContextBundle,
    ContextItem,
    Freshness,
    Provenance,
    ReindexResult,
)
from lha.live_context import freshness as fr
from lha.tasks.spec import TaskSpec
from lha.verifiers.verdict import Check, Verdict


def _invoke(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def _make_run(runs_dir: Path, run_id: str, status: RunStatus = "DONE") -> Path:
    run_dir = runs_dir / run_id
    workdir = run_dir / "workdir"
    workdir.mkdir(parents=True)
    state = RunState.new(
        TaskSpec(
            kind="issue_to_pr",
            title="Fix <script>alert('x')</script>",
            description="report fixture",
            target_repo="fixture",
            context_requirement="optional",
        ),
        run_id,
        str(run_dir),
        str(workdir),
        config=Config(runs_dir=runs_dir),
    )
    step = Step(
        step_id="s1",
        kind="context",
        action="gather_context",
        goal="fixture check",
        verifiers=["freshness"],
        context_requirement="optional",
    )
    state.plan = Plan(task_id=run_id, summary="fixture plan", steps=[step])
    state.status = status
    plan_json = state.plan.model_dump_json(indent=2)
    plan_sha256 = hashlib.sha256(plan_json.encode()).hexdigest()
    (run_dir / "plan.json").write_text(plan_json)
    (run_dir / "plans").mkdir()
    (run_dir / "plans" / "initial.json").write_text(plan_json)
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id="-",
            phase="plan",
            artifact_ref="plans/initial.json",
            evidence_sha256=plan_sha256,
        ),
    )
    if status in ("DONE", "FAILED"):
        attempt_id = "s1-r0"
        state.attempt_ids["s1"] = attempt_id
        step_dir = run_dir / "steps" / "s1"
        step_dir.mkdir(parents=True)
        attempt_dir = step_dir / "attempts" / attempt_id
        attempt_dir.mkdir(parents=True)
        artifact_path = step_dir / "context_bundle.json"
        artifact_path.write_text(
            ContextBundle(
                query="fixture",
                freshness=Freshness(
                    index_version="fixture",
                    indexed_at=now(),
                ),
                status="empty",
            ).model_dump_json(indent=2)
        )
        attempt_artifact = attempt_dir / "context_bundle.json"
        attempt_artifact.write_bytes(artifact_path.read_bytes())
        (run_dir / "context_bundle.json").write_bytes(
            attempt_artifact.read_bytes()
        )
        artifact_sha256 = hashlib.sha256(
            attempt_artifact.read_bytes()
        ).hexdigest()
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id="s1",
                phase="context",
                artifact_ref=f"attempts/{attempt_id}/context_bundle.json",
                evidence_sha256=artifact_sha256,
                attempt_id=attempt_id,
            ),
        )
        verdict = Verdict.from_checks(
            "s1",
            [
                Check(
                    name="freshness",
                    family="context",
                    passed=status == "DONE",
                    detail={"summary": "fixture terminal evidence"},
                )
            ],
            artifact_ref=f"attempts/{attempt_id}/context_bundle.json",
            artifact_sha256=artifact_sha256,
            attempt_id=attempt_id,
        )
        verdict_json = verdict.model_dump_json(indent=2)
        verdict_sha256 = hashlib.sha256(verdict_json.encode()).hexdigest()
        verdict_ref = f"attempts/{attempt_id}/verify.json"
        (attempt_dir / "verify.json").write_text(verdict_json)
        for path in (run_dir / "verify.json", step_dir / "verify.json"):
            path.write_text(verdict_json)
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id="s1",
                phase="verify",
                verdict_ref=verdict_ref,
                evidence_sha256=verdict_sha256,
                attempt_id=attempt_id,
                notes="objective check",
            ),
        )
        if status == "DONE":
            state.complete_step(step)
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id="s1",
                    phase="complete",
                    evidence_sha256=verdict_sha256,
                    attempt_id=attempt_id,
                ),
            )
        else:
            state.fail_current(step)
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id="s1",
                    phase="fail",
                    attempt_id=attempt_id,
                    idempotency_key=f"{attempt_id}:fail",
                ),
            )
    save_state(state)
    return run_dir


def _add_report_artifacts(run_dir: Path) -> None:
    state = load_state(run_dir)
    state.llm_usage = LLMUsageState(
        calls=1,
        wall_s=0.5,
        input_tokens=12,
        output_tokens=4,
        cost_usd=0.01,
    )
    save_state(state)
    (run_dir / "llm_trace.jsonl").write_text(
        json.dumps(
            {
                "kind": "complete",
                "backend": "stub",
                "duration_s": 0.5,
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "cost_usd": 0.01,
                    "model": "fixture-model",
                    "retries": 1,
                },
            }
        )
        + "\n"
    )


def _make_old(run_dir: Path, days: int = 30) -> None:
    timestamp = time.time() - days * 86400
    os.utime(run_dir / "state.json", (timestamp, timestamp))


def test_trace_html_is_self_contained_complete_and_escaped(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "html-run")
    _add_report_artifacts(run_dir)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["trace", "html-run", "--html"]) == 0
    output = capsys.readouterr().out
    report_path = run_dir / "trace.html"
    assert str(report_path) in output
    html = report_path.read_text()
    for heading in ("State", "Ledger", "Patch", "Approval", "Verdict", "LLM usage"):
        assert heading in html
    assert "12" in html and "fixture-model" in html
    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "http://" not in html and "https://" not in html


def test_approval_cli_refuses_terminal_and_allows_only_one_pending_decision(
    tmp_path, monkeypatch, capsys
):
    runs_dir = tmp_path / "runs"
    terminal = _make_run(runs_dir, "terminal", "DONE")
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))
    assert _invoke(["approve", "terminal"]) == 1
    assert not (terminal / "approval.json").exists()

    paused = Harness(
        Config(
            llm_backend="stub",
            code_backend="null",
            runs_dir=runs_dir,
            data_dir=tmp_path / "nodata",
        )
    ).run(
        hermetic_task("data/tasks/fix_average_approval.yaml"),
        run_id="pending",
    )
    assert paused.status == "AWAITING_APPROVAL"
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda command: _invoke([command, "pending"]),
                ["approve", "reject"],
            )
        )
    assert sorted(results) == [0, 1]
    decision = ApprovalDecision.model_validate_json(
        (runs_dir / "pending" / "approval.json").read_text()
    )
    assert decision.step_id == "s2-fix"
    assert decision.artifact_sha256
    assert not (runs_dir / "pending" / "pending_approval.json").exists()
    capsys.readouterr()


def test_result_summary_does_not_inherit_a_previous_step_verdict(
    tmp_path, capsys
):
    paused = Harness(
        Config(
            llm_backend="stub",
            code_backend="null",
            runs_dir=tmp_path / "runs",
            data_dir=tmp_path / "nodata",
            use_skill_memory=False,
        )
    ).run(
        hermetic_task("data/tasks/fix_average_approval.yaml"),
        run_id="stale-verdict",
    )
    run_dir = Path(paused.state.run_dir)
    assert paused.status == "AWAITING_APPROVAL"
    assert Verdict.model_validate_json(
        (run_dir / "verify.json").read_text()
    ).passed

    assert cli._result_dict(paused)["verified"] is None
    cli._print_result(paused)
    output = capsys.readouterr().out
    assert "status : AWAITING_APPROVAL" in output
    assert "verify :" not in output


def test_result_summary_uses_the_validated_final_attempt_verdict(tmp_path):
    completed = Harness(
        Config(
            llm_backend="stub",
            code_backend="null",
            runs_dir=tmp_path / "runs",
            data_dir=tmp_path / "nodata",
            use_skill_memory=False,
        )
    ).run(
        hermetic_task("data/tasks/fix_average.yaml"),
        run_id="terminal-verdict",
    )

    assert completed.status == "DONE"
    assert cli._result_dict(completed)["verified"] is True


def test_trace_html_keeps_approval_history_after_transient_files_are_cleared(
    tmp_path, monkeypatch, capsys
):
    runs_dir = tmp_path / "runs"
    config = Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=runs_dir,
        data_dir=tmp_path / "nodata",
    )
    paused = Harness(config).run(
        hermetic_task("data/tasks/fix_average_approval.yaml"),
        run_id="approval-history",
    )
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(
        approved=True, note="approved by reviewer"
    )
    done = Harness(config).resume(paused.state.run_id)
    assert done.status == "DONE"
    assert not (run_dir / "pending_approval.json").exists()
    assert not (run_dir / "approval.json").exists()
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["trace", "approval-history", "--html"]) == 0
    html = (run_dir / "trace.html").read_text()
    assert "ledger approval history" in html
    assert "approved by reviewer" in html
    assert "approval_decision.json" in html


def test_trace_html_honors_custom_output_and_refuses_path_escape(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "safe-run")
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))
    custom = tmp_path / "reports" / "run.html"

    assert _invoke(["trace", "safe-run", "--html", "--out", str(custom)]) == 0
    assert custom.exists()
    assert _invoke(["trace", "../safe-run", "--html"]) != 0
    assert "invalid run id" in capsys.readouterr().err


def test_trace_html_refuses_to_overwrite_a_symlink(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "linked-output")
    victim = tmp_path / "victim.html"
    victim.write_text("keep me")
    (run_dir / "trace.html").symlink_to(victim)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["trace", "linked-output", "--html"]) != 0
    assert "symlink" in capsys.readouterr().err
    assert victim.read_text() == "keep me"


def test_runs_list_and_show_surface_status_and_usage(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    done = _make_run(runs_dir, "done-run", "DONE")
    _add_report_artifacts(done)
    _make_run(runs_dir, "paused-run", "PAUSED")
    ignored = runs_dir / "horizon"
    ignored.mkdir()
    (ignored / "horizon_report.json").write_text("{}")
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["runs", "list"]) == 0
    listed = capsys.readouterr().out
    assert "done-run" in listed and "DONE" in listed
    assert "paused-run" in listed and "PAUSED" in listed
    assert "horizon" not in listed

    assert _invoke(["runs", "show", "done-run"]) == 0
    shown = capsys.readouterr().out
    assert "ledger events : 4" in shown
    assert "LLM calls     : 1" in shown
    assert "input tokens  : 12" in shown


def test_state_usage_remains_authoritative_when_trace_is_incomplete(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "incomplete-trace")
    _add_report_artifacts(run_dir)
    state = load_state(run_dir)
    state.llm_usage = LLMUsageState(
        calls=2,
        wall_s=1.25,
        input_tokens=20,
        output_tokens=8,
        cost_usd=0.02,
    )
    save_state(state)

    from lha.reporting import collect_run, render_html

    report = collect_run(runs_dir, "incomplete-trace")
    assert report.usage.calls == 2
    assert report.usage.input_tokens == 20
    assert report.usage.trace_calls == 1
    assert report.usage.trace_complete is False
    assert "durable usage checkpoint is authoritative" in render_html(report)


def test_reporting_refuses_schema_two_state_without_budget_limits(tmp_path):
    from lha.reporting import ReportingError, collect_run

    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "missing-budget-limits")
    checkpoint = run_dir / "state.json"
    envelope = json.loads(checkpoint.read_text())
    envelope["payload"].pop("budget_limits")
    canonical = json.dumps(
        envelope["payload"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    envelope["sha256"] = hashlib.sha256(canonical).hexdigest()
    checkpoint.write_text(json.dumps(envelope))

    with pytest.raises(ReportingError, match="missing its budget limits"):
        collect_run(runs_dir, "missing-budget-limits")


def test_reporting_recovers_usage_written_before_the_run_state(tmp_path):
    from lha.llm.stub import DeterministicStub
    from lha.llm.trace import TracedLLM
    from lha.reporting import collect_run

    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "usage-ahead")
    traced = TracedLLM(DeterministicStub()).bind(run_dir)
    traced.restore_totals(LLMUsageState())
    traced.complete("system", "prompt")

    # The state still says zero calls, modeling a crash before the step-level
    # checkpoint. The per-call write-ahead file is the newer durable record.
    assert load_state(run_dir).llm_usage.calls == 0
    report = collect_run(runs_dir, "usage-ahead")
    assert report.usage.calls == 1
    assert report.usage.trace_calls == 1
    assert report.usage.trace_complete is True


@pytest.mark.parametrize("terminal_status", ["DONE", "FAILED"])
def test_runs_prune_is_dry_run_by_default_and_deletes_only_with_apply(
    terminal_status, tmp_path, monkeypatch, capsys
):
    runs_dir = tmp_path / "runs"
    old = _make_run(runs_dir, "old-terminal", terminal_status)
    _make_old(old)
    recent = _make_run(runs_dir, "recent-terminal", terminal_status)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["runs", "prune", "--older-than-days", "7"]) == 0
    assert "WOULD DELETE" in capsys.readouterr().out
    assert old.exists() and recent.exists()

    assert _invoke(["runs", "prune", "--older-than-days", "7", "--apply"]) == 0
    assert "DELETED" in capsys.readouterr().out
    assert not old.exists()
    assert recent.exists()


@pytest.mark.parametrize("unsafe_status", ["RUNNING", "AWAITING_APPROVAL", "PAUSED"])
def test_runs_prune_refuses_every_nonterminal_status(unsafe_status, tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    unsafe = _make_run(runs_dir, "unsafe-run", unsafe_status)
    _make_old(unsafe)
    terminal = _make_run(runs_dir, "terminal-run", "DONE")
    _make_old(terminal)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["runs", "prune", "--older-than-days", "7", "--apply"]) != 0
    output = capsys.readouterr().out
    assert "REFUSE" in output and unsafe_status in output
    assert unsafe.exists()
    assert not terminal.exists()


def test_runs_prune_refuses_corrupt_state_and_ledger(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    broken_state = runs_dir / "broken-state"
    broken_state.mkdir(parents=True)
    (broken_state / "state.json").write_text("{not-json")
    _make_old(broken_state)
    broken_ledger = _make_run(runs_dir, "broken-ledger", "DONE")
    (broken_ledger / "ledger.jsonl").write_text("{bad}\n{}\n")
    _make_old(broken_ledger)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["runs", "prune", "--older-than-days", "7", "--apply"]) != 0
    output = capsys.readouterr().out
    assert output.count("REFUSE") == 2
    assert broken_state.exists() and broken_ledger.exists()


@pytest.mark.parametrize("damage", ["missing-plan", "missing-ledger", "wrong-verifier"])
def test_runs_prune_refuses_unproven_terminal_state(
    damage, tmp_path, monkeypatch, capsys
):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, f"unproven-{damage}", "DONE")
    if damage == "missing-plan":
        state = load_state(run_dir)
        state.plan = None
        save_state(state)
    elif damage == "missing-ledger":
        (run_dir / "ledger.jsonl").unlink()
    else:
        verdict = Verdict.from_checks(
            "s1",
            [
                Check(
                    name="unrelated",
                    family="context",
                    passed=True,
                    detail={"summary": "not the planned verifier"},
                )
            ],
        )
        (run_dir / "steps" / "s1" / "verify.json").write_text(
            verdict.model_dump_json()
        )
    _make_old(run_dir)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["runs", "prune", "--older-than-days", "7", "--apply"]) != 0
    assert "REFUSE" in capsys.readouterr().out
    assert run_dir.exists()


def test_runs_prune_refuses_corrupt_optional_evidence(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    broken = _make_run(runs_dir, "broken-patch", "DONE")
    (broken / "patch.json").write_text("{bad")
    _make_old(broken)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["runs", "prune", "--older-than-days", "7", "--apply"]) != 0
    assert "REFUSE" in capsys.readouterr().out
    assert broken.exists()


def test_runs_prune_refuses_corrupt_transaction_evidence(
    tmp_path, monkeypatch, capsys
):
    runs_dir = tmp_path / "runs"
    broken = _make_run(runs_dir, "broken-transaction", "DONE")
    transaction = broken / "transactions" / "s1" / "s1-r0.json"
    transaction.parent.mkdir(parents=True)
    transaction.write_text("{bad")
    _make_old(broken)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["runs", "prune", "--older-than-days", "7", "--apply"]) != 0
    assert "REFUSE" in capsys.readouterr().out
    assert broken.exists()


@pytest.mark.parametrize("damage", ["missing-manifest", "corrupt-backup"])
def test_runs_prune_refuses_incomplete_recovery_evidence(
    damage, tmp_path, monkeypatch, capsys
):
    runs_dir = tmp_path / "runs"
    config = Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=runs_dir,
        data_dir=tmp_path / "nodata",
    )
    paused = Harness(config).run(
        hermetic_task("data/tasks/fix_average_approval.yaml"),
        run_id=f"damaged-{damage}",
    )
    run_dir = Path(paused.state.run_dir)
    transaction = list_transactions(run_dir, "s2-fix")[0]
    if damage == "missing-manifest":
        (
            run_dir
            / "steps"
            / "s2-fix"
            / "attempts"
            / transaction.attempt_id
            / "manifest.json"
        ).unlink()
    else:
        (run_dir / transaction.backup_ref).write_text("{broken")
    state = load_state(run_dir)
    state.status = "FAILED"
    save_state(state)
    _make_old(run_dir)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    assert _invoke(["runs", "prune", "--older-than-days", "7", "--apply"]) != 0
    assert "REFUSE" in capsys.readouterr().out
    assert run_dir.exists()


def test_runs_prune_refuses_a_terminal_run_while_its_lock_is_held(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    locked = _make_run(runs_dir, "locked-terminal", "DONE")
    _make_old(locked)
    monkeypatch.setenv("LHA_RUNS_DIR", str(runs_dir))

    with run_lock(locked):
        assert _invoke(["runs", "prune", "--older-than-days", "7", "--apply"]) != 0
    assert "REFUSE" in capsys.readouterr().out
    assert locked.exists()


def test_index_docs_returns_nonzero_when_any_reindex_fails(monkeypatch, capsys):
    monkeypatch.setattr(cli.live_context, "configure", lambda **kwargs: None)
    monkeypatch.setattr(
        cli.live_context,
        "index_docs",
        lambda: [
            ReindexResult(kind="paper", ok=True, version_after="p1"),
            ReindexResult(kind="experiment", ok=False, detail="build exploded"),
        ],
    )

    assert _invoke(["index-docs"]) != 0
    output = capsys.readouterr().out
    assert "paper: ok" in output
    assert "experiment: index_failed" in output
    assert "build exploded" in output


def test_index_docs_empty_is_explicit_but_not_a_failed_reindex(monkeypatch, capsys):
    monkeypatch.setattr(cli.live_context, "configure", lambda **kwargs: None)
    monkeypatch.setattr(cli.live_context, "index_docs", lambda: [])

    assert _invoke(["index-docs"]) == 0
    assert "index status: empty" in capsys.readouterr().out


def test_index_docs_all_success_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr(cli.live_context, "configure", lambda **kwargs: None)
    monkeypatch.setattr(
        cli.live_context,
        "index_docs",
        lambda: [ReindexResult(kind="paper", ok=True, version_after="p1")],
    )

    assert _invoke(["index-docs"]) == 0
    assert "index status: ok" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("status", "unavailable", "expected_label", "expected_rc"),
    [
        ("ok", [], "ok", 0),
        ("empty", [], "empty", 1),
        ("backend_unavailable", ["code"], "backend_unavailable", 1),
        ("index_failed", [], "index_failed", 1),
        ("ok", ["paper"], "backend_unavailable", 1),
    ],
)
def test_ask_statuses_are_explicit_and_fail_closed(
    status, unavailable, expected_label, expected_rc, monkeypatch, capsys
):
    item = ContextItem(
        text="evidence",
        provenance=Provenance(source_kind="code", locator="src/x.py:1"),
    )
    bundle = ContextBundle(
        query="q",
        items=[item] if status == "ok" else [],
        freshness=fr.fresh_now("v1"),
        status=status,
        unavailable_kinds=unavailable,
        status_notes=[f"{kind}: unavailable" for kind in unavailable],
        requested_kinds=["code", "paper"],
    )
    monkeypatch.setattr(cli.live_context, "configure", lambda **kwargs: None)
    monkeypatch.setattr(cli.live_context, "get_fresh_context", lambda *args, **kwargs: bundle)

    assert _invoke(["ask", "question"]) == expected_rc
    assert f"context status: {expected_label}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "failure",
    [
        cli.live_context.StaleContextError("refresh boom"),
        RuntimeError("refresh boom"),
    ],
)
def test_ask_stale_refresh_failure_reports_index_failed(failure, monkeypatch, capsys):
    bundle = ContextBundle(
        query="q",
        freshness=fr.fresh_now("v1"),
        status="empty",
        requested_kinds=["code"],
    )
    bundle.freshness.is_stale_flag = True
    bundle.freshness.reasons = ["source changed"]
    monkeypatch.setattr(cli.live_context, "configure", lambda **kwargs: None)
    monkeypatch.setattr(cli.live_context, "get_fresh_context", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        cli.live_context,
        "reject_stale",
        lambda bundle: (_ for _ in ()).throw(failure),
    )

    assert _invoke(["ask", "question"]) != 0
    output = capsys.readouterr().out
    assert "context status: index_failed" in output
    assert "refresh boom" in output


def test_ask_rejects_unknown_kinds_before_querying(monkeypatch, capsys):
    monkeypatch.setattr(cli.live_context, "configure", lambda **kwargs: None)

    def should_not_query(*args, **kwargs):
        raise AssertionError("invalid kinds must not reach the facade")

    monkeypatch.setattr(cli.live_context, "get_fresh_context", should_not_query)
    assert _invoke(["ask", "question", "--kinds", "code,not-a-kind"]) == 2
    assert "invalid --kinds" in capsys.readouterr().err
