"""Recovery barriers that prevent stale writers and forged terminal evidence."""

from __future__ import annotations

import importlib
import os
import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import pytest
from conftest import hermetic_task

from lha.clock import now
from lha.config import Config
from lha.harness import Harness
from lha.harness.checkpoint import load_state, save_state
from lha.harness.errors import CheckpointCorrupt
from lha.harness.state import RunState
from lha.reporting import ReportingError, collect_run, prune_runs
from lha.verifiers.verdict import Verdict


def _config(tmp_path: Path, **overrides: Any) -> Config:
    values: dict[str, Any] = {
        "llm_backend": "stub",
        "code_backend": "null",
        "runs_dir": tmp_path / "runs",
        "data_dir": tmp_path / "nodata",
    }
    values.update(overrides)
    return Config(**values)


def _suspended_state(
    tmp_path: Path,
    run_id: str,
    *,
    config: Config | None = None,
) -> RunState:
    run_dir = tmp_path / "runs" / run_id
    workdir = run_dir / "workdir"
    workdir.mkdir(parents=True)
    config = config or _config(tmp_path)
    state = RunState.new(
        hermetic_task("data/tasks/fix_average.yaml"),
        run_id,
        str(run_dir),
        str(workdir),
        config=config,
    )
    state.status = "PAUSED"
    save_state(state)
    return state


def _runtime(runtime: str, config: Config):
    if runtime == "langgraph":
        pytest.importorskip("langgraph")
        from lha.runtime.langgraph_runner import LangGraphHarness

        return LangGraphHarness(config), importlib.import_module(
            "lha.runtime.langgraph_runner"
        )
    return Harness(config), importlib.import_module("lha.harness.loop")


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
@pytest.mark.parametrize("crash_phase", ["verify", "complete"])
def test_durable_verdict_transition_is_consumed_without_reexecution(
    runtime: str,
    crash_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    runner, runtime_module = _runtime(runtime, config)
    original_append = runtime_module.append_ledger
    interrupted = False

    def append_then_exit(state: RunState, record) -> None:
        nonlocal interrupted
        original_append(state, record)
        if (
            not interrupted
            and record.step_id == "s2-fix"
            and record.phase == crash_phase
        ):
            interrupted = True
            raise KeyboardInterrupt(
                f"simulated exit after durable {crash_phase} append"
            )

    monkeypatch.setattr(runtime_module, "append_ledger", append_then_exit)
    with pytest.raises(KeyboardInterrupt):
        runner.run(
            hermetic_task("data/tasks/fix_average.yaml"),
            run_id=f"{runtime}-{crash_phase}-boundary",
        )
    monkeypatch.setattr(runtime_module, "append_ledger", original_append)

    def verifier_must_not_repeat(*args: Any, **kwargs: Any):
        raise AssertionError("durable verdict was executed a second time")

    monkeypatch.setattr(
        "lha.agents.verifier_agent.VerifierAgent.verify",
        verifier_must_not_repeat,
    )
    resumed, _module = _runtime(runtime, config)
    result = resumed.resume(f"{runtime}-{crash_phase}-boundary")

    assert result.status == "DONE"
    records = runtime_module.read_ledger(result.state.run_dir)
    assert sum(
        record.step_id == "s2-fix" and record.phase == "verify"
        for record in records
    ) == 1
    assert sum(
        record.step_id == "s2-fix" and record.phase == "complete"
        for record in records
    ) == 1
    assert "len(values) - 1" not in (
        Path(result.state.workdir) / "mathutils.py"
    ).read_text()


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_orphan_immutable_verdict_is_adopted_without_reexecution(
    runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    runner, runtime_module = _runtime(runtime, config)
    original_persist = runtime_module._persist_verdict
    interrupted = False

    def persist_then_exit(run_dir, step_id, attempt_id, verdict_json):
        nonlocal interrupted
        reference = original_persist(
            run_dir, step_id, attempt_id, verdict_json
        )
        if step_id == "s2-fix" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt(
                "simulated exit before verdict ledger append"
            )
        return reference

    monkeypatch.setattr(
        runtime_module, "_persist_verdict", persist_then_exit
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run(
            hermetic_task("data/tasks/fix_average.yaml"),
            run_id=f"{runtime}-orphan-verdict",
        )
    monkeypatch.setattr(
        runtime_module, "_persist_verdict", original_persist
    )

    def verifier_must_not_repeat(*args: Any, **kwargs: Any):
        raise AssertionError("immutable orphan verdict was recomputed")

    monkeypatch.setattr(
        "lha.agents.verifier_agent.VerifierAgent.verify",
        verifier_must_not_repeat,
    )
    resumed, _module = _runtime(runtime, config)
    result = resumed.resume(f"{runtime}-orphan-verdict")
    assert result.status == "DONE"
    records = runtime_module.read_ledger(result.state.run_dir)
    assert sum(
        record.step_id == "s2-fix" and record.phase == "verify"
        for record in records
    ) == 1


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_stale_resumer_reloads_terminal_state_after_lock_barrier(
    runtime: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _suspended_state(tmp_path, f"{runtime}-stale-resumer")
    config = _config(tmp_path)
    runner, runtime_module = _runtime(runtime, config)
    entered_lock = threading.Event()
    terminal_saved = threading.Event()
    writer_errors: list[BaseException] = []

    @contextmanager
    def barrier_lock(run_dir: str | Path) -> Iterator[None]:
        entered_lock.set()
        assert terminal_saved.wait(timeout=5), "concurrent writer did not finish"
        yield

    def finish_while_resumer_waits() -> None:
        try:
            assert entered_lock.wait(timeout=5), "resumer never reached the lock"
            current = load_state(state.run_dir)
            current.status = "FAILED"
            save_state(current)
        except BaseException as error:
            writer_errors.append(error)
        finally:
            terminal_saved.set()

    monkeypatch.setattr(runtime_module, "run_lock", barrier_lock)

    def stale_drive_must_not_run(_state: RunState):
        raise AssertionError("a pre-lock state escaped into the runtime")

    monkeypatch.setattr(runner, "_drive", stale_drive_must_not_run)
    writer = threading.Thread(target=finish_while_resumer_waits)
    writer.start()
    result = runner.resume(state.run_id)
    writer.join(timeout=5)

    assert not writer.is_alive()
    assert not writer_errors
    assert result.status == "FAILED"
    assert load_state(state.run_dir).status == "FAILED"


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_interrupted_active_window_exhausts_deadline_before_planning(
    runtime: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, deadline_s=0.1)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-interrupted-deadline",
        config=config,
    )
    state.status = "RUNNING"
    state.active_since = now() - timedelta(seconds=5)
    save_state(state)
    runner, _runtime_module = _runtime(runtime, config)

    def planning_after_deadline(*args: Any, **kwargs: Any):
        raise AssertionError("expired recovery called Supervisor")

    monkeypatch.setattr("lha.agents.supervisor.Supervisor.plan", planning_after_deadline)
    result = runner.resume(state.run_id)
    recovered = load_state(state.run_dir)

    assert result.status == "PAUSED"
    assert "interrupted activity" in result.message
    assert recovered.status == "PAUSED"
    assert recovered.active_since is None
    assert recovered.elapsed_s >= 4.5


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_future_active_since_fails_closed_without_resetting_budget(
    runtime: str, tmp_path: Path
) -> None:
    config = _config(tmp_path, deadline_s=10)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-future-active",
        config=config,
    )
    state.status = "RUNNING"
    state.active_since = now() + timedelta(hours=1)
    save_state(state)
    runner, _runtime_module = _runtime(runtime, config)

    with pytest.raises(CheckpointCorrupt, match="active_since is in the future"):
        runner.resume(state.run_id)

    unchanged = load_state(state.run_dir)
    assert unchanged.status == "RUNNING"
    assert unchanged.active_since == state.active_since
    assert unchanged.elapsed_s == 0


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 21),
        ("max_repairs", 2),
        ("deadline_s", 10.0),
        ("max_llm_calls", 1),
    ],
)
def test_resume_rejects_any_budget_limit_drift(
    runtime: str,
    field: str,
    value: int | float,
    tmp_path: Path,
) -> None:
    recorded = _config(tmp_path)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-changed-{field}",
        config=recorded,
    )
    changed = recorded.model_copy(update={field: value})
    runner, _runtime_module = _runtime(runtime, changed)

    with pytest.raises(
        CheckpointCorrupt,
        match=rf"budget limits changed.*{field}",
    ):
        runner.resume(state.run_id)

    unchanged = load_state(state.run_dir)
    assert unchanged.status == "PAUSED"
    assert unchanged.budget_limits == state.budget_limits


def test_replaced_terminal_verdict_is_refused_by_reporting_and_pruning(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    completed = Harness(config).run(
        hermetic_task("data/tasks/fix_average.yaml"),
        run_id="forged-terminal-verdict",
    )
    assert completed.status == "DONE"
    collect_run(config.runs_dir, completed.state.run_id)

    final_step = completed.state.completed_steps[-1]
    run_dir = Path(completed.state.run_dir)
    step_verdict_path = run_dir / "steps" / final_step / "verify.json"
    original = Verdict.model_validate_json(step_verdict_path.read_text())
    forged = original.model_copy(
        update={
            "checks": [
                check.model_copy(
                    update={
                        "detail": {
                            **check.detail,
                            "summary": "syntactically valid replacement evidence",
                        }
                    }
                )
                for check in original.checks
            ],
            "timestamp": now(),
        }
    )
    assert forged.passed
    assert [check.name for check in forged.checks] == [
        check.name for check in original.checks
    ]
    forged_json = forged.model_dump_json(indent=2)
    step_verdict_path.write_text(forged_json)
    (run_dir / "verify.json").write_text(forged_json)

    with pytest.raises(ReportingError, match="verdict is not bound"):
        collect_run(config.runs_dir, completed.state.run_id)

    old = time.time() - 30 * 86400
    os.utime(run_dir / "state.json", (old, old))
    result = prune_runs(config.runs_dir, older_than_days=7, apply=True)

    assert [(entry.action, entry.status) for entry in result.entries] == [
        ("REFUSE", "CORRUPT")
    ]
    assert run_dir.exists()


def test_done_worktree_drift_is_refused_by_reporting_and_pruning(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    completed = Harness(config).run(
        hermetic_task("data/tasks/fix_average.yaml"),
        run_id="drifted-done-worktree",
    )
    assert completed.status == "DONE"
    collect_run(config.runs_dir, completed.state.run_id)

    run_dir = Path(completed.state.run_dir)
    target = run_dir / "workdir" / "mathutils.py"
    target.write_text("def average(values):\n    return 999\n")

    with pytest.raises(
        ReportingError, match="does not match the last VERIFIED transaction"
    ):
        collect_run(config.runs_dir, completed.state.run_id)

    old = time.time() - 30 * 86400
    os.utime(run_dir / "state.json", (old, old))
    result = prune_runs(config.runs_dir, older_than_days=7, apply=True)

    assert [(entry.action, entry.status) for entry in result.entries] == [
        ("REFUSE", "CORRUPT")
    ]
    assert run_dir.exists()
