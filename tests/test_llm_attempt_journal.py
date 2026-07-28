"""Crash-replay contracts for paid plan and patch calls."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.artifacts import Patch, Plan, Step
from lha.clock import now
from lha.config import Config
from lha.harness import Harness
from lha.harness.errors import CheckpointCorrupt
from lha.harness.state import LLMUsageState
from lha.live_context.models import ContextBundle, Freshness
from lha.llm.base import LLMClient
from lha.llm.trace import TracedLLM
from lha.reporting import ReportingError, collect_run


class _CountingLLM(LLMClient):
    name = "counting"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def complete(self, system: str, prompt: str) -> str:
        raise AssertionError("complete is not used by this test backend")

    def plan(self, task, template):
        self.calls += 1
        if self.fail:
            raise RuntimeError("backend interrupted")
        return template

    def propose_patch(self, step, bundle, workdir):
        self.calls += 1
        if self.fail:
            raise AssertionError("a completed patch call was submitted again")
        return Patch(
            step_id=step.step_id,
            file_contents={"value.py": "VALUE = 2\n"},
        )


class _WrongIdentityLLM(_CountingLLM):
    def propose_patch(self, step, bundle, workdir):
        self.calls += 1
        return Patch(
            step_id="another-step",
            file_contents={"mathutils.py": "def average(values):\n    return 0\n"},
        )


class _FailingPatchLLM(_CountingLLM):
    name = "failing-patch"

    def plan(self, task, template):
        self.calls += 1
        return None

    def propose_patch(self, step, bundle, workdir):
        self.calls += 1
        raise RuntimeError("private-secret-marker patch backend failed")


class _FailingPlanLLM(_CountingLLM):
    name = "failing-plan"

    def plan(self, task, template):
        self.calls += 1
        raise RuntimeError("private-plan-secret-marker planning failed")


def _context(run_id: str, attempt_id: str, task, **config):
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "task": task.model_dump(mode="json"),
        "config": config,
    }


def test_completed_plan_result_is_reused_without_a_second_call(tmp_path: Path) -> None:
    task = hermetic_task("data/tasks/fix_average.yaml")
    template = Plan(
        task_id=task.title,
        summary="fixed",
        steps=[
            Step(
                step_id="s",
                kind="code",
                action="edit_code",
                goal="fix",
                verifiers=["pytest"],
            )
        ],
    )
    first_backend = _CountingLLM()
    first = TracedLLM(first_backend, max_calls=1).bind(tmp_path)
    first.restore_totals(LLMUsageState())
    first.set_call_context(**_context("run", "plan", task, model="a"))
    assert first.plan(task, template) == template
    assert first_backend.calls == 1

    replay_backend = _CountingLLM(fail=True)
    replay = TracedLLM(replay_backend, max_calls=1).bind(tmp_path)
    replay.restore_totals(LLMUsageState())
    replay.set_call_context(**_context("run", "plan", task, model="a"))
    assert replay.plan(task, template) == template
    assert replay_backend.calls == 0
    assert replay.totals.calls == 1


def test_started_plan_rejects_changed_config_in_the_same_logical_slot(
    tmp_path: Path,
) -> None:
    task = hermetic_task("data/tasks/fix_average.yaml")
    template = Plan(
        task_id=task.title,
        summary="fixed",
        steps=[
            Step(
                step_id="s",
                kind="code",
                action="edit_code",
                goal="fix",
                verifiers=["pytest"],
            )
        ],
    )
    interrupted = TracedLLM(_CountingLLM(fail=True)).bind(tmp_path)
    interrupted.restore_totals(LLMUsageState())
    interrupted.set_call_context(
        **_context("run", "plan", task, max_steps=20, model="a")
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted.plan(task, template)

    replacement = _CountingLLM()
    resumed = TracedLLM(replacement).bind(tmp_path)
    resumed.restore_totals(LLMUsageState())
    resumed.set_call_context(
        **_context("run", "plan", task, max_steps=21, model="b")
    )
    with pytest.raises(CheckpointCorrupt, match="does not match"):
        resumed.plan(task, template)
    assert replacement.calls == 0


def test_patch_replay_key_ignores_observation_timestamps(tmp_path: Path) -> None:
    task = hermetic_task("data/tasks/fix_average.yaml")
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "value.py").write_text("VALUE = 1\n")
    step = Step(
        step_id="s-fix",
        kind="code",
        action="edit_code",
        goal="fix",
        verifiers=["pytest"],
    )
    first_bundle = ContextBundle(
        query="value",
        freshness=Freshness(index_version="v1", indexed_at=now()),
        status="empty",
    )
    first_backend = _CountingLLM()
    first = TracedLLM(first_backend).bind(tmp_path)
    first.restore_totals(LLMUsageState())
    first.set_call_context(**_context("run", "s-fix-r0", task, model="a"))
    expected = first.propose_patch(step, first_bundle, worktree)

    later_bundle = first_bundle.model_copy(
        update={
            "freshness": first_bundle.freshness.model_copy(
                update={"indexed_at": now()}
            )
        }
    )
    replay_backend = _CountingLLM(fail=True)
    replay = TracedLLM(replay_backend).bind(tmp_path)
    replay.restore_totals(LLMUsageState())
    replay.set_call_context(**_context("run", "s-fix-r0", task, model="a"))
    assert replay.propose_patch(step, later_bundle, worktree) == expected
    assert replay_backend.calls == 0


def test_wrong_patch_identity_never_reaches_done_or_trusted_reporting(
    tmp_path: Path,
) -> None:
    config = Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        max_repairs=0,
        use_skill_memory=False,
    )
    harness = Harness(config)
    harness.llm = TracedLLM(_WrongIdentityLLM())
    task = hermetic_task("data/tasks/fix_average.yaml").model_copy(
        update={"context_requirement": "optional"}
    )
    result = harness.run(task, run_id="wrong-patch-identity")

    assert result.status == "FAILED"
    assert "return sum(values) / len(values)" in (
        Path(result.state.workdir) / "mathutils.py"
    ).read_text()
    with pytest.raises(ReportingError, match="journaled patch call is orphaned"):
        collect_run(config.runs_dir, result.state.run_id)


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_failed_patch_call_has_replayable_terminal_evidence(
    runtime: str,
    tmp_path: Path,
) -> None:
    config = Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        use_skill_memory=False,
    )
    if runtime == "langgraph":
        pytest.importorskip("langgraph")
        from lha.runtime.langgraph_runner import LangGraphHarness

        harness = LangGraphHarness(config)
        harness._h.llm = TracedLLM(_FailingPatchLLM())
    else:
        harness = Harness(config)
        harness.llm = TracedLLM(_FailingPatchLLM())
    task = hermetic_task("data/tasks/fix_average.yaml").model_copy(
        update={"context_requirement": "optional"}
    )

    result = harness.run(task, run_id=f"{runtime}-failed-patch")

    assert result.status == "FAILED"
    report = collect_run(config.runs_dir, result.state.run_id)
    assert report.state.failed_steps == ["s2-fix"]
    run_dir = Path(result.state.run_dir)
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert b"private-secret-marker" not in path.read_bytes()

    if runtime == "langgraph":
        resumed_harness = LangGraphHarness(config)
        resumed_harness._h.llm = TracedLLM(_FailingPatchLLM())
    else:
        resumed_harness = Harness(config)
        resumed_harness.llm = TracedLLM(_FailingPatchLLM())
    resumed = resumed_harness.resume(result.state.run_id)
    assert resumed.status == "FAILED"


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_failed_plan_call_has_replayable_terminal_evidence(
    runtime: str,
    tmp_path: Path,
) -> None:
    config = Config(
        llm_backend="stub",
        code_backend="null",
        dynamic_planning=True,
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        use_skill_memory=False,
    )
    if runtime == "langgraph":
        pytest.importorskip("langgraph")
        from lha.runtime.langgraph_runner import LangGraphHarness

        harness = LangGraphHarness(config)
        harness._h.llm = TracedLLM(_FailingPlanLLM())
    else:
        harness = Harness(config)
        harness.llm = TracedLLM(_FailingPlanLLM())
    task = hermetic_task("data/tasks/fix_average.yaml")

    result = harness.run(task, run_id=f"{runtime}-failed-plan")

    assert result.status == "FAILED"
    report = collect_run(config.runs_dir, result.state.run_id)
    assert report.state.plan is None
    run_dir = Path(result.state.run_dir)
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert b"private-plan-secret-marker" not in path.read_bytes()

    if runtime == "langgraph":
        resumed_harness = LangGraphHarness(config)
        resumed_harness._h.llm = TracedLLM(_FailingPlanLLM())
    else:
        resumed_harness = Harness(config)
        resumed_harness.llm = TracedLLM(_FailingPlanLLM())
    resumed = resumed_harness.resume(result.state.run_id)
    assert resumed.status == "FAILED"
