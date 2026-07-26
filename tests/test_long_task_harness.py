"""End-to-end evidence for the fixed ten-step repository protocol."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

from lha.agents import ContextEngineer
from lha.agents.supervisor import Supervisor
from lha.artifacts import Patch
from lha.config import Config
from lha.harness import Harness
from lha.harness.approval import ApprovalDecision, HumanApprovalGate
from lha.harness.transaction import (
    PatchTransaction,
    list_transactions,
    transaction_log_path,
    transaction_path,
)
from lha.llm.base import LLMClient
from lha.llm.trace import TracedLLM
from lha.repo_adapter import (
    RepoAdapterSpec,
    RepoStageAmbiguous,
    RepoStageRequest,
    execute_repo_stage_once,
    repository_tree_sha256,
)
from lha.reporting import ReportingError, collect_run, prune_runs
from lha.sandbox import ExecutionBackend, TrustedLocalBackend
from lha.tasks.spec import TaskSpec
from lha.tools.shell import ProcResult
from lha.verifiers.verdict import Verdict

ROOT = Path(__file__).resolve().parents[1]
LONG_TASKS = ROOT / "data" / "long_tasks"
TASK_IDS = (
    "config_parser",
    "sqlite_migration",
    "concurrency_failure",
    "cli_contract",
    "experiment_repro",
)


def _rewind_to_last_durable_prepared(run_dir: Path, tx: PatchTransaction) -> None:
    """Model a crash after apply but before the APPLIED state rename."""
    log_path = transaction_log_path(run_dir, tx.step_id, tx.attempt_id)
    first_line = log_path.read_text().splitlines()[0]
    first_envelope = json.loads(first_line)
    first_event = first_envelope["payload"]
    prepared = tx.model_copy(
        update={
            "status": "PREPARED",
            "applied_state": {},
            "updated_at": first_event["at"],
        }
    )
    payload = prepared.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = sha256(canonical).hexdigest()
    assert digest == first_event["transaction_sha256"]
    transaction_path(run_dir, tx.step_id, tx.attempt_id).write_text(
        json.dumps({"schema_version": 1, "sha256": digest, "payload": payload})
    )
    log_path.write_text(first_line + "\n")


class _ReferencePatchLLM(LLMClient):
    """Test-only oracle: production code never reads or applies this patch."""

    name = "reference-fixture"

    def __init__(self, patch: str, *, fail_first: bool):
        self.patch = patch
        self.fail_first = fail_first

    def complete(self, system: str, prompt: str) -> str:
        return ""

    def plan(self, task, template):
        return None

    def propose_patch(self, step, bundle, workdir) -> Patch:
        if self.fail_first and not step.prior_failures:
            return Patch(step_id=step.step_id, rationale="test-only empty first attempt")
        return Patch(
            step_id=step.step_id,
            unified_diff=self.patch,
            rationale="test-only fixed reference patch",
            based_on_context=bundle.locators(),
        )


class _CountingBackend(ExecutionBackend):
    name = "counting-local"

    def __init__(self, *, fail_compile: bool = False, interrupt_first: bool = False):
        self.inner = TrustedLocalBackend()
        self.calls: list[tuple[str, ...]] = []
        self.fail_compile = fail_compile
        self.interrupt_first = interrupt_first

    def python(self) -> str:
        return self.inner.python()

    def tool(self, name: str) -> str:
        return self.inner.tool(name)

    def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
        normalized = (Path(cmd[0]).name if cmd else "", *cmd[1:])
        self.calls.append(normalized)
        if self.interrupt_first:
            self.interrupt_first = False
            raise KeyboardInterrupt("simulated process death after stage PREPARED")
        if self.fail_compile and tuple(cmd[1:3]) == ("-m", "compileall"):
            return ProcResult(1, "", "forced build failure", 0.0)
        return self.inner.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            input=input,
            limits=limits,
        )


def _config(tmp_path: Path, **updates) -> Config:
    values = {
        "llm_backend": "stub",
        "code_backend": "null",
        "runs_dir": tmp_path / "runs",
        "data_dir": tmp_path / "data",
        "use_skill_memory": False,
        "parallel_verify": False,
        "max_steps": 30,
        "max_repairs": 2,
    }
    values.update(updates)
    return Config(**values)


def _task(task_id: str) -> TaskSpec:
    return TaskSpec.from_file(LONG_TASKS / task_id / "task.yaml")


def _harness(
    config: Config,
    task_id: str,
    *,
    fail_first: bool,
    backend: ExecutionBackend | None = None,
    auto_approve: bool = True,
):
    harness = Harness(config, auto_approve=auto_approve)
    harness.llm = TracedLLM(
        _ReferencePatchLLM(
            (LONG_TASKS / task_id / "reference.patch").read_text(),
            fail_first=fail_first,
        )
    )
    if backend is not None:
        harness.exec = backend
    return harness


def _ledger(run_dir: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (Path(run_dir) / "ledger.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_all_long_tasks_approval_and_safe_crash_match_an_uninterrupted_run(
    task_id: str,
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    interrupted_backend = _CountingBackend()
    first = _harness(
        config,
        task_id,
        fail_first=True,
        backend=interrupted_backend,
        auto_approve=False,
    ).run(_task(task_id), run_id=f"{task_id}-interrupted")
    assert first.status == "AWAITING_APPROVAL"

    # The approved empty patch is objectively rejected by the focused gate.
    HumanApprovalGate(first.state.run_dir).resolve(approved=True)
    repaired = _harness(
        config,
        task_id,
        fail_first=True,
        backend=interrupted_backend,
        auto_approve=False,
    ).resume(first.state.run_id)
    assert repaired.status == "AWAITING_APPROVAL"
    assert repaired.state.repairs == {"s06-edit": 1}

    # Approve the repaired bytes, then die only after s06 has reached a durable
    # verified checkpoint and before s07 can execute a command.
    HumanApprovalGate(repaired.state.run_dir).resolve(approved=True)
    real_gather = ContextEngineer.gather
    injected = {"done": False}

    def crash_at_safe_boundary(self, step, *args, **kwargs):
        if step.step_id == "s07-targeted" and not injected["done"]:
            injected["done"] = True
            raise KeyboardInterrupt("simulated death after the s06 checkpoint")
        return real_gather(self, step, *args, **kwargs)

    monkeypatch.setattr(ContextEngineer, "gather", crash_at_safe_boundary)
    with pytest.raises(KeyboardInterrupt):
        _harness(
            config,
            task_id,
            fail_first=True,
            backend=interrupted_backend,
            auto_approve=False,
        ).resume(repaired.state.run_id)
    monkeypatch.setattr(ContextEngineer, "gather", real_gather)

    result = _harness(
        config,
        task_id,
        fail_first=True,
        backend=interrupted_backend,
        auto_approve=False,
    ).resume(repaired.state.run_id)

    clean_backend = _CountingBackend()
    clean = _harness(
        config,
        task_id,
        fail_first=True,
        backend=clean_backend,
        auto_approve=True,
    ).run(_task(task_id), run_id=f"{task_id}-clean")

    assert result.status == "DONE", result.message
    assert clean.status == "DONE", clean.message
    assert result.state.plan is not None
    assert len(result.state.plan.steps) == 10
    assert result.state.completed_steps == [step.step_id for step in result.state.plan.steps]
    assert result.state.completed_steps == clean.state.completed_steps
    assert result.state.repairs == {"s06-edit": 1}
    assert result.state.repairs == clean.state.repairs
    assert result.state.llm_usage.calls == 2
    assert clean.state.llm_usage.calls == 2
    assert repository_tree_sha256(result.state.workdir) == repository_tree_sha256(
        clean.state.workdir
    )
    assert Counter(interrupted_backend.calls) == Counter(clean_backend.calls)

    complete = [record for record in _ledger(result.state.run_dir) if record["phase"] == "complete"]
    assert [record["step_id"] for record in complete] == result.state.completed_steps
    assert len({record["idempotency_key"] for record in complete}) == 10
    repair_events = [
        record
        for record in _ledger(result.state.run_dir)
        if record["phase"] == "repair" and record["step_id"] == "s06-edit"
    ]
    assert len(repair_events) == 1
    approvals = [
        record
        for record in _ledger(result.state.run_dir)
        if record["phase"] == "approval" and record["step_id"] == "s06-edit"
    ]
    assert [record["attempt_id"] for record in approvals] == [
        "s06-edit-r0",
        "s06-edit-r1",
    ]
    report = collect_run(config.runs_dir, result.state.run_id)
    decisions = [
        item.value
        for item in report.approvals
        if isinstance(item.value, ApprovalDecision)
    ]
    assert [decision.attempt_id for decision in decisions] == [
        "s06-edit-r0",
        "s06-edit-r1",
    ]

    for step in result.state.plan.steps:
        verdict = Verdict.model_validate_json(
            (
                Path(result.state.run_dir)
                / "steps"
                / step.step_id
                / "verify.json"
            ).read_text()
        )
        assert verdict.passed, (task_id, step.step_id, verdict.failures)
    for step_id in ("s07-targeted", "s08-full", "s09-lint", "s10-build"):
        stage = json.loads(
            (
                Path(result.state.run_dir)
                / "steps"
                / step_id
                / "repo_stage.json"
            ).read_text()
        )
        assert stage["status"] == "passed"
        assert stage["commands"] and all(command["passed"] for command in stage["commands"])

    summary = Path(result.state.pr_summary_path or "").read_text()
    assert "s08-full/repo-stage" in summary
    assert "s09-lint/repo-stage" in summary
    assert "s10-build/repo-stage" in summary

    baseline = json.loads(
        (
            Path(result.state.run_dir)
            / "steps"
            / "s03-baseline"
            / "repo_stage.json"
        ).read_text()
    )
    assert baseline["status"] == "passed"
    assert baseline["commands"][-1]["returncode"] == 1


def test_long_task_plan_is_fixed_and_does_not_send_reference_to_planner():
    class _PlannerMustNotRun:
        def plan(self, task, template):
            raise AssertionError("long-task protocol must not be model-planned")

    plan = Supervisor(
        Config(dynamic_planning=True),
        _PlannerMustNotRun(),
    ).plan(_task("config_parser"))

    assert [step.action for step in plan.steps] == [
        "repo_integrity",
        "repo_stage",
        "repo_stage",
        "repo_stage",
        "gather_context",
        "edit_code",
        "repo_stage",
        "repo_stage",
        "repo_stage",
        "repo_stage",
    ]
    edit = plan.steps[5]
    assert edit.requires_approval
    assert edit.verifiers == ["repo-targeted"]
    assert all("reference" not in key for key in edit.params)


def test_stage_prepared_crash_fails_closed_without_duplicate_execution(tmp_path: Path):
    task_id = "config_parser"
    task = _task(task_id)
    config = _config(tmp_path)
    crashing = _CountingBackend(interrupt_first=True)
    first = _harness(
        config,
        task_id,
        fail_first=False,
        backend=crashing,
    )
    with pytest.raises(KeyboardInterrupt):
        first.run(task, run_id="stage-crash")
    assert len(crashing.calls) == 1

    recovery_backend = _CountingBackend()
    recovered = _harness(
        config,
        task_id,
        fail_first=False,
        backend=recovery_backend,
    ).resume("stage-crash")
    assert recovered.status == "FAILED"
    assert "refusing to duplicate its side effects" in recovered.message
    assert recovery_backend.calls == []


def test_late_stage_failure_reverts_the_already_verified_patch(tmp_path: Path):
    task_id = "config_parser"
    backend = _CountingBackend(fail_compile=True)
    result = _harness(
        _config(tmp_path, max_repairs=0),
        task_id,
        fail_first=False,
        backend=backend,
    ).run(_task(task_id))

    assert result.status == "FAILED"
    assert result.state.failed_steps == ["s10-build"]
    source = LONG_TASKS / task_id / "repo"
    assert repository_tree_sha256(Path(result.state.workdir)) == repository_tree_sha256(source)
    transactions = list_transactions(Path(result.state.run_dir), "s06-edit")
    assert transactions
    assert all(transaction.status == "REVERTED" for transaction in transactions)


def test_approval_repair_and_prepared_patch_recovery_are_idempotent(tmp_path: Path):
    task_id = "config_parser"
    config = _config(tmp_path)
    backend = _CountingBackend()

    first = _harness(
        config,
        task_id,
        fail_first=True,
        backend=backend,
        auto_approve=False,
    ).run(_task(task_id), run_id="approval-recovery")
    assert first.status == "AWAITING_APPROVAL"
    assert first.state.cursor == 5

    HumanApprovalGate(first.state.run_dir).resolve(approved=True)
    second = _harness(
        config,
        task_id,
        fail_first=True,
        backend=backend,
        auto_approve=False,
    ).resume(first.state.run_id)
    assert second.status == "AWAITING_APPROVAL"
    transactions = list_transactions(Path(second.state.run_dir), "s06-edit")
    reference = next(tx for tx in transactions if tx.attempt_id == "s06-edit-r1")
    _rewind_to_last_durable_prepared(Path(second.state.run_dir), reference)

    HumanApprovalGate(second.state.run_dir).resolve(approved=True)
    final = _harness(
        config,
        task_id,
        fail_first=True,
        backend=backend,
        auto_approve=False,
    ).resume(second.state.run_id)
    assert final.status == "DONE", final.message
    assert len(final.state.completed_steps) == 10
    assert final.state.llm_usage.calls == 2
    expected = tmp_path / "expected-patched-tree"
    shutil.copytree(LONG_TASKS / task_id / "repo", expected)
    patch_apply = TrustedLocalBackend().run(
        [
            TrustedLocalBackend().tool("git"),
            "apply",
            str(LONG_TASKS / task_id / "reference.patch"),
        ],
        cwd=expected,
    )
    assert patch_apply.returncode == 0, patch_apply.stderr
    assert repository_tree_sha256(final.state.workdir) == repository_tree_sha256(expected)

    counts = Counter(call for call in backend.calls)
    assert counts[(Path(backend.python()).name, "--version")] == 1
    assert sum(1 for call in backend.calls if "tests/test_config.py" in call) == 3
    assert sum(1 for call in backend.calls if call[-1:] == ("-q",)) == 2
    assert sum(1 for call in backend.calls if "compileall" in call) == 1
    before_terminal_resume = list(backend.calls)
    terminal = _harness(
        config,
        task_id,
        fail_first=True,
        backend=backend,
        auto_approve=False,
    ).resume(final.state.run_id)
    assert terminal.status == "DONE"
    assert backend.calls == before_terminal_resume

    complete = [record for record in _ledger(final.state.run_dir) if record["phase"] == "complete"]
    assert len(complete) == 10
    assert len({record["idempotency_key"] for record in complete}) == 10
    approvals = [
        record
        for record in _ledger(final.state.run_dir)
        if record["phase"] == "approval" and record["step_id"] == "s06-edit"
    ]
    assert [record["attempt_id"] for record in approvals] == [
        "s06-edit-r0",
        "s06-edit-r1",
    ]


def test_stage_journal_rejects_an_ambiguous_direct_replay(tmp_path: Path):
    source = LONG_TASKS / "config_parser"
    worktree = tmp_path / "worktree"
    run_dir = tmp_path / "run"
    shutil.copytree(source / "repo", worktree)
    run_dir.mkdir()
    spec = RepoAdapterSpec.from_file(source / "adapter.yaml")
    crashing = _CountingBackend(interrupt_first=True)

    with pytest.raises(KeyboardInterrupt):
        execute_repo_stage_once(
            worktree=worktree,
            run_dir=run_dir,
            step_id="setup",
            attempt_id="setup-r0",
            spec=spec,
            backend=crashing,
            stage=RepoStageRequest(stage="setup").stage,
        )
    recovery = _CountingBackend()
    with pytest.raises(RepoStageAmbiguous, match="refusing to duplicate"):
        execute_repo_stage_once(
            worktree=worktree,
            run_dir=run_dir,
            step_id="setup",
            attempt_id="setup-r0",
            spec=spec,
            backend=recovery,
            stage="setup",
        )
    assert len(crashing.calls) == 1
    assert recovery.calls == []


@pytest.mark.parametrize("mutation", ("tampered", "plain"))
def test_stage_evidence_rejects_tampered_or_plain_storage(
    mutation: str,
    tmp_path: Path,
):
    source = LONG_TASKS / "config_parser"
    worktree = tmp_path / "worktree"
    run_dir = tmp_path / "run"
    shutil.copytree(source / "repo", worktree)
    run_dir.mkdir()
    spec = RepoAdapterSpec.from_file(source / "adapter.yaml")
    first = _CountingBackend()

    result = execute_repo_stage_once(
        worktree=worktree,
        run_dir=run_dir,
        step_id="setup",
        attempt_id="setup-r0",
        spec=spec,
        backend=first,
        stage="setup",
    )
    assert result.passed
    attempt_dir = run_dir / "steps" / "setup" / "attempts" / "setup-r0"
    evidence_path = attempt_dir / "repo_stage_evidence.json"
    intent_path = attempt_dir / "repo_stage_intent.json"
    evidence_envelope = json.loads(evidence_path.read_text())
    intent_envelope = json.loads(intent_path.read_text())
    assert set(evidence_envelope) == {"schema_version", "sha256", "payload"}
    assert set(intent_envelope) == {"schema_version", "sha256", "payload"}

    if mutation == "tampered":
        evidence_envelope["payload"]["result"]["status"] = "failed"
        evidence_path.write_text(json.dumps(evidence_envelope))
        match = "integrity check"
    else:
        evidence_path.write_text(json.dumps(evidence_envelope["payload"]))
        match = "checksummed envelope"

    replay = _CountingBackend()
    with pytest.raises(RepoStageAmbiguous, match=match):
        execute_repo_stage_once(
            worktree=worktree,
            run_dir=run_dir,
            step_id="setup",
            attempt_id="setup-r0",
            spec=spec,
            backend=replay,
            stage="setup",
        )
    assert replay.calls == []


def test_stage_intent_rejects_legacy_plain_storage(tmp_path: Path):
    source = LONG_TASKS / "config_parser"
    worktree = tmp_path / "worktree"
    run_dir = tmp_path / "run"
    shutil.copytree(source / "repo", worktree)
    run_dir.mkdir()
    spec = RepoAdapterSpec.from_file(source / "adapter.yaml")
    crashing = _CountingBackend(interrupt_first=True)

    with pytest.raises(KeyboardInterrupt):
        execute_repo_stage_once(
            worktree=worktree,
            run_dir=run_dir,
            step_id="setup",
            attempt_id="setup-r0",
            spec=spec,
            backend=crashing,
            stage="setup",
        )
    intent_path = (
        run_dir
        / "steps"
        / "setup"
        / "attempts"
        / "setup-r0"
        / "repo_stage_intent.json"
    )
    envelope = json.loads(intent_path.read_text())
    intent_path.write_text(json.dumps(envelope["payload"]))

    with pytest.raises(RepoStageAmbiguous, match="checksummed envelope"):
        execute_repo_stage_once(
            worktree=worktree,
            run_dir=run_dir,
            step_id="setup",
            attempt_id="setup-r0",
            spec=spec,
            backend=_CountingBackend(),
            stage="setup",
        )


def test_reference_patch_hash_used_by_test_oracle_is_fixed():
    for task_id in TASK_IDS:
        task_root = LONG_TASKS / task_id
        manifest = json.loads((task_root / "reference_manifest.json").read_text())
        assert sha256((task_root / "reference.patch").read_bytes()).hexdigest() == (
            manifest["reference_patch_sha256"]
        )


def test_langgraph_loads_typed_long_task_artifacts(tmp_path: Path):
    pytest.importorskip("langgraph")
    from lha.runtime.langgraph_runner import LangGraphHarness

    task_id = "config_parser"
    harness = LangGraphHarness(_config(tmp_path), auto_approve=True)
    harness._h.llm = TracedLLM(
        _ReferencePatchLLM(
            (LONG_TASKS / task_id / "reference.patch").read_text(),
            fail_first=False,
        )
    )
    result = harness.run(_task(task_id), run_id="long-langgraph")

    assert result.status == "DONE", result.message
    assert len(result.state.completed_steps) == 10


def test_reporting_refuses_deleted_repository_stage_evidence(
    tmp_path: Path,
) -> None:
    task_id = "config_parser"
    config = _config(tmp_path)
    result = _harness(
        config,
        task_id,
        fail_first=False,
        backend=_CountingBackend(),
        auto_approve=True,
    ).run(_task(task_id), run_id="stage-retention")
    assert result.status == "DONE", result.message
    collect_run(config.runs_dir, result.state.run_id)

    evidence = (
        Path(result.state.run_dir)
        / "steps"
        / "s02-setup"
        / "attempts"
        / "s02-setup-r0"
        / "repo_stage_evidence.json"
    )
    evidence.unlink()
    with pytest.raises(ReportingError, match="missing"):
        collect_run(config.runs_dir, result.state.run_id)
    pruned = prune_runs(config.runs_dir, older_than_days=0, apply=True)
    entry = next(
        item for item in pruned.entries if item.run_id == result.state.run_id
    )
    assert entry.action == "REFUSE"
    assert Path(result.state.run_dir).exists()
