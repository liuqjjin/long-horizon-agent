"""Recovery barriers that prevent stale writers and forged terminal evidence."""

from __future__ import annotations

import hashlib
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

import lha.harness.state as state_module
from lha.clock import now
from lha.config import Config
from lha.harness import Harness
from lha.harness.checkpoint import load_state, save_state
from lha.harness.errors import CheckpointCorrupt, TransactionCorrupt
from lha.harness.state import CLIIdentity, RunRuntimeContract, RunState
from lha.harness.transaction import list_transactions
from lha.operation_lease import OperationLeaseStore, OperationRecoveryResult
from lha.process_result import ProcResult
from lha.reporting import ReportingError, collect_run, prune_runs
from lha.verifiers.code import PytestVerifier
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
    runtime: str = "loop",
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
        runtime=runtime,  # type: ignore[arg-type]
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


class _FakeCodexCLI:
    def _resolved_cli_identity(
        self,
    ) -> tuple[str, int, int, int, int, str]:
        return (
            "/test/bin/codex",
            1,
            1,
            1,
            1,
            "a" * 64,
        )

    def _cli_version(self) -> str:
        return "codex-cli test"


def test_docker_runtime_capture_binds_path_and_digest_around_image_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = CLIIdentity(
        path="/fixed/docker",
        sha256="d" * 64,
        version="Docker version test",
    )
    binds: list[bool] = []

    class Backend:
        docker = "docker"
        image = "lha:test"

        @staticmethod
        def bind_control_plane(*, verify_digest: bool):
            binds.append(verify_digest)
            return {"path": cli.path, "sha256": cli.sha256}

    monkeypatch.setattr(
        state_module,
        "_capture_cli_identity",
        lambda _value: cli,
    )
    monkeypatch.setattr(
        "lha.tools.shell.run",
        lambda *_args, **_kwargs: ProcResult(
            0,
            f"sha256:{'a' * 64}\n",
            "",
            0.01,
        ),
    )
    contract = RunRuntimeContract.capture(
        _config(tmp_path, exec_backend="docker", exec_image="lha:test"),
        runtime="loop",
        exec_backend=Backend(),
        code_root=tmp_path,
    )

    assert contract.docker_cli == cli
    assert contract.exec_image == "lha:test"
    assert contract.exec_image_id == f"sha256:{'a' * 64}"
    assert binds == [True, True]


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
    state = _suspended_state(
        tmp_path, f"{runtime}-stale-resumer", runtime=runtime
    )
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
def test_terminal_resume_validates_redundant_transaction_backups(
    runtime: str, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    runner, _runtime_module = _runtime(runtime, config)
    completed = runner.run(
        hermetic_task("data/tasks/fix_average.yaml"),
        run_id=f"{runtime}-damaged-terminal-backup",
    )
    assert completed.status == "DONE"
    run_dir = Path(completed.state.run_dir)
    transaction = list_transactions(run_dir, "s2-fix")[0]
    (run_dir / transaction.backup_ref).write_text("{broken")

    with pytest.raises(
        TransactionCorrupt, match="terminal transaction backup is unusable"
    ):
        runner.resume(completed.state.run_id)


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_interrupted_active_window_exhausts_deadline_before_planning(
    runtime: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, deadline_s=0.1)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-interrupted-deadline",
        config=config,
        runtime=runtime,
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
        runtime=runtime,
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
        runtime=runtime,
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


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parallel_verify", False),
        ("dynamic_planning", True),
        ("use_skill_memory", False),
        ("freshness_max_age_s", 99.0),
        ("code_backend", "ccc"),
        ("embedder_model", "different-embedder"),
        ("data_dir", Path("different-data")),
    ],
)
def test_resume_rejects_runtime_configuration_drift(
    runtime: str,
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    recorded = _config(tmp_path)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-runtime-{field}",
        config=recorded,
        runtime=runtime,
    )
    changed = recorded.model_copy(update={field: value})
    runner, _runtime_module = _runtime(runtime, changed)

    with pytest.raises(
        CheckpointCorrupt,
        match=rf"runtime contract changed.*{field}",
    ):
        runner.resume(state.run_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("codex_max_retries", 4),
        ("codex_retry_backoff_s", 0.25),
    ],
)
def test_resume_contract_tracks_codex_retry_controls(
    field: str,
    value: int | float,
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        llm_backend="codex_cli",
        codex_model="gpt-test-pinned",
        codex_cli_path="unused-in-test",
    )
    run_dir = tmp_path / "runs" / f"codex-{field}"
    workdir = run_dir / "workdir"
    workdir.mkdir(parents=True)
    client = _FakeCodexCLI()
    state = RunState.new(
        hermetic_task("data/tasks/fix_average.yaml"),
        f"codex-{field}",
        str(run_dir),
        str(workdir),
        config=config,
        runtime="loop",
        llm=client,
    )
    assert state.runtime_contract is not None
    assert state.runtime_contract.codex_max_retries == 2
    assert state.runtime_contract.codex_retry_backoff_s == 1.0

    changed = config.model_copy(update={field: value})
    with pytest.raises(
        CheckpointCorrupt,
        match=rf"runtime contract changed.*{field}",
    ):
        state.require_matching_runtime_contract(
            changed,
            runtime="loop",
            llm=client,
        )


def test_cli_run_without_explicit_model_cannot_resume(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        llm_backend="codex_cli",
        codex_model="",
        codex_cli_path="unused-in-test",
    )
    run_dir = tmp_path / "runs" / "codex-unpinned"
    workdir = run_dir / "workdir"
    workdir.mkdir(parents=True)
    client = _FakeCodexCLI()
    state = RunState.new(
        hermetic_task("data/tasks/fix_average.yaml"),
        "codex-unpinned",
        str(run_dir),
        str(workdir),
        config=config,
        runtime="loop",
        llm=client,
    )
    assert state.runtime_contract is not None
    assert state.runtime_contract.llm_model == "cli-default"
    assert state.runtime_contract.llm_model_pinned is False

    with pytest.raises(
        CheckpointCorrupt,
        match="did not record an explicitly pinned codex_cli model",
    ):
        state.require_matching_runtime_contract(
            config,
            runtime="loop",
            llm=client,
        )


def test_resume_contract_tracks_resolved_auto_code_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lha.live_context.resolve_code_backend_name",
        lambda **_kwargs: "null",
    )
    config = _config(tmp_path, code_backend="auto")
    state = _suspended_state(
        tmp_path,
        "resolved-code-backend",
        config=config,
    )
    assert state.runtime_contract is not None
    assert state.runtime_contract.code_backend == "auto"
    assert state.runtime_contract.resolved_code_backend == "null"

    monkeypatch.setattr(
        "lha.live_context.resolve_code_backend_name",
        lambda **_kwargs: "ccc",
    )
    with pytest.raises(
        CheckpointCorrupt,
        match=r"runtime contract changed.*resolved_code_backend",
    ):
        state.require_matching_runtime_contract(
            config,
            runtime="loop",
        )


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_legacy_docker_contract_without_cli_identity_cannot_resume(
    runtime: str,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-docker-to-local",
        config=config,
        runtime=runtime,
    )
    assert state.runtime_contract is not None
    state.runtime_contract = RunRuntimeContract.model_validate(
        {
            **state.runtime_contract.model_dump(mode="json"),
            "exec_backend": "docker",
            "exec_image": "lha:test",
            "exec_image_id": f"sha256:{'a' * 64}",
        }
    )
    save_state(state)

    runner, _runtime_module = _runtime(runtime, config)
    result = runner.resume(state.run_id)

    assert result.status == "PAUSED"
    assert result.state.quarantine is not None
    assert (
        result.state.quarantine.kind
        == "active_operation_recovery_unconfirmed"
    )
    assert "no recorded client identity" in result.state.quarantine.detail


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_resume_verifies_recorded_docker_bytes_before_operation_recovery(
    runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-changed-docker-cli",
        config=config,
        runtime=runtime,
    )
    fake_docker = tmp_path / "docker"
    original = b"#!/bin/sh\nexit 0\n"
    fake_docker.write_bytes(original)
    fake_docker.chmod(0o755)
    assert state.runtime_contract is not None
    state.runtime_contract = RunRuntimeContract.model_validate(
        {
            **state.runtime_contract.model_dump(mode="json"),
            "exec_backend": "docker",
            "exec_image": "lha:test",
            "exec_image_id": f"sha256:{'a' * 64}",
            "docker_cli": CLIIdentity(
                path=str(fake_docker.resolve()),
                sha256=hashlib.sha256(original).hexdigest(),
                version="Docker version test",
            ).model_dump(mode="json"),
        }
    )
    save_state(state)
    fake_docker.write_bytes(b"#!/bin/sh\nexit 1\n")

    def recovery_must_not_run(*_args: Any, **_kwargs: Any):
        raise AssertionError("operation recovery ran with changed Docker bytes")

    monkeypatch.setattr(
        "lha.sandbox.docker.DockerBackend.recover_active_operations",
        recovery_must_not_run,
    )
    runner, _runtime_module = _runtime(runtime, config)
    result = runner.resume(state.run_id)

    assert result.status == "PAUSED"
    assert result.state.quarantine is not None
    assert "Docker executable bytes changed" in result.state.quarantine.detail


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_legacy_state_without_runtime_contract_is_inspectable_but_not_resumable(
    runtime: str,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-legacy-contract",
        config=config,
        runtime=runtime,
    )
    state.runtime_contract = None
    save_state(state)
    assert load_state(state.run_dir).runtime_contract is None

    runner, _runtime_module = _runtime(runtime, config)
    with pytest.raises(CheckpointCorrupt, match="no persisted runtime contract"):
        runner.resume(state.run_id)


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_resume_quarantines_before_transaction_recovery_when_operation_survives(
    runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-active-operation",
        config=config,
        runtime=runtime,
    )
    worktree_file = Path(state.workdir) / "sentinel.txt"
    worktree_file.write_bytes(b"unchanged-worktree")
    transaction_file = Path(state.run_dir) / "transactions" / "sentinel.bin"
    transaction_file.parent.mkdir()
    transaction_file.write_bytes(b"unchanged-transaction")
    runner, runtime_module = _runtime(runtime, config)
    order: list[str] = []

    def unconfirmed(_run_dir: str | Path) -> OperationRecoveryResult:
        order.append("operation")
        return OperationRecoveryResult(
            False,
            quarantined_operation_ids=("a" * 32,),
            detail="simulated surviving operation",
        )

    backend = getattr(getattr(runner, "_h", runner), "exec")
    monkeypatch.setattr(backend, "recover_active_operations", unconfirmed)

    def transaction_recovery_must_not_run(*args: Any, **kwargs: Any) -> None:
        order.append("transaction")
        raise AssertionError("transaction recovery ran after unconfirmed cleanup")

    monkeypatch.setattr(
        runtime_module,
        "recover_transaction_journals",
        transaction_recovery_must_not_run,
    )
    result = runner.resume(state.run_id)

    assert result.status == "PAUSED"
    assert order == ["operation"]
    assert result.state.quarantine is not None
    assert (
        result.state.quarantine.kind
        == "active_operation_recovery_unconfirmed"
    )
    assert worktree_file.read_bytes() == b"unchanged-worktree"
    assert transaction_file.read_bytes() == b"unchanged-transaction"


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_operation_recovery_precedes_runtime_contract_mismatch(
    runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _config(tmp_path)
    state = _suspended_state(
        tmp_path,
        f"{runtime}-recovery-before-contract",
        config=recorded,
        runtime=runtime,
    )
    changed = recorded.model_copy(update={"parallel_verify": False})
    runner, _runtime_module = _runtime(runtime, changed)
    backend = getattr(getattr(runner, "_h", runner), "exec")
    calls: list[str] = []

    def confirmed(_run_dir: str | Path) -> OperationRecoveryResult:
        calls.append("operation")
        return OperationRecoveryResult(True)

    monkeypatch.setattr(backend, "recover_active_operations", confirmed)
    with pytest.raises(CheckpointCorrupt, match="runtime contract changed"):
        runner.resume(state.run_id)
    assert calls == ["operation"]


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_runtime_passes_persisted_pytest_inventory_to_verifier(
    runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    runner, _runtime_module = _runtime(runtime, config)
    original_verify = PytestVerifier.verify
    observed = []

    def record_inventory(self, artifact, context):
        if context.step.step_id == "s2-fix":
            observed.append(context.pytest_oracle_inventory)
        return original_verify(self, artifact, context)

    monkeypatch.setattr(PytestVerifier, "verify", record_inventory)
    completed = runner.run(
        hermetic_task("data/tasks/fix_average.yaml"),
        run_id=f"{runtime}-oracle-verifier-context",
    )

    assert completed.status == "DONE"
    persisted = completed.state.pytest_oracle_inventories["s2-fix"]
    assert observed
    assert all(inventory is not None for inventory in observed)
    assert all(inventory.sha256 == persisted.sha256 for inventory in observed)


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_disposable_pytest_collection_uses_the_run_owned_operation_store(
    runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    runner, _runtime_module = _runtime(runtime, config)
    original_prepare = OperationLeaseStore.prepare_local
    observed: list[tuple[Path, Path]] = []

    def record_prepare(self, command, *, cwd, operation_id=None):
        lease = original_prepare(
            self,
            command,
            cwd=cwd,
            operation_id=operation_id,
        )
        observed.append((self.run_dir, Path(cwd).resolve()))
        return lease

    monkeypatch.setattr(
        OperationLeaseStore,
        "prepare_local",
        record_prepare,
    )
    completed = runner.run(
        hermetic_task("data/tasks/fix_average.yaml"),
        run_id=f"{runtime}-disposable-operation-store",
    )

    assert completed.status == "DONE"
    run_dir = Path(completed.state.run_dir).resolve()
    disposable = [
        (lease_dir, cwd)
        for lease_dir, cwd in observed
        if not cwd.is_relative_to(run_dir)
    ]
    assert disposable
    assert all(lease_dir == run_dir for lease_dir, _cwd in disposable)
    assert OperationLeaseStore(run_dir).list() == []


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_resume_reuses_and_validates_persisted_pytest_oracle_inventory(
    runtime: str,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    runner, _runtime_module = _runtime(runtime, config)
    completed = runner.run(
        hermetic_task("data/tasks/fix_average.yaml"),
        run_id=f"{runtime}-oracle-inventory",
    )
    assert completed.status == "DONE"
    inventory = completed.state.pytest_oracle_inventories["s2-fix"]
    inventory_path = (
        Path(completed.state.run_dir)
        / "oracle_inventories"
        / "s2-fix.json"
    )
    inventory_bytes = inventory_path.read_bytes()
    assert inventory.sha256

    oracle = Path(completed.state.workdir) / inventory.protected_paths[0]
    oracle.write_bytes(oracle.read_bytes() + b"\n# changed after completion\n")
    resumed, _module = _runtime(runtime, config)
    result = resumed.resume(completed.state.run_id)

    assert result.status == "PAUSED"
    assert result.state.quarantine is not None
    assert result.state.quarantine.kind == "pytest_oracle_inventory_invalid"
    assert inventory_path.read_bytes() == inventory_bytes
    assert (
        result.state.pytest_oracle_inventories["s2-fix"].sha256
        == inventory.sha256
    )


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
