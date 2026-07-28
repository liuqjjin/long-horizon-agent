"""A process that may still write forces the whole run into quarantine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.agents.experimenter import Experimenter
from lha.artifacts import Step
from lha.clock import now
from lha.config import Config
from lha.harness import Harness
from lha.harness.checkpoint import read_ledger
from lha.harness.transaction import load_transaction
from lha.live_context.models import ContextBundle, Freshness
from lha.llm.stub import DeterministicStub
from lha.llm.trace import TracedLLM
from lha.process_result import ProcResult
from lha.pytest_evidence import run_driver
from lha.repo_adapter import (
    RepoAdapter,
    RepoAdapterSpec,
    RepoCommand,
    RepoStageRequest,
    execute_repo_stage_once,
)
from lha.sandbox import (
    ExecutionBackend,
    ProcessCleanupUnconfirmed,
    TrustedLocalBackend,
)
from lha.tools.patch import make_unified_diff
from lha.verifiers import VerifyContext
from lha.verifiers.code import RepoStageVerifier, RuffVerifier
from lha.verifiers.experiment import PSNRVerifier, ReproVerifier


class _UnconfirmedCleanupBackend(ExecutionBackend):
    name = "cleanup-unconfirmed"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
        self.calls.append(list(cmd))
        if any("--collect-only" in str(part) for part in cmd):
            return TrustedLocalBackend().run(
                list(cmd),
                cwd=cwd,
                timeout=timeout,
                input=input,
                limits=limits,
            )
        return ProcResult(
            126,
            "",
            "descendant may still be running",
            0.01,
            cleanup_confirmed=False,
            cleanup_detail="process group probe was denied",
        )

    def python(self) -> str:
        return "python"

    def tool(self, name: str) -> str:
        return name


def _config(tmp_path: Path) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
        parallel_verify=False,
        use_skill_memory=False,
    )


def _assert_quarantined_without_rollback(result) -> None:
    assert result.status == "PAUSED"
    assert result.state.status == "PAUSED"
    assert result.state.cursor == 1
    assert result.state.completed_steps == ["s1-context"]
    assert result.state.failed_steps == []
    assert result.state.repairs == {}
    assert result.state.quarantine is not None
    assert result.state.quarantine.kind == "process_cleanup_unconfirmed"
    assert result.state.quarantine.step_id == "s2-fix"
    assert result.state.quarantine.returncode == 126
    assert result.state.quarantine.detail.endswith(
        "process group probe was denied"
    )

    worktree = Path(result.state.workdir)
    assert "sum(values) / len(values) - 1" not in (
        worktree / "mathutils.py"
    ).read_text()
    transaction = load_transaction(
        Path(result.state.run_dir),
        "s2-fix",
        "s2-fix-r0",
    )
    assert transaction is not None
    assert transaction.status == "APPLIED"

    phases = [
        record.phase
        for record in read_ledger(result.state.run_dir)
        if record.step_id == "s2-fix"
    ]
    assert "verify" in phases
    assert not {"repair", "complete", "fail"} & set(phases)


def test_loop_quarantines_cleanup_failure_without_repair_or_rollback(
    tmp_path: Path,
) -> None:
    backend = _UnconfirmedCleanupBackend()
    harness = Harness(_config(tmp_path))
    harness.exec = backend

    result = harness.run(hermetic_task("data/tasks/fix_average.yaml"))

    _assert_quarantined_without_rollback(result)
    calls = len(backend.calls)
    resumed = harness.resume(result.state.run_id)
    _assert_quarantined_without_rollback(resumed)
    assert len(backend.calls) == calls


def test_langgraph_quarantines_cleanup_failure_without_repair_or_rollback(
    tmp_path: Path,
) -> None:
    pytest.importorskip("langgraph")
    from lha.runtime.langgraph_runner import LangGraphHarness

    backend = _UnconfirmedCleanupBackend()
    harness = LangGraphHarness(_config(tmp_path))
    harness._h.exec = backend

    result = harness.run(hermetic_task("data/tasks/fix_average.yaml"))

    _assert_quarantined_without_rollback(result)
    calls = len(backend.calls)
    resumed = harness.resume(result.state.run_id)
    _assert_quarantined_without_rollback(resumed)
    assert len(backend.calls) == calls


def test_experiment_cleanup_failure_is_not_read_and_is_non_retryable(
    tmp_path: Path,
) -> None:
    backend = _UnconfirmedCleanupBackend()
    step = Step(
        step_id="experiment",
        kind="experiment",
        action="run_experiment",
        goal="run",
        verifiers=["psnr", "reproducibility"],
        params={"experiment_cmd": ["python", "experiment.py"]},
        context_requirement="optional",
    )
    bundle = ContextBundle(
        query="q",
        freshness=Freshness(index_version="test", indexed_at=now()),
    )

    artifact = Experimenter(backend).run(step, bundle, tmp_path)

    assert artifact.returncode == 126
    assert artifact.reference_path is None
    assert artifact.prediction_path is None
    context = VerifyContext(
        workdir=tmp_path,
        step=step,
        bundle=bundle,
        exec=backend,
    )
    for verifier in (PSNRVerifier(), ReproVerifier()):
        check = verifier.verify(artifact, context)
        assert check.passed is False
        assert check.detail["non_retryable"] is True
        assert check.detail["process_cleanup_unconfirmed"] is True
        assert check.detail["process_cleanup"]["returncode"] == 126


def test_repo_stage_preserves_backend_cleanup_failure_for_quarantine(
    tmp_path: Path,
) -> None:
    backend = _UnconfirmedCleanupBackend()
    spec = RepoAdapterSpec(
        allowed_tools=frozenset({"python"}),
        targeted=(
            RepoCommand(
                id="targeted",
                tool="python",
                args=("-c", "raise SystemExit(1)"),
            ),
            RepoCommand(
                id="must-not-run",
                tool="python",
                args=("-c", "raise SystemExit(0)"),
            ),
        ),
    )
    result = RepoAdapter(tmp_path, spec, backend).run_stage(
        RepoStageRequest(stage="targeted", stop_on_failure=False)
    )
    assert len(result.commands) == 1
    assert len(backend.calls) == 1
    command = result.commands[0]
    assert command.returncode == 126
    assert command.cleanup_unconfirmed is True
    assert command.cleanup_detail == "process group probe was denied"

    step = Step(
        step_id="stage",
        kind="code",
        action="repo_stage",
        goal="targeted",
        verifiers=["repo-stage"],
        params={"repo_stage": "targeted"},
        context_requirement="optional",
    )
    check = RepoStageVerifier().verify(
        result,
        VerifyContext(workdir=tmp_path, step=step, exec=backend),
    )

    assert check.passed is False
    assert check.detail["non_retryable"] is True
    assert check.detail["process_cleanup_unconfirmed"] is True
    assert check.detail["process_cleanup"]["returncode"] == 126


def test_repo_stage_cleanup_failure_is_not_hashed_or_published(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import lha.repo_adapter as repo_module

    repo = tmp_path / "repo"
    repo.mkdir()
    run_dir = tmp_path / "runs" / "run"
    run_dir.mkdir(parents=True)
    backend = _UnconfirmedCleanupBackend()
    spec = RepoAdapterSpec(
        allowed_tools=frozenset({"python"}),
        targeted=(
            RepoCommand(
                id="targeted",
                tool="python",
                args=("-c", "raise SystemExit(1)"),
            ),
        ),
    )

    def unexpected_hash(_root):
        raise AssertionError("an unconfirmed process makes the repository unsafe to hash")

    monkeypatch.setattr(repo_module, "repository_tree_sha256", unexpected_hash)

    with pytest.raises(ProcessCleanupUnconfirmed, match="process group probe was denied"):
        execute_repo_stage_once(
            worktree=repo,
            run_dir=run_dir,
            step_id="stage",
            attempt_id="stage-r0",
            spec=spec,
            backend=backend,
            stage="targeted",
        )

    attempt_dir = run_dir / "steps" / "stage" / "attempts" / "stage-r0"
    assert (attempt_dir / "repo_stage_intent.json").exists()
    assert not (attempt_dir / "repo_stage_evidence.json").exists()
    assert not (run_dir / "artifacts" / "stage" / "repo_stage.json").exists()


def test_pytest_driver_does_not_touch_receipt_after_cleanup_failure(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("unchanged")

    class Backend(_UnconfirmedCleanupBackend):
        def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
            assert input is not None
            report = Path(cwd) / json.loads(input)["report_name"]
            report.symlink_to(sentinel)
            return ProcResult(
                126,
                "",
                "descendant may still be running",
                0.01,
                cleanup_confirmed=False,
                cleanup_detail="process group probe was denied",
            )

    result = run_driver(tmp_path, Backend(), mode="collect")

    assert result.returncode == 126
    assert "cleanup could not be confirmed" in result.detail
    receipts = list(tmp_path.glob(".lha-scorer-*.json"))
    assert len(receipts) == 1
    assert receipts[0].is_symlink()
    assert sentinel.read_text() == "unchanged"


def test_git_apply_cleanup_interruption_pauses_without_rollback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import lha.sandbox.base as sandbox_base

    class DiffLLM(DeterministicStub):
        def propose_patch(self, step, bundle, workdir):
            source = Path(workdir) / "mathutils.py"
            before = source.read_text()
            after = before.replace(
                "sum(values) / len(values) - 1",
                "sum(values) / len(values)",
            )
            from lha.artifacts import Patch

            return Patch(
                step_id=step.step_id,
                unified_diff=make_unified_diff(before, after, "mathutils.py"),
            )

    real_run_bounded_process = sandbox_base.run_bounded_process

    def unconfirmed(cmd, *args, **kwargs):
        cwd = Path(kwargs.get("cwd", "."))
        if (
            not cmd
            or Path(cmd[0]).name != "git"
            or "--check" in cmd
            or "--numstat" in cmd
            or cwd.name != "workdir"
        ):
            return real_run_bounded_process(cmd, *args, **kwargs)
        return ProcResult(
            126,
            "",
            "git process group may still be running",
            0.01,
            cleanup_confirmed=False,
            cleanup_detail="git process group probe was denied",
        )

    monkeypatch.setattr(sandbox_base, "run_bounded_process", unconfirmed)
    harness = Harness(_config(tmp_path))
    harness.llm = TracedLLM(DiffLLM())

    def rollback_is_unsafe(*_args, **_kwargs):
        raise AssertionError("cleanup uncertainty must not enter rollback")

    monkeypatch.setattr(harness, "_revert_step", rollback_is_unsafe)

    result = harness.run(hermetic_task("data/tasks/fix_average.yaml"))

    assert result.status == "PAUSED"
    assert result.state.cursor == 1
    assert result.state.failed_steps == []
    assert result.state.repairs == {}
    assert result.state.quarantine is not None
    assert result.state.quarantine.kind == "process_cleanup_interrupted"
    assert result.state.quarantine.detail == "git process group probe was denied"
    assert "sum(values) / len(values) - 1" in (
        Path(result.state.workdir) / "mathutils.py"
    ).read_text()
    transaction = load_transaction(
        Path(result.state.run_dir),
        "s2-fix",
        "s2-fix-r0",
    )
    assert transaction is not None
    assert transaction.status == "PREPARED"

    resumed = harness.resume(result.state.run_id)
    assert resumed.status == "PAUSED"
    assert resumed.state.quarantine == result.state.quarantine


def test_literal_exit_126_with_confirmed_cleanup_is_an_ordinary_failure(
    tmp_path: Path,
) -> None:
    class Backend(_UnconfirmedCleanupBackend):
        def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
            return ProcResult(
                126,
                "",
                "tool returned 126",
                0.01,
                cleanup_confirmed=True,
                cleanup_detail="process group stopped",
            )

    step = Step(
        step_id="ruff",
        kind="code",
        action="edit_code",
        goal="lint",
        verifiers=["ruff"],
        context_requirement="optional",
    )
    check = RuffVerifier().verify(
        object(),
        VerifyContext(workdir=tmp_path, step=step, exec=Backend()),
    )

    assert check.passed is False
    assert check.detail["returncode"] == 126
    assert "process_cleanup_unconfirmed" not in check.detail
    assert "non_retryable" not in check.detail
