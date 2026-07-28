"""At-most-once experiment execution and isolated reproducibility checks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import lha.durable_io as durable_io
from lha.agents import experimenter as experimenter_module
from lha.agents.experimenter import (
    ExperimentAmbiguous,
    ExperimentEvidence,
    execute_experiment_once,
)
from lha.artifacts import Step
from lha.clock import now
from lha.config import Config
from lha.harness import Harness
from lha.live_context.models import ContextBundle, Freshness
from lha.reporting import ReportingError, collect_run, prune_runs
from lha.sandbox import TrustedLocalBackend
from lha.tasks.spec import TaskSpec
from lha.verifiers import VerifyContext
from lha.verifiers.experiment import ReproVerifier


def _bundle() -> ContextBundle:
    return ContextBundle(
        query="experiment",
        freshness=Freshness(index_version="v1", indexed_at=now()),
        status="empty",
    )


def _write_script(worktree: Path) -> None:
    (worktree / "exp.py").write_text(
        "import argparse, hashlib, json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "p = argparse.ArgumentParser(); p.add_argument('--out', required=True)\n"
        "a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)\n"
        "counter = Path('main_counter.txt')\n"
        "counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')\n"
        "ref = np.arange(256, dtype=np.float32).reshape(16, 16) / 256\n"
        "pred = np.clip(ref + 0.001, 0, 1).astype(np.float32)\n"
        "np.save(out / 'reference.npy', ref); np.save(out / 'prediction.npy', pred)\n"
        "json.dump({'psnr': 99.0, 'ssim': 1.0}, open(out / 'metrics.json', 'w'))\n"
        "json.dump({'seed': 1, 'versions': {'numpy': np.__version__}, "
        "'input_sha256': hashlib.sha256(ref.tobytes()).hexdigest(), "
        "'data_range': 1.0, 'channel_axis': None}, open(out / 'repro.json', 'w'))\n"
    )


def _step() -> Step:
    return Step(
        step_id="exp",
        kind="experiment",
        action="run_experiment",
        goal="run",
        verifiers=["reproducibility"],
        context_requirement="optional",
        params={
            "experiment_script": "exp.py",
            "experiment_args": ["--out", "out"],
            "out_dir": "out",
            "data_range": 1.0,
            "channel_axis": None,
        },
    )


def test_completed_experiment_attempt_is_reused(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    worktree = run_dir / "workdir"
    worktree.mkdir(parents=True)
    _write_script(worktree)
    first = execute_experiment_once(
        step=_step(),
        bundle=_bundle(),
        workdir=worktree,
        run_dir=run_dir,
        attempt_id="exp-r0",
        backend=TrustedLocalBackend(),
    )
    replay = execute_experiment_once(
        step=_step(),
        bundle=_bundle(),
        workdir=worktree,
        run_dir=run_dir,
        attempt_id="exp-r0",
        backend=TrustedLocalBackend(),
    )
    assert replay == first
    assert (worktree / "main_counter.txt").read_text() == "1"


def test_intent_without_result_refuses_to_repeat_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    worktree = run_dir / "workdir"
    worktree.mkdir(parents=True)
    _write_script(worktree)
    real_write = experimenter_module._write_checksummed_model

    def crash_before_result(path, model, *, run_dir):
        if isinstance(model, ExperimentEvidence):
            raise KeyboardInterrupt("after experiment command")
        return real_write(path, model, run_dir=run_dir)

    monkeypatch.setattr(
        experimenter_module, "_write_checksummed_model", crash_before_result
    )
    with pytest.raises(KeyboardInterrupt):
        execute_experiment_once(
            step=_step(),
            bundle=_bundle(),
            workdir=worktree,
            run_dir=run_dir,
            attempt_id="exp-r0",
            backend=TrustedLocalBackend(),
        )
    monkeypatch.setattr(
        experimenter_module, "_write_checksummed_model", real_write
    )

    with pytest.raises(ExperimentAmbiguous, match="refusing to duplicate"):
        execute_experiment_once(
            step=_step(),
            bundle=_bundle(),
            workdir=worktree,
            run_dir=run_dir,
            attempt_id="exp-r0",
            backend=TrustedLocalBackend(),
        )
    assert (worktree / "main_counter.txt").read_text() == "1"


@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
@pytest.mark.parametrize(
    "artifact_name",
    ("experiment_intent.json", "experiment_evidence.json"),
)
def test_experiment_evidence_rejects_linked_storage(
    link_kind: str,
    artifact_name: str,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    worktree = run_dir / "workdir"
    worktree.mkdir(parents=True)
    _write_script(worktree)
    execute_experiment_once(
        step=_step(),
        bundle=_bundle(),
        workdir=worktree,
        run_dir=run_dir,
        attempt_id="exp-r0",
        backend=TrustedLocalBackend(),
    )
    artifact = (
        run_dir
        / "steps"
        / "exp"
        / "attempts"
        / "exp-r0"
        / artifact_name
    )
    payload = artifact.read_bytes()
    external = tmp_path / f"external-{artifact_name}-{link_kind}"
    external.write_bytes(payload)
    artifact.unlink()
    if link_kind == "symlink":
        artifact.symlink_to(external)
    else:
        os.link(external, artifact)

    with pytest.raises(ExperimentAmbiguous, match="persisted experiment evidence"):
        execute_experiment_once(
            step=_step(),
            bundle=_bundle(),
            workdir=worktree,
            run_dir=run_dir,
            attempt_id="exp-r0",
            backend=TrustedLocalBackend(),
        )

    assert external.read_bytes() == payload


def test_experiment_evidence_read_rejects_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    worktree = run_dir / "workdir"
    worktree.mkdir(parents=True)
    _write_script(worktree)
    execute_experiment_once(
        step=_step(),
        bundle=_bundle(),
        workdir=worktree,
        run_dir=run_dir,
        attempt_id="exp-r0",
        backend=TrustedLocalBackend(),
    )
    attempts = run_dir / "steps" / "exp" / "attempts"
    attempt_dir = attempts / "exp-r0"
    detached = attempts / "detached-exp-r0"
    outside = tmp_path / "outside-read"
    outside.mkdir()
    external = outside / "experiment_evidence.json"
    external.write_bytes(b"outside")
    real_open = durable_io.os.open
    raced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal raced
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "experiment_evidence.json" and not raced:
            raced = True
            attempt_dir.rename(detached)
            attempt_dir.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(durable_io.os, "open", racing_open)

    with pytest.raises(ExperimentAmbiguous, match="persisted experiment evidence"):
        execute_experiment_once(
            step=_step(),
            bundle=_bundle(),
            workdir=worktree,
            run_dir=run_dir,
            attempt_id="exp-r0",
            backend=TrustedLocalBackend(),
        )

    assert raced
    assert external.read_bytes() == b"outside"


def test_experiment_evidence_write_rejects_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    worktree = run_dir / "workdir"
    worktree.mkdir(parents=True)
    _write_script(worktree)
    attempts = run_dir / "steps" / "exp" / "attempts"
    attempt_dir = attempts / "exp-r0"
    detached = attempts / "detached-exp-r0"
    outside = tmp_path / "outside-write"
    outside.mkdir()
    external = outside / "experiment_evidence.json"
    external.write_bytes(b"outside")
    real_open = durable_io.os.open
    raced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal raced
        if (
            isinstance(path, str)
            and path.startswith(".experiment_evidence.json.")
            and path.endswith(".tmp")
            and not raced
        ):
            raced = True
            attempt_dir.rename(detached)
            attempt_dir.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(durable_io.os, "open", racing_open)

    with pytest.raises(ExperimentAmbiguous, match="evidence could not be persisted"):
        execute_experiment_once(
            step=_step(),
            bundle=_bundle(),
            workdir=worktree,
            run_dir=run_dir,
            attempt_id="exp-r0",
            backend=TrustedLocalBackend(),
        )

    assert raced
    assert external.read_bytes() == b"outside"
    assert (worktree / "main_counter.txt").read_text() == "1"


def test_repro_verifier_runs_only_in_disposable_worktree_copy(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    worktree = run_dir / "workdir"
    worktree.mkdir(parents=True)
    _write_script(worktree)
    step = _step()
    artifact = execute_experiment_once(
        step=step,
        bundle=_bundle(),
        workdir=worktree,
        run_dir=run_dir,
        attempt_id="exp-r0",
        backend=TrustedLocalBackend(),
    )
    check = ReproVerifier().verify(
        artifact,
        VerifyContext(
            workdir=worktree,
            step=step,
            exec=TrustedLocalBackend(),
            attempt_id="exp-r0",
        ),
    )
    assert check.passed
    assert check.detail["rerun_isolation"] == "ephemeral-worktree-copy"
    assert (worktree / "main_counter.txt").read_text() == "1"


@pytest.mark.parametrize("mutation", ["changed", "missing", "symlink"])
def test_terminal_reporting_rejects_changed_experiment_arrays(
    mutation: str,
    tmp_path: Path,
) -> None:
    task = TaskSpec.from_file("data/tasks/run_sr_experiment.yaml").model_copy(
        update={"context_requirement": "optional"}
    )
    config = Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        use_skill_memory=False,
    )
    result = Harness(config).run(task, run_id=f"experiment-{mutation}")
    assert result.status == "DONE", result.message
    collect_run(config.runs_dir, result.state.run_id)

    prediction = Path(result.state.workdir) / "out" / "prediction.npy"
    if mutation == "changed":
        import numpy as np

        value = np.load(prediction, allow_pickle=False)
        np.save(prediction, value + 0.25)
    elif mutation == "missing":
        prediction.unlink()
    else:
        external = tmp_path / "external.npy"
        external.write_bytes(prediction.read_bytes())
        prediction.unlink()
        prediction.symlink_to(external)

    with pytest.raises(ReportingError, match="experiment outputs changed"):
        collect_run(config.runs_dir, result.state.run_id)
    pruned = prune_runs(config.runs_dir, older_than_days=0, apply=True)
    entry = next(
        item for item in pruned.entries if item.run_id == result.state.run_id
    )
    assert entry.action == "REFUSE"
