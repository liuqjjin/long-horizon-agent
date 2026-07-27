"""Adversarial boundaries for experiment output collection and planning."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from numpy.lib import format as npy_format

from lha.agents.experimenter import Experimenter, load_bounded_npy
from lha.agents.supervisor import Supervisor
from lha.artifacts import Step
from lha.clock import now
from lha.config import Config
from lha.live_context.models import ContextBundle, Freshness
from lha.llm.base import LLMClient
from lha.sandbox.base import ExecutionBackend
from lha.tasks.spec import TaskSpec
from lha.tools.shell import ProcResult

_EXPERIMENT_TASK = Path("data/tasks/run_sr_experiment.yaml")


def _bundle() -> ContextBundle:
    return ContextBundle(
        query="q",
        freshness=Freshness(index_version="test", indexed_at=now()),
        status="empty",
    )


def _command_step(command: list[str], *, out_dir: str = "out") -> Step:
    return Step(
        step_id="run",
        kind="experiment",
        action="run_experiment",
        goal="run",
        verifiers=["psnr", "ssim", "reproducibility"],
        params={"experiment_cmd": command, "out_dir": out_dir},
        context_requirement="optional",
    )


def _script_step() -> Step:
    return Step(
        step_id="run",
        kind="experiment",
        action="run_experiment",
        goal="run",
        verifiers=["psnr", "ssim", "reproducibility"],
        params={
            "experiment_script": "experiment.py",
            "experiment_args": ["--out", "out"],
            "out_dir": "out",
        },
        context_requirement="optional",
    )


def _write_array_script(path: Path) -> None:
    path.write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--out', required=True)\n"
        "args = parser.parse_args()\n"
        "out = Path(args.out)\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "reference = np.arange(16, dtype=np.float32).reshape(4, 4)\n"
        "np.save(out / 'reference.npy', reference)\n"
        "np.save(out / 'prediction.npy', reference + 0.5)\n"
    )


class _TruncatedBackend(ExecutionBackend):
    name = "truncated"

    def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
        return ProcResult(
            125,
            "partial",
            "output exceeded capture limit",
            0.01,
            output_truncated=True,
        )

    def python(self) -> str:
        return "python"

    def tool(self, name: str) -> str:
        return name


def test_experiment_records_incomplete_subprocess_output(tmp_path: Path) -> None:
    result = Experimenter(_TruncatedBackend()).run(
        _command_step(["python", "experiment.py"]),
        _bundle(),
        tmp_path,
    )

    assert result.returncode == 125
    assert result.output_truncated is True


def test_successful_true_command_cannot_reuse_stale_arrays(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    np.save(out / "reference.npy", np.ones((4, 4), dtype=np.float32))
    np.save(out / "prediction.npy", np.ones((4, 4), dtype=np.float32))
    (out / "metrics.json").write_text('{"psnr": 99, "ssim": 1}')

    result = Experimenter().run(
        _command_step(["/usr/bin/true"]), _bundle(), tmp_path
    )

    assert result.returncode == 0
    assert result.reference_path is None
    assert result.prediction_path is None
    assert result.metrics == {}
    assert not (out / "reference.npy").exists()
    assert not (out / "prediction.npy").exists()


def test_retry_starts_from_an_empty_output_directory(tmp_path: Path) -> None:
    script = tmp_path / "experiment.py"
    _write_array_script(script)
    step = _script_step()
    first = Experimenter().run(step, _bundle(), tmp_path)
    assert first.reference_path == "out/reference.npy"
    assert first.prediction_path == "out/prediction.npy"

    script.write_text("pass\n")
    retry = Experimenter().run(step.as_repair(["forced retry"]), _bundle(), tmp_path)

    assert retry.returncode == 0
    assert retry.reference_path is None
    assert retry.prediction_path is None
    assert not (tmp_path / "out" / "reference.npy").exists()


def test_output_directory_cannot_be_workdir_or_a_symlink(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe experiment artifact path"):
        Experimenter().run(
            _command_step(["/usr/bin/true"], out_dir="."), _bundle(), tmp_path
        )

    source = tmp_path / "src"
    source.mkdir()
    source_file = source / "model.py"
    source_file.write_text("value = 1\n")
    with pytest.raises(ValueError, match="not owned by LHA"):
        Experimenter().run(
            _command_step(["/usr/bin/true"], out_dir="src"), _bundle(), tmp_path
        )
    assert source_file.read_text() == "value = 1\n"

    workdir = tmp_path / "work"
    outside = tmp_path / "outside"
    workdir.mkdir()
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep")
    (workdir / "out").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        Experimenter().run(
            _command_step(["/usr/bin/true"]), _bundle(), workdir
        )
    assert sentinel.read_text() == "keep"


def test_symlink_array_is_not_collected_as_invocation_evidence(tmp_path: Path) -> None:
    external = tmp_path / "external.npy"
    np.save(external, np.ones((4, 4), dtype=np.float32))
    script = tmp_path / "link.py"
    script.write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--out', required=True)\n"
        "args = parser.parse_args()\n"
        "out = Path(args.out)\n"
        f"(out / 'reference.npy').symlink_to({str(external)!r})\n"
        f"(out / 'prediction.npy').symlink_to({str(external)!r})\n"
    )
    step = _script_step().model_copy(
        update={
            "params": {
                "experiment_script": "link.py",
                "experiment_args": ["--out", "out"],
                "out_dir": "out",
            }
        }
    )

    result = Experimenter().run(step, _bundle(), tmp_path)

    assert result.returncode == 0
    assert result.reference_path is None
    assert result.prediction_path is None
    assert "collected" not in result.repro


def test_bounded_npy_rejects_large_shape_before_allocating(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "large-shape.npy"
    with artifact.open("wb") as stream:
        npy_format.write_array_header_1_0(
            stream,
            {
                "descr": "<f8",
                "fortran_order": False,
                "shape": (1_000_000,),
            },
        )
        stream.write(b"\0" * 8)

    def allocation_would_be_a_bug(*args, **kwargs):
        raise AssertionError("payload allocation happened before shape validation")

    monkeypatch.setattr(np, "fromfile", allocation_would_be_a_bug)
    with pytest.raises(ValueError, match="elements"):
        load_bounded_npy(artifact, max_elements=100)


def test_bounded_npy_rejects_large_header_before_loading_payload(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "large-header.npy"
    header = (
        b"{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }"
        + b" " * 450
        + b"\n"
    )
    artifact.write_bytes(
        b"\x93NUMPY"
        + b"\x01\x00"
        + struct.pack("<H", len(header))
        + header
        + b"\0" * 4
    )

    def allocation_would_be_a_bug(*args, **kwargs):
        raise AssertionError("payload allocation happened before header validation")

    monkeypatch.setattr(np, "fromfile", allocation_would_be_a_bug)
    with pytest.raises(ValueError, match="header"):
        load_bounded_npy(artifact, max_header_bytes=128)


@pytest.mark.parametrize(
    "array",
    [
        np.ones((1,) * 9, dtype=np.float32),
        np.array([{"unsafe": True}], dtype=object),
        np.ones((4,), dtype="S8"),
    ],
)
def test_bounded_npy_rejects_unsupported_structure(
    tmp_path: Path, array: np.ndarray
) -> None:
    artifact = tmp_path / "unsupported.npy"
    np.save(artifact, array)

    with pytest.raises(ValueError, match="dimensions|dtype"):
        load_bounded_npy(artifact)


def test_bounded_npy_rejects_file_size_and_trailing_payload(tmp_path: Path) -> None:
    artifact = tmp_path / "array.npy"
    np.save(artifact, np.ones((4,), dtype=np.float32))

    with pytest.raises(ValueError, match="file size"):
        load_bounded_npy(artifact, max_file_bytes=8)

    with artifact.open("ab") as stream:
        stream.write(b"trailing")
    with pytest.raises(ValueError, match="payload size"):
        load_bounded_npy(artifact)


class _PlanLLM(LLMClient):
    name = "plan-test"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, system: str, prompt: str) -> str:
        return json.dumps(self.payload)


def _dynamic_experiment_plan(
    params: dict, verifiers: list[str]
) -> dict:
    return {
        "summary": "candidate",
        "steps": [
            {
                "step_id": "candidate-run",
                "kind": "experiment",
                "action": "run_experiment",
                "goal": "run",
                "verifiers": verifiers,
                "params": params,
            }
        ],
    }


def test_dynamic_experiment_plan_cannot_omit_required_verifier() -> None:
    task = TaskSpec.from_file(_EXPERIMENT_TASK)
    template = Supervisor(Config()).plan(task)
    params = next(
        step.params for step in template.steps if step.action == "run_experiment"
    )
    payload = _dynamic_experiment_plan(
        params, ["psnr", "reproducibility"]
    )

    plan = Supervisor(Config(dynamic_planning=True), _PlanLLM(payload)).plan(task)

    assert [step.step_id for step in plan.steps] == ["s1-context", "s2-run"]
    assert plan.steps[-1].verifiers == ["psnr", "ssim", "reproducibility"]


def test_dynamic_experiment_plan_must_retain_runnable_template_params() -> None:
    task = TaskSpec.from_file(_EXPERIMENT_TASK)
    payload = _dynamic_experiment_plan(
        {"out_dir": "out"}, ["psnr", "ssim", "reproducibility"]
    )

    plan = Supervisor(Config(dynamic_planning=True), _PlanLLM(payload)).plan(task)

    assert [step.step_id for step in plan.steps] == ["s1-context", "s2-run"]


def test_dynamic_experiment_plan_with_fixed_protocol_is_accepted() -> None:
    task = TaskSpec.from_file(_EXPERIMENT_TASK)
    template = Supervisor(Config()).plan(task)
    params = next(
        step.params for step in template.steps if step.action == "run_experiment"
    )
    payload = _dynamic_experiment_plan(
        params, ["psnr", "ssim", "reproducibility"]
    )

    plan = Supervisor(Config(dynamic_planning=True), _PlanLLM(payload)).plan(task)

    assert [step.step_id for step in plan.steps] == ["candidate-run"]
