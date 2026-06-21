"""Interrupt mid-run (budget pause) and resume to the same terminal state."""

from __future__ import annotations

from pathlib import Path

from lha.config import Config
from lha.harness import Harness
from lha.tasks.spec import TaskSpec


def test_pause_then_resume(tmp_path):
    task = TaskSpec.from_file("data/tasks/fix_average.yaml")
    runs = tmp_path / "runs"

    nodata = tmp_path / "nodata"
    # max_steps=1 forces a checkpointed pause after the first step
    paused = Harness(
        Config(llm_backend="stub", code_backend="null", runs_dir=runs, data_dir=nodata, max_steps=1)
    )
    r1 = paused.run(task)
    assert r1.status == "PAUSED"
    assert r1.state.cursor == 1
    assert r1.state.completed_steps == ["s1-context"]

    # a fresh harness (new process) resumes from the checkpoint
    resumed = Harness(
        Config(
            llm_backend="stub", code_backend="null", runs_dir=runs, data_dir=nodata, max_steps=20
        )
    )
    r2 = resumed.resume(r1.state.run_id)
    assert r2.status == "DONE"

    fixed = (Path(r2.state.run_dir) / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" not in fixed
