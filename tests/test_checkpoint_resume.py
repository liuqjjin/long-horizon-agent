"""A persisted run budget cannot be replaced by a new process."""

from __future__ import annotations

import pytest
from conftest import hermetic_task

from lha.config import Config
from lha.harness import Harness
from lha.harness.errors import CheckpointCorrupt


def test_pause_then_resume_refuses_a_larger_step_budget(tmp_path):
    task = hermetic_task("data/tasks/fix_average.yaml")
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

    # A fresh harness cannot alter the contract that was fixed at run creation.
    resumed = Harness(
        Config(
            llm_backend="stub", code_backend="null", runs_dir=runs, data_dir=nodata, max_steps=20
        )
    )
    with pytest.raises(CheckpointCorrupt, match=r"max_steps.*recorded=1.*current=20"):
        resumed.resume(r1.state.run_id)
