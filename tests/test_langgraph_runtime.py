"""The LangGraph durable runtime reaches the same verified result as the loop.

Hermetic: stub LLM, null code backend, isolated data dir (no ccc / no model).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lha.config import Config
from lha.tasks.spec import TaskSpec
from lha.verifiers.verdict import Verdict

pytest.importorskip("langgraph")


def test_langgraph_runtime_reaches_verified_done(tmp_path):
    from lha.runtime.langgraph_runner import LangGraphHarness

    cfg = Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
    )
    result = LangGraphHarness(cfg).run(TaskSpec.from_file("data/tasks/fix_average.yaml"))
    assert result.status == "DONE"

    run_dir = Path(result.state.run_dir)
    assert (run_dir / "graph.sqlite").exists()  # durable checkpoint
    verdict = Verdict.model_validate_json((run_dir / "verify.json").read_text())
    assert verdict.passed
    fixed = (run_dir / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" not in fixed
