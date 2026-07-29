"""Code context always comes from the current per-run workdir."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.agents.context_engineer import ContextEngineer
from lha.artifacts import Step
from lha.clock import now
from lha.config import Config
from lha.harness import Harness
from lha.live_context.models import (
    ContextBundle,
    ContextItem,
    Freshness,
    Provenance,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
    )


def _task_with_private_repo(tmp_path: Path):
    task = hermetic_task("data/tasks/fix_average.yaml")
    source = tmp_path / "source"
    assert task.target_repo is not None
    shutil.copytree(Path(task.target_repo), source)
    return task.model_copy(update={"target_repo": str(source)}), source


def _runner(runtime: str, config: Config):
    if runtime == "loop":
        return Harness(config)
    pytest.importorskip("langgraph")
    from lha.runtime.langgraph_runner import LangGraphHarness

    return LangGraphHarness(config)


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_resume_context_uses_run_workdir_after_source_repo_changes(
    runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, source = _task_with_private_repo(tmp_path)
    original = ContextEngineer.gather
    interrupted = False

    def interrupt_after_plan(self, step, workdir=None):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("stop before first context")
        assert workdir is not None
        assert Path(workdir) == tmp_path / "runs" / runtime / "workdir"
        assert "SOURCE_CHANGED" not in (Path(workdir) / "mathutils.py").read_text()
        return original(self, step, workdir=workdir)

    monkeypatch.setattr(ContextEngineer, "gather", interrupt_after_plan)
    with pytest.raises(KeyboardInterrupt):
        _runner(runtime, _config(tmp_path)).run(task, run_id=runtime)

    source.joinpath("mathutils.py").write_text("SOURCE_CHANGED = True\n")
    result = _runner(runtime, _config(tmp_path)).resume(runtime)

    assert result.status == "DONE"


def test_later_non_repair_context_reloads_current_workdir_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, _source = _task_with_private_repo(tmp_path)
    workdir = tmp_path / "workdir"
    Harness(_config(tmp_path))._prepare_workdir(task, workdir)
    target = workdir / "mathutils.py"
    indexed_text = target.read_text()
    target.write_text(indexed_text.replace(" - 1", ""))

    stale = ContextBundle(
        query="average",
        items=[
            ContextItem(
                text=indexed_text,
                provenance=Provenance(
                    source_kind="code",
                    locator="mathutils.py:1-20",
                ),
            )
        ],
        freshness=Freshness(index_version="test", indexed_at=now()),
        requested_kinds=["code"],
    )
    monkeypatch.setattr(
        "lha.agents.context_engineer.get_fresh_context",
        lambda *args, **kwargs: stale.model_copy(deep=True),
    )
    step = Step(
        step_id="later-context",
        kind="code",
        action="gather_context",
        goal="read the implementation after the preceding verified edit",
        verifiers=["freshness"],
    )

    bundle = ContextEngineer(_config(tmp_path)).gather(step, workdir=workdir)

    assert step.repair_of is None
    assert bundle.items
    assert " - 1" not in bundle.items[0].text
    assert bundle.items[0].provenance.source_root == str(workdir.resolve())
