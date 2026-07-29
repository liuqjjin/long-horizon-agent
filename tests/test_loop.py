"""The raw quickstart tasks work with a REAL pytest and no code index."""

from __future__ import annotations

from pathlib import Path

from lha.config import Config
from lha.harness import Harness
from lha.harness.approval import HumanApprovalGate
from lha.tasks.spec import TaskSpec
from lha.verifiers.verdict import Verdict


def _config(tmp_path: Path) -> Config:
    # isolate data_dir so paper/experiment/skill backends are unavailable (hermetic)
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
    )


def test_issue_to_pr_reaches_verified_done(tmp_path):
    task = TaskSpec.from_file("data/tasks/fix_average.yaml")
    result = Harness(_config(tmp_path)).run(task)

    assert result.status == "DONE"

    run_dir = Path(result.state.run_dir)
    verdict = Verdict.model_validate_json((run_dir / "verify.json").read_text())
    assert verdict.passed
    names = {c.name: c.passed for c in verdict.checks}
    assert names.get("pytest") is True
    assert names.get("ruff") is True

    # the real fix landed in the sandbox
    fixed = (run_dir / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" not in fixed

    # PR summary written with provenance
    assert result.state.pr_summary_path
    pr = Path(result.state.pr_summary_path).read_text()
    assert "Verification" in pr
    assert "pytest" in pr


def test_approval_quickstart_pauses_and_resumes_without_code_index(tmp_path):
    task = TaskSpec.from_file("data/tasks/fix_average_approval.yaml")
    config = _config(tmp_path)

    paused = Harness(config, interactive_approval=False).run(task)
    assert paused.status == "AWAITING_APPROVAL"

    HumanApprovalGate(paused.state.run_dir).resolve(
        approved=True,
        note="quickstart regression",
    )
    resumed = Harness(config, interactive_approval=False).resume(paused.state.run_id)

    assert resumed.status == "DONE"
    verdict = Verdict.model_validate_json(
        (Path(resumed.state.run_dir) / "verify.json").read_text()
    )
    assert verdict.passed
