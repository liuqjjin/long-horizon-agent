"""Oracle-protection policy: a normal run cannot fake success by editing the oracle.

Invariants:
  - protected paths (tests, conftest, pytest/build/CI config) are refused at the
    patch boundary, in both the default loop and the LangGraph runtime;
  - the refusal feeds the repair loop as structured feedback;
  - a run whose model only ever rewrites the tests ends FAILED with the
    canonical tests untouched and the source still buggy;
  - a task manifest can authorize exact protected paths explicitly;
  - the policy reads the diff itself, not just the declared file list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.artifacts import Patch
from lha.config import Config
from lha.harness import Harness
from lha.tools import policy
from lha.verifiers.verdict import Verdict


# --- the predicate -----------------------------------------------------------
@pytest.mark.parametrize(
    "rel",
    [
        "tests/test_mathutils.py",
        "tests/nested/helper.py",
        "test_app.py",
        "pkg/module_test.py",
        "conftest.py",
        "src/pkg/conftest.py",
        "pytest.py",
        "src/pkg/PyTest.py",
        "pytest/__main__.py",
        "_pytest/config.py",
        "pytest_jsonreport/plugin.py",
        "sitecustomize.py",
        "src/pkg/usercustomize.py",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "noxfile.py",
        "pytest.ini",
        "ruff.toml",
        "uv.lock",
        "requirements-dev.txt",
        "Dockerfile",
        "dockerfile.release",
        "Makefile",
        ".gitlab-ci.yml",
        ".circleci/config.yml",
        "Jenkinsfile",
        "package.json",
        "Cargo.toml",
        "go.mod",
        ".github/workflows/ci.yml",
    ],
)
def test_protected_paths(rel):
    assert policy.is_protected(rel)


@pytest.mark.parametrize(
    "rel",
    ["src/app.py", "mathutils.py", "README.md", "pkg/testing_utils.py", "docs/latest.md"],
)
def test_unprotected_paths(rel):
    assert not policy.is_protected(rel)


def test_check_patch_reads_the_diff_not_just_declared_files():
    diff = (
        "diff --git a/tests/test_x.py b/tests/test_x.py\n"
        "--- a/tests/test_x.py\n"
        "+++ b/tests/test_x.py\n"
        "@@ -1 +1 @@\n-assert False\n+assert True\n"
    )
    patch = Patch(step_id="s", unified_diff=diff, touched_files=["src/app.py"])
    assert "tests/test_x.py" in policy.check_patch(patch)


def test_check_patch_respects_explicit_authorization():
    patch = Patch(step_id="s", file_contents={"tests/test_x.py": "ok"})
    assert policy.check_patch(patch) == ["tests/test_x.py"]
    assert policy.check_patch(patch, ["tests/test_x.py"]) == []


def test_strip_protected_removes_only_protected():
    patch = Patch(
        step_id="s",
        file_contents={"src/app.py": "code", "conftest.py": "evil"},
        touched_files=["src/app.py", "conftest.py"],
    )
    stripped = policy.strip_protected(patch)
    assert set(stripped.file_contents) == {"src/app.py"}
    assert stripped.touched_files == ["src/app.py"]


# --- end to end: tampering cannot produce DONE --------------------------------
class _TamperingLLM:
    """A 'model' that always cheats: it rewrites the test to pass instead of
    fixing the bug."""

    name = "tampering"

    def propose_patch(self, step, bundle, workdir):
        return Patch(
            step_id=step.step_id,
            file_contents={
                "tests/test_mathutils.py": "def test_average():\n    assert True\n"
            },
            touched_files=["tests/test_mathutils.py"],
            rationale="make the tests pass",
        )

    def plan(self, task, template):
        return None


def _cfg(tmp_path: Path, **over) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
        **over,
    )


def _run_with_tampering(tmp_path, harness_cls, **cfg_over):
    task = hermetic_task("data/tasks/fix_average.yaml")
    h = harness_cls(_cfg(tmp_path, **cfg_over))
    inner = getattr(h, "_h", h)  # LangGraphHarness wraps a Harness
    inner.llm = _TamperingLLM()
    return h.run(task)


def test_tampering_model_ends_failed_with_oracle_intact(tmp_path):
    result = _run_with_tampering(tmp_path, Harness)
    assert result.status == "FAILED"

    workdir = Path(result.state.run_dir) / "workdir"
    # canonical oracle untouched: identical to the source repo's test file
    canonical = Path("data/sample_repo/tests/test_mathutils.py").read_text()
    assert (workdir / "tests/test_mathutils.py").read_text() == canonical
    # and the bug is still there — nothing pretended to fix it
    assert "len(values) - 1" in (workdir / "mathutils.py").read_text()

    # the refusal was recorded as a failing oracle-policy check
    verdict = Verdict.model_validate_json(
        (Path(result.state.run_dir) / "verify.json").read_text()
    )
    assert not verdict.passed
    assert any(c.name == "oracle-policy" for c in verdict.checks)


def test_tampering_model_fails_under_langgraph_too(tmp_path):
    pytest.importorskip("langgraph")
    from lha.runtime.langgraph_runner import LangGraphHarness

    result = _run_with_tampering(tmp_path, LangGraphHarness)
    assert result.status == "FAILED"
    workdir = Path(result.state.run_dir) / "workdir"
    canonical = Path("data/sample_repo/tests/test_mathutils.py").read_text()
    assert (workdir / "tests/test_mathutils.py").read_text() == canonical


def test_policy_refusal_feeds_repair_before_failing(tmp_path):
    result = _run_with_tampering(tmp_path, Harness)
    ledger = (Path(result.state.run_dir) / "ledger.jsonl").read_text()
    assert '"phase":"repair"' in ledger.replace(" ", "")  # it retried with feedback


def test_task_manifest_can_authorize_protected_paths(tmp_path):
    task = hermetic_task("data/tasks/fix_average.yaml").model_copy(
        update={"allowed_protected_files": ["tests/test_mathutils.py"]}
    )
    h = Harness(_cfg(tmp_path))
    h.llm = _TamperingLLM()
    result = h.run(task)
    # explicitly authorized: the patch applies and the (rewritten) suite passes.
    # This is the manifest's job — an auditable, per-task decision.
    assert result.status == "DONE"
