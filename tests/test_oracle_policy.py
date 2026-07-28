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
from lha.harness.approval import HumanApprovalGate
from lha.tools import policy
from lha.tools.patch import resolve_patch
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
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
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


def test_custom_pytest_collection_paths_are_protected(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        'testpaths = ["quality"]\n'
        'python_files = ["checks_*.py"]\n'
    )
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "checks_behavior.py").write_text(
        "def test_behavior():\n    assert False\n"
    )
    (tmp_path / "quality" / "helper.py").write_text("VALUE = 1\n")

    discovered = policy.discover_pytest_oracle_paths(tmp_path)

    assert discovered == ["quality/checks_behavior.py"]
    resolved = resolve_patch(
        Patch(
            step_id="s",
            file_contents={"quality/checks_behavior.py": "def test_behavior(): pass\n"},
        )
    )
    assert policy.check_resolved(
        resolved,
        additional_protected_files=discovered,
    ) == ["quality/checks_behavior.py"]


class _CustomCollectorTamperingLLM:
    name = "custom-collector-tampering"

    def propose_patch(self, step, bundle, workdir):
        return Patch(
            step_id=step.step_id,
            file_contents={
                "quality/checks_behavior.py": "def test_behavior():\n    assert True\n"
            },
            touched_files=["quality/checks_behavior.py"],
        )

    def plan(self, task, template):
        return None


def test_harness_refuses_patch_to_custom_collected_test(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        'testpaths = ["quality"]\n'
        'python_files = ["checks_*.py"]\n'
    )
    (repo / "quality").mkdir()
    oracle = repo / "quality" / "checks_behavior.py"
    oracle.write_text("def test_behavior():\n    assert False\n")
    task = hermetic_task("data/tasks/fix_average.yaml").model_copy(
        update={"target_repo": str(repo)}
    )
    harness = Harness(_cfg(tmp_path))
    harness.llm = _CustomCollectorTamperingLLM()

    result = harness.run(task)

    assert result.status == "FAILED"
    run_oracle = Path(result.state.workdir) / "quality" / "checks_behavior.py"
    assert run_oracle.read_text() == oracle.read_text()


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


@pytest.mark.parametrize("runtime", ["loop", "langgraph"])
def test_authorized_test_change_survives_approval_resume(
    runtime: str,
    tmp_path: Path,
) -> None:
    config = _cfg(tmp_path)
    if runtime == "langgraph":
        pytest.importorskip("langgraph")
        from lha.runtime.langgraph_runner import LangGraphHarness

        harness = LangGraphHarness(config)
    else:
        harness = Harness(config)
    inner = getattr(harness, "_h", harness)
    inner.llm = _TamperingLLM()
    task = hermetic_task(
        "data/tasks/fix_average_approval.yaml"
    ).model_copy(
        update={
            "allowed_protected_files": [
                "tests/test_mathutils.py",
            ]
        }
    )

    paused = harness.run(task)

    assert paused.status == "AWAITING_APPROVAL"
    HumanApprovalGate(paused.state.run_dir).resolve(
        approved=True,
        note="authorized test change",
    )
    if runtime == "langgraph":
        resumed = LangGraphHarness(config).resume(paused.state.run_id)
    else:
        resumed = Harness(config).resume(paused.state.run_id)

    assert resumed.status == "DONE", resumed.message
    changed = (
        Path(resumed.state.workdir) / "tests" / "test_mathutils.py"
    ).read_text()
    assert changed == "def test_average():\n    assert True\n"
