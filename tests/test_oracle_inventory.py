"""Pristine Pytest inventory protects nonstandard tests and support files."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import lha.oracle_inventory as oracle_inventory_module
from lha.artifacts import Patch, Step
from lha.clock import now
from lha.harness.errors import CheckpointCorrupt
from lha.live_context.models import ContextBundle, ContextItem, Freshness, Provenance
from lha.llm.base import LLMClient
from lha.llm.trace import TracedLLM
from lha.oracle_inventory import (
    OracleInventoryError,
    build_pytest_oracle_inventory,
    validate_pytest_oracle_inventory,
)
from lha.pytest_evidence import DriverResult, InventoryResult
from lha.sandbox import ProcessCleanupUnconfirmed, TrustedLocalBackend
from lha.tools.patch import resolve_patch
from lha.tools.policy import check_resolved


def _custom_repo(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        'testpaths = ["quality"]\n'
        'python_files = ["checks_*.py"]\n'
        'addopts = ["extra/explicit_case.py"]\n'
    )
    (root / "quality").mkdir()
    (root / "quality" / "checks_behavior.py").write_text(
        "def test_configured_name():\n    assert True\n"
    )
    (root / "quality" / "helper.py").write_text("ORACLE_SECRET = 41\n")
    (root / "quality" / "expected.json").write_text('{"answer": 42}\n')
    (root / "extra").mkdir()
    (root / "extra" / "explicit_case.py").write_text(
        "def test_explicit_addopts_file():\n    assert True\n"
    )
    (root / "app.py").write_text("def answer():\n    return 42\n")


def test_inventory_combines_configured_tree_and_actual_addopts_nodeid(tmp_path):
    _custom_repo(tmp_path)

    inventory = build_pytest_oracle_inventory(
        tmp_path,
        TrustedLocalBackend(),
    )

    assert inventory.nodeids == (
        "extra/explicit_case.py::test_explicit_addopts_file",
    )
    assert set(inventory.protected_paths) == {
        "extra/explicit_case.py",
        "quality/checks_behavior.py",
        "quality/expected.json",
        "quality/helper.py",
    }
    assert all(len(item.sha256) == 64 for item in inventory.files)
    assert len(inventory.sha256) == 64

    resolved = resolve_patch(
        Patch(
            step_id="s",
            file_contents={"quality/helper.py": "ORACLE_SECRET = 0\n"},
        )
    )
    assert check_resolved(
        resolved,
        additional_protected_files=inventory,
    ) == ["quality/helper.py"]
    added = resolve_patch(
        Patch(
            step_id="s",
            file_contents={"quality/new_fixture.py": "VALUE = 0\n"},
        )
    )
    assert check_resolved(
        added,
        additional_protected_files=inventory,
    ) == ["quality/new_fixture.py"]

    (tmp_path / "quality" / "helper.py").write_text("ORACLE_SECRET = 0\n")
    with pytest.raises(OracleInventoryError, match="changed: quality/helper.py"):
        validate_pytest_oracle_inventory(tmp_path, inventory)


def test_repository_root_testpath_protects_the_complete_tree(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        'testpaths = ["."]\n'
    )
    (tmp_path / "test_behavior.py").write_text(
        "def test_behavior():\n    assert True\n"
    )
    (tmp_path / "app.py").write_text("VALUE = 1\n")

    inventory = build_pytest_oracle_inventory(
        tmp_path,
        TrustedLocalBackend(),
    )

    assert "app.py" in inventory.protected_paths
    resolved = resolve_patch(
        Patch(step_id="s", file_contents={"new_module.py": "VALUE = 2\n"})
    )
    assert check_resolved(
        resolved,
        additional_protected_files=inventory,
    ) == ["new_module.py"]


def test_collection_with_unconfirmed_cleanup_never_touches_canonical_workdir(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_behavior.py").write_text(
        "def test_behavior():\n    assert True\n"
    )
    collected_in: list[Path] = []

    def fake_collect(workdir, backend, *, timeout, autoload_plugins):
        del backend, timeout, autoload_plugins
        disposable = Path(workdir)
        collected_in.append(disposable)
        (disposable / "collection-side-effect.txt").write_text("left behind\n")
        return InventoryResult(
            expected_nodeids=(),
            protocol_valid=False,
            collection_failed=False,
            driver=DriverResult(
                returncode=126,
                receipt=None,
                receipt_sha256="",
                detail="cleanup unconfirmed",
                duration_s=0.01,
                cleanup_unconfirmed=True,
                cleanup_detail="descendant still present",
            ),
        )

    monkeypatch.setattr(
        oracle_inventory_module,
        "collect_inventory",
        fake_collect,
    )
    try:
        with pytest.raises(ProcessCleanupUnconfirmed, match="cleanup could not be confirmed"):
            build_pytest_oracle_inventory(tmp_path, TrustedLocalBackend())

        assert collected_in
        assert collected_in[0] != tmp_path
        assert not (tmp_path / "collection-side-effect.txt").exists()
        assert (collected_in[0] / "collection-side-effect.txt").exists()
    finally:
        if collected_in:
            shutil.rmtree(collected_in[0].parent, ignore_errors=True)


def test_without_testpaths_collected_module_support_directory_is_protected(
    tmp_path,
):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        'python_files = ["checks_*.py"]\n'
    )
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "checks_behavior.py").write_text(
        "def test_behavior():\n    assert True\n"
    )
    (tmp_path / "quality" / "helper.py").write_text("EXPECTED = 42\n")
    (tmp_path / "quality" / "expected.json").write_text('{"answer": 42}\n')
    (tmp_path / "app.py").write_text("def answer():\n    return 42\n")

    inventory = build_pytest_oracle_inventory(
        tmp_path,
        TrustedLocalBackend(),
    )

    assert inventory.configured_testpaths == ()
    assert inventory.support_roots == ("quality",)
    assert set(inventory.protected_paths) == {
        "quality/checks_behavior.py",
        "quality/expected.json",
        "quality/helper.py",
    }
    assert "app.py" not in inventory.protected_paths
    added = resolve_patch(
        Patch(
            step_id="s",
            file_contents={"quality/new_expected.json": "{}\n"},
        )
    )
    assert check_resolved(
        added,
        additional_protected_files=inventory,
    ) == ["quality/new_expected.json"]


class _CapturePrompt(LLMClient):
    name = "capture"

    def __init__(self):
        self.prompt = ""

    def complete(self, system: str, prompt: str) -> str:
        self.prompt = prompt
        return ""


def test_traced_llm_does_not_send_inventory_files_in_source_prompt(tmp_path):
    _custom_repo(tmp_path)
    inventory = build_pytest_oracle_inventory(
        tmp_path,
        TrustedLocalBackend(),
    )
    inner = _CapturePrompt()
    client = TracedLLM(inner)
    client.set_trusted_oracle_paths(inventory)
    step = Step(step_id="s", kind="code", action="edit_code", goal="fix answer")
    bundle = ContextBundle(
        query="answer",
        freshness=Freshness(index_version="v", indexed_at=now()),
        items=[
            ContextItem(
                text="ORACLE_CONTEXT_SECRET = 99",
                provenance=Provenance(
                    source_kind="code",
                    locator="quality/helper.py:1-1",
                ),
            )
        ],
    )

    client.propose_patch(step, bundle, tmp_path)

    assert "def answer()" in inner.prompt
    assert "ORACLE_SECRET" not in inner.prompt
    assert "ORACLE_CONTEXT_SECRET" not in inner.prompt
    assert "test_explicit_addopts_file" not in inner.prompt
    assert "test_configured_name" not in inner.prompt


def test_traced_patch_journal_binds_trusted_oracle_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _custom_repo(repo)
    step = Step(step_id="s", kind="code", action="edit_code", goal="fix answer")
    bundle = ContextBundle(
        query="answer",
        freshness=Freshness(index_version="v", indexed_at=now()),
    )
    run_dir = tmp_path / "run"
    first = TracedLLM(_CapturePrompt()).bind(run_dir)
    first.set_call_context(attempt_id="s-r0")
    first.set_trusted_oracle_paths(["quality/helper.py"])
    first.propose_patch(step, bundle, repo)

    replay = TracedLLM(_CapturePrompt()).bind(run_dir)
    replay.restore_totals(first.totals)
    replay.set_call_context(attempt_id="s-r0")
    replay.set_trusted_oracle_paths(["extra/explicit_case.py"])
    with pytest.raises(CheckpointCorrupt, match="does not match"):
        replay.propose_patch(step, bundle, repo)
