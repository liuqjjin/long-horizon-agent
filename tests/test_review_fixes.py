"""Regression tests for bugs found by the adversarial review."""

from __future__ import annotations

import os
import stat
from datetime import timedelta
from pathlib import Path

import pytest

from lha.agents.verifier_agent import VerifierAgent
from lha.artifacts import Patch, Step
from lha.clock import now
from lha.live_context import freshness as fr
from lha.live_context.models import ContextItem, Provenance
from lha.process_result import ProcResult
from lha.sandbox.base import run_bounded_process
from lha.tools.patch import apply_patch, make_unified_diff
from lha.verifiers import VerifyContext


def test_missing_verifier_fails_not_silently_passes(tmp_path):
    step = Step(
        step_id="s", kind="code", action="edit_code", goal="g", verifiers=["does_not_exist"]
    )
    verdict = VerifierAgent().verify(
        step, Patch(step_id="s"), VerifyContext(workdir=tmp_path, step=step)
    )
    assert verdict.passed is False
    assert any(c.name == "does_not_exist" and not c.passed for c in verdict.checks)


def test_diff_apply_rejects_an_unjournaled_duplicate(tmp_path):
    target = tmp_path / "f.py"
    original = "a = 1\nb = 2\n"
    target.write_text(original)
    updated = "a = 1\nb = 3\n"
    patch = Patch(
        step_id="s",
        unified_diff=make_unified_diff(original, updated, "f.py"),
        touched_files=["f.py"],
    )
    apply_patch(patch, tmp_path)
    assert target.read_text() == updated
    # Resume idempotency belongs to PatchTransaction. Silently accepting a raw
    # duplicate would hide state drift (especially for mode-only diffs).
    with pytest.raises(RuntimeError, match="git apply failed"):
        apply_patch(patch, tmp_path)
    assert target.read_text() == updated


def test_diff_apply_uses_a_bounded_scrubbed_control_process(tmp_path, monkeypatch):
    target = tmp_path / "f.py"
    original = "value = 1\n"
    updated = "value = 2\n"
    target.write_text(original)
    patch = Patch(
        step_id="s",
        unified_diff=make_unified_diff(original, updated, "f.py"),
        touched_files=["f.py"],
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-git")
    observed: dict[str, object] = {}

    def recording_runner(cmd, **kwargs):
        observed["cmd"] = cmd
        observed.update(kwargs)
        return run_bounded_process(cmd, **kwargs)

    monkeypatch.setattr("lha.sandbox.base.run_bounded_process", recording_runner)

    apply_patch(patch, tmp_path)

    assert target.read_text() == updated
    assert observed["start_new_session"] is True
    assert observed["output_bytes"] == 1024 * 1024
    assert "AWS_SECRET_ACCESS_KEY" not in observed["env"]
    assert observed["env"]["GIT_CONFIG_NOSYSTEM"] == "1"


def test_diff_apply_rejects_unsupported_process_groups_before_spawn(
    tmp_path, monkeypatch
):
    import lha.sandbox.base as sandbox_base

    target = tmp_path / "f.py"
    original = "value = 1\n"
    updated = "value = 2\n"
    target.write_text(original)
    patch = Patch(
        step_id="s",
        unified_diff=make_unified_diff(original, updated, "f.py"),
        touched_files=["f.py"],
    )

    def unexpected_spawn(*_args, **_kwargs):
        raise AssertionError("unsupported hosts must fail before spawning git")

    monkeypatch.setattr(
        sandbox_base,
        "process_group_cleanup_supported",
        lambda: False,
    )
    monkeypatch.setattr(sandbox_base, "run_bounded_process", unexpected_spawn)

    with pytest.raises(RuntimeError, match="requires POSIX process-group cleanup"):
        apply_patch(patch, tmp_path)

    assert target.read_text() == original


def test_target_git_probe_uses_an_absolute_control_executable(
    tmp_path,
    monkeypatch,
):
    import lha.agents.verifier_agent as verifier_module

    observed: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(cmd, *, cwd=None, env=None, **_kwargs):
        assert cwd is not None
        assert env is not None
        observed.append((cmd, env))
        if "--show-toplevel" in cmd:
            return ProcResult(0, f"{tmp_path}\n", "", 0.0)
        return ProcResult(0, f"{'a' * 40}\n", "", 0.0)

    monkeypatch.setenv("PATH", ".")
    monkeypatch.setattr(verifier_module, "run", fake_run)

    assert verifier_module._target_git_commit(tmp_path) == "a" * 40
    assert len(observed) == 2
    for argv, environment in observed:
        assert Path(argv[0]).is_absolute()
        assert all(
            Path(component).is_absolute()
            for component in environment["PATH"].split(os.pathsep)
            if component
        )


def test_backup_persists_to_disk_and_reverts(tmp_path):
    from lha.tools.patch import Backup, load_backup, revert_patch, save_backup

    target = tmp_path / "m.py"
    target.write_text("orig\n")
    backup = Backup(
        originals={"m.py": b"orig\n"},
        modes={"m.py": stat.S_IMODE(target.stat().st_mode)},
    )
    target.write_text("changed\n")  # simulate an applied patch

    save_backup(backup, tmp_path / "backups" / "s.json")
    # simulate a fresh process: only the disk copy survives
    loaded = load_backup(tmp_path / "backups" / "s.json")
    assert loaded is not None
    revert_patch(loaded, tmp_path)
    assert target.read_text() == "orig\n"


def test_freshness_resolves_absolute_locator(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("hi")
    indexed_long_ago = now() - timedelta(hours=1)
    item = ContextItem(
        text="hi",
        provenance=Provenance(source_kind="code", locator=str(f), indexed_at=indexed_long_ago),
    )
    # base_dir is wrong on purpose; an absolute locator must still resolve
    verdict = fr.assess(
        [item], index_version="v", indexed_at=indexed_long_ago, base_dir=Path("/nonexistent")
    )
    assert verdict.is_stale_flag
