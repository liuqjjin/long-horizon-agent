"""Regression tests for bugs found by the adversarial review."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from lha.agents.verifier_agent import VerifierAgent
from lha.artifacts import Patch, Step
from lha.clock import now
from lha.live_context import freshness as fr
from lha.live_context.models import ContextItem, Provenance
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


def test_diff_apply_is_idempotent(tmp_path):
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
    # second application (e.g. on resume) must not raise "already applied"
    apply_patch(patch, tmp_path)
    assert target.read_text() == updated


def test_backup_persists_to_disk_and_reverts(tmp_path):
    from lha.tools.patch import Backup, load_backup, revert_patch, save_backup

    target = tmp_path / "m.py"
    target.write_text("orig\n")
    backup = Backup(originals={"m.py": "orig\n"})
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
