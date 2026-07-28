"""Adversarial inode tests for approval and loop-owned run evidence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import lha.durable_io as durable_io
from lha.artifacts import Step
from lha.harness.approval import (
    HumanApprovalGate,
    approval_decision_path,
    approval_request_path,
    read_approval_decision,
    read_approval_request,
)
from lha.harness.errors import CheckpointCorrupt
from lha.harness.loop import (
    _dump,
    _validate_optional_aliases,
    _write_immutable,
)


def _approval_step() -> Step:
    return Step(
        step_id="review",
        kind="code",
        action="edit_code",
        goal="review exact patch bytes",
        requires_approval=True,
    )


def _replace_with_external_hardlink(path: Path, external: Path) -> bytes:
    payload = path.read_bytes()
    external.write_bytes(payload)
    path.unlink()
    os.link(external, path)
    return payload


@pytest.mark.parametrize("kind", ["request", "decision"])
def test_approval_authority_rejects_external_hardlink(
    tmp_path: Path,
    kind: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = HumanApprovalGate(run_dir)
    step = _approval_step()
    gate.request(step, "review-r0", "review")
    if kind == "decision":
        gate.resolve(approved=True, note="ok")
        path = approval_decision_path(run_dir, step.step_id, "review-r0")
        reader = read_approval_decision
    else:
        path = approval_request_path(run_dir, step.step_id, "review-r0")
        reader = read_approval_request

    external = tmp_path / f"external-{kind}.json"
    before = _replace_with_external_hardlink(path, external)

    with pytest.raises(ValueError, match="invalid approval evidence"):
        reader(run_dir, step.step_id, "review-r0")

    assert external.read_bytes() == before


def test_approval_lock_rejects_external_hardlink_without_overwrite(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = HumanApprovalGate(run_dir)
    gate.request(_approval_step(), "review-r0", "review")
    external = tmp_path / "external-lock"
    external.write_bytes(b"outside-lock")
    os.link(external, run_dir / ".approval.lock")

    with pytest.raises(ValueError, match="unsafe approval lock"):
        gate.resolve(approved=True)

    assert external.read_bytes() == b"outside-lock"


def test_approval_alias_rejects_external_hardlink_without_overwrite(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = HumanApprovalGate(run_dir)
    gate.request(_approval_step(), "review-r0", "review")
    pending = run_dir / "pending_approval.json"
    external = tmp_path / "external-pending.json"
    before = _replace_with_external_hardlink(pending, external)

    with pytest.raises(ValueError, match="unsafe approval alias"):
        gate.resolve(approved=True)

    assert external.read_bytes() == before


def test_approval_decision_alias_does_not_overwrite_external_hardlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = HumanApprovalGate(run_dir)
    gate.request(_approval_step(), "review-r0", "review")
    external = tmp_path / "external-decision.json"
    external.write_bytes(b"outside-decision")
    os.link(external, run_dir / "approval.json")

    with pytest.raises(ValueError, match="unsafe approval path"):
        gate.resolve(approved=True)

    assert external.read_bytes() == b"outside-decision"


@pytest.mark.parametrize("alias_name", ["pending", "decision"])
@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_approval_clear_rejects_linked_alias_without_touching_external_file(
    tmp_path: Path,
    alias_name: str,
    link_kind: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = HumanApprovalGate(run_dir)
    gate.request(_approval_step(), "review-r0", "review")
    if alias_name == "decision":
        gate.resolve(approved=True, note="ok")
        alias = run_dir / "approval.json"
    else:
        alias = run_dir / "pending_approval.json"
    external = tmp_path / f"external-{alias_name}-{link_kind}.json"
    external.write_bytes(alias.read_bytes())
    alias.unlink()
    if link_kind == "hardlink":
        os.link(external, alias)
    else:
        alias.symlink_to(external)
    before = external.read_bytes()

    with pytest.raises(ValueError, match="unsafe approval alias"):
        gate.clear_transient()

    assert external.read_bytes() == before
    if link_kind == "hardlink":
        assert external.stat().st_nlink == 2
    else:
        assert alias.is_symlink()


def test_approval_clear_allows_both_aliases_to_be_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    HumanApprovalGate(run_dir).clear_transient()

    assert not (run_dir / "pending_approval.json").exists()
    assert not (run_dir / "approval.json").exists()


def test_approval_clear_rejects_name_replacement_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = HumanApprovalGate(run_dir)
    gate.request(_approval_step(), "review-r0", "review")
    pending = run_dir / "pending_approval.json"
    external = tmp_path / "external-raced-pending.json"
    external.write_bytes(pending.read_bytes())
    before = external.read_bytes()
    real_open = durable_io.os.open
    raced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal raced
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == pending.name and not raced:
            raced = True
            pending.unlink()
            os.link(external, pending)
        return descriptor

    monkeypatch.setattr(durable_io.os, "open", racing_open)

    with pytest.raises(ValueError, match="unsafe approval alias"):
        gate.clear_transient()

    assert raced
    assert external.read_bytes() == before
    assert pending.samefile(external)


@pytest.mark.parametrize(
    "relative",
    [
        Path("plans/initial.json"),
        Path("steps/review/attempts/review-r0/patch.json"),
        Path("steps/review/attempts/review-r0/manifest.json"),
        Path("steps/review/attempts/review-r0/verify.json"),
    ],
)
def test_loop_write_once_evidence_rejects_external_hardlink(
    tmp_path: Path,
    relative: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / relative
    path.parent.mkdir(parents=True)
    external = tmp_path / f"outside-{path.name}"
    external.write_bytes(b"outside")
    os.link(external, path)

    with pytest.raises(CheckpointCorrupt, match="changed or is unsafe"):
        _write_immutable(path, b"trusted", run_dir=run_dir)

    assert external.read_bytes() == b"outside"


def test_loop_write_once_evidence_allows_identical_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "plans" / "initial.json"

    _write_immutable(path, b"trusted", run_dir=run_dir)
    _write_immutable(path, b"trusted", run_dir=run_dir)

    assert path.read_bytes() == b"trusted"
    assert path.stat().st_nlink == 1


def test_loop_alias_read_rejects_external_hardlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    external = tmp_path / "outside-alias"
    external.write_bytes(b"trusted")
    alias = run_dir / "patch.json"
    os.link(external, alias)

    with pytest.raises(CheckpointCorrupt, match="alias is missing or unsafe"):
        _validate_optional_aliases(
            run_dir,
            b"trusted",
            [alias],
            label="patch",
        )

    assert external.read_bytes() == b"trusted"


def test_loop_display_alias_does_not_replace_external_hardlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    external = tmp_path / "outside-display"
    external.write_bytes(b"outside")
    os.link(external, run_dir / "verify.json")

    with pytest.raises(OSError, match="unsafe"):
        _dump(run_dir, "review", "verify.json", "trusted")

    assert external.read_bytes() == b"outside"
