"""Approval is bound to an immutable artifact, not to "whatever is on disk".

Invariants:
  - the pending request and the decision carry the SHA-256 of the exact
    patch.json bytes under review;
  - a patch tampered after approval fails the run (never applied);
  - a decision whose hash was altered fails the run;
  - a decision naming no step / another step is never honored;
  - the LangGraph runtime cannot regenerate a patch on approval resume
    (execute lives in a node checkpointed before the interrupt);
  - an ArtifactManifest records the artifact hash, touched files, and the
    pre-apply base state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.config import Config
from lha.harness import Harness
from lha.harness.approval import (
    ApprovalDecision,
    HumanApprovalGate,
    approval_decision_path,
    approval_decision_ref,
    approval_request_path,
    read_approval_decision,
)
from lha.harness.checkpoint import append_ledger, load_state, read_ledger
from lha.harness.manifest import ArtifactManifest, sha256_bytes
from lha.harness.state import StepRecord
from lha.harness.transaction import list_transactions
from lha.reporting import ReportingError, collect_run

APPROVAL_TASK = "data/tasks/fix_average_approval.yaml"


def _cfg(tmp_path: Path, **over) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
        use_skill_memory=False,
        **over,
    )


def _pause_at_approval(tmp_path, harness_cls=Harness):
    h = harness_cls(_cfg(tmp_path))
    paused = h.run(hermetic_task(APPROVAL_TASK))
    assert paused.status == "AWAITING_APPROVAL"
    return paused


def _step_patch(run_dir: str | Path) -> Path:
    return Path(run_dir) / "steps" / "s2-fix" / "patch.json"


# --- the request/decision carry the artifact hash ----------------------------
def test_pending_request_and_decision_carry_artifact_hash(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    pending = json.loads((run_dir / "pending_approval.json").read_text())
    expected = sha256_bytes(_step_patch(run_dir).read_bytes())
    assert pending["artifact_sha256"] == expected

    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")
    decision = ApprovalDecision.model_validate_json((run_dir / "approval.json").read_text())
    assert decision.artifact_sha256 == expected
    assert decision.step_id == "s2-fix"


def test_request_and_decision_are_checksummed_attempt_evidence(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    attempt_id = "s2-fix-r0"
    request_path = approval_request_path(run_dir, "s2-fix", attempt_id)
    request_bytes = request_path.read_bytes()
    request_envelope = json.loads(request_bytes)
    assert request_envelope["schema_version"] == 1
    assert request_envelope["payload"]["attempt_id"] == attempt_id
    assert request_envelope["payload"]["goal"] == paused.state.next_step().goal

    resolved = HumanApprovalGate(run_dir).resolve(
        approved=True, note="reviewed exact bytes"
    )
    decision_path = approval_decision_path(run_dir, "s2-fix", attempt_id)
    decision_envelope = json.loads(decision_path.read_bytes())
    assert decision_envelope["schema_version"] == 1
    assert decision_envelope["payload"]["request_sha256"] == sha256_bytes(
        request_bytes
    )
    assert decision_envelope["payload"]["outcome"] == "approved"
    assert resolved.sha256 == sha256_bytes(decision_path.read_bytes())


def test_completed_run_keeps_decision_after_alias_cleanup(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(
        approved=True, note="approved after review"
    )
    done = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert done.status == "DONE"
    assert not (run_dir / "pending_approval.json").exists()
    assert not (run_dir / "approval.json").exists()

    report = collect_run(run_dir.parent, done.state.run_id)
    decisions = [
        item.value
        for item in report.approvals
        if isinstance(item.value, ApprovalDecision)
    ]
    assert len(decisions) == 1
    assert decisions[0].outcome == "approved"
    assert decisions[0].note == "approved after review"
    record = next(item for item in report.ledger if item.phase == "approval")
    evidence = read_approval_decision(run_dir, "s2-fix", "s2-fix-r0")
    assert evidence is not None
    assert record.attempt_id == "s2-fix-r0"
    assert record.artifact_ref == approval_decision_ref("s2-fix-r0")
    assert record.evidence_sha256 == evidence.sha256


def test_rejection_is_reportable_after_transient_cleanup(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(
        approved=False, note="unsafe change"
    )
    failed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert failed.status == "FAILED"
    report = collect_run(run_dir.parent, failed.state.run_id)
    decision = next(
        item.value
        for item in report.approvals
        if isinstance(item.value, ApprovalDecision)
    )
    assert decision.outcome == "rejected"
    assert decision.note == "unsafe change"
    assert [record.phase for record in report.ledger[-2:]] == [
        "approval",
        "fail",
    ]


def test_manifest_records_artifact_and_base_state(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    manifest = ArtifactManifest.model_validate_json(
        (run_dir / "steps" / "s2-fix" / "manifest.json").read_text()
    )
    assert manifest.step_id == "s2-fix"
    assert manifest.artifact_sha256 == sha256_bytes(_step_patch(run_dir).read_bytes())
    assert "mathutils.py" in manifest.touched_files
    # base_state hashes the PRE-apply file, which contains the bug
    assert manifest.base_state["mathutils.py"] is not None


# --- tampering after review fails closed --------------------------------------
def test_patch_tampered_after_approval_fails_and_reverts(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")

    # attacker rewrites the reviewed artifact before the resume
    tampered = json.loads(_step_patch(run_dir).read_text())
    tampered["file_contents"] = {"mathutils.py": "def average(v):\n    return 0\n"}
    for path in (_step_patch(run_dir), run_dir / "patch.json"):
        path.write_text(json.dumps(tampered, indent=2))

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"
    assert "hash mismatch" in resumed.message
    # the previously-applied (reviewed) change was reverted; the tampered one never landed
    src = (run_dir / "workdir" / "mathutils.py").read_text()
    assert "return 0" not in src
    assert "len(values) - 1" in src  # original bug restored


@pytest.mark.parametrize("damage", ["missing", "policy"])
def test_attempt_manifest_damage_fails_and_reverts(tmp_path, damage):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    tx = list_transactions(run_dir, "s2-fix")[0]
    manifest_path = (
        run_dir
        / "steps"
        / "s2-fix"
        / "attempts"
        / tx.attempt_id
        / "manifest.json"
    )
    if damage == "missing":
        manifest_path.unlink()
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["policy_overrides"] = ["tests/test_mathutils.py"]
        manifest_path.write_text(json.dumps(manifest))
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"
    assert "manifest" in resumed.message
    assert "len(values) - 1" in (
        run_dir / "workdir" / "mathutils.py"
    ).read_text()


def test_review_diff_tampering_fails_and_reverts(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    review = run_dir / "steps" / "s2-fix" / "patch.diff"
    assert "sum(values) / len(values)" in review.read_text()
    review.write_text("--- a/mathutils.py\n+++ b/mathutils.py\n+misleading\n")
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"
    assert "review artifact" in resumed.message
    assert "len(values) - 1" in (
        run_dir / "workdir" / "mathutils.py"
    ).read_text()


def test_decision_hash_tampered_fails(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")
    decision = json.loads((run_dir / "approval.json").read_text())
    decision["artifact_sha256"] = "0" * 64
    (run_dir / "approval.json").write_text(json.dumps(decision))

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"


@pytest.mark.parametrize("name", ["approval_request.json", "approval_decision.json"])
def test_immutable_approval_tampering_fails_closed(tmp_path, name):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    if name == "approval_decision.json":
        HumanApprovalGate(run_dir).resolve(approved=True, note="ok")
    path = (
        run_dir
        / "steps"
        / "s2-fix"
        / "attempts"
        / "s2-fix-r0"
        / name
    )
    envelope = json.loads(path.read_text())
    envelope["payload"]["goal" if name == "approval_request.json" else "note"] = "tampered"
    path.write_text(json.dumps(envelope))

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"
    assert "len(values) - 1" in (
        run_dir / "workdir" / "mathutils.py"
    ).read_text()


def test_reporting_rejects_terminal_decision_tampering(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")
    done = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert done.status == "DONE"
    path = approval_decision_path(run_dir, "s2-fix", "s2-fix-r0")
    envelope = json.loads(path.read_text())
    envelope["payload"]["note"] = "changed after completion"
    path.write_text(json.dumps(envelope))

    with pytest.raises(ReportingError, match="approval evidence"):
        collect_run(run_dir.parent, done.state.run_id)


def test_reporting_rejects_missing_terminal_decision(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")
    done = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert done.status == "DONE"
    approval_decision_path(
        run_dir, "s2-fix", "s2-fix-r0"
    ).unlink()

    with pytest.raises(ReportingError, match="approval"):
        collect_run(run_dir.parent, done.state.run_id)


def test_reporting_rejects_resigned_misbound_decision(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")
    done = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert done.status == "DONE"
    path = approval_decision_path(run_dir, "s2-fix", "s2-fix-r0")
    envelope = json.loads(path.read_text())
    envelope["payload"]["request_sha256"] = "0" * 64
    canonical = json.dumps(
        envelope["payload"], sort_keys=True, separators=(",", ":")
    ).encode()
    envelope["sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(envelope))

    with pytest.raises(ReportingError, match="does not bind"):
        collect_run(run_dir.parent, done.state.run_id)


def test_resume_rejects_symlinked_approval_evidence(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    path = approval_request_path(run_dir, "s2-fix", "s2-fix-r0")
    target = tmp_path / "request.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"
    assert target.is_file()


def test_reporting_rejects_extra_approval_evidence(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")
    done = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert done.status == "DONE"
    assert done.state.plan is not None
    non_gated = done.state.plan.steps[0]
    gate = HumanApprovalGate(run_dir)
    gate.request(
        non_gated,
        "s1-context-r0",
        "fabricated extra review",
    )
    gate.resolve(approved=True, note="extra")
    gate.clear_transient()

    with pytest.raises(ReportingError, match="approval evidence set"):
        collect_run(run_dir.parent, done.state.run_id)


def test_resume_recovers_decision_ledgered_before_checkpoint_and_cleanup(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    gate = HumanApprovalGate(run_dir)
    evidence = gate.resolve(approved=True, note="crash-window")
    state = load_state(run_dir)
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id="s2-fix",
            phase="approval",
            artifact_ref=evidence.reference,
            evidence_sha256=evidence.sha256,
            attempt_id="s2-fix-r0",
            idempotency_key="s2-fix-r0:approval",
            notes="approved: crash-window",
        ),
    )
    # Model a process death before state.json catches up and aliases are cleared.
    done = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert done.status == "DONE"
    approvals = [
        record
        for record in read_ledger(run_dir)
        if record.phase == "approval"
    ]
    assert len(approvals) == 1
    assert not (run_dir / "approval.json").exists()


def test_decision_without_step_or_hash_is_not_honored(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    (run_dir / "approval.json").write_text(ApprovalDecision(approved=True).model_dump_json())

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"


# --- LangGraph: approval cannot regenerate the artifact -----------------------
@pytest.fixture
def lg():
    pytest.importorskip("langgraph")
    from lha.runtime.langgraph_runner import LangGraphHarness

    return LangGraphHarness


def test_langgraph_pauses_and_resumes_via_approval(tmp_path, lg, monkeypatch):
    paused = _pause_at_approval(tmp_path, lg)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")

    # If resume re-ran the implementer, this would raise and the run would FAIL.
    from lha.agents import implementer

    def boom(self, *a, **k):
        raise RuntimeError("must not regenerate a patch on approval resume")

    monkeypatch.setattr(implementer.Implementer, "implement", boom)
    done = lg(_cfg(tmp_path)).resume(paused.state.run_id)
    assert done.status == "DONE"
    fixed = (run_dir / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" not in fixed  # the reviewed fix is in place
    report = collect_run(run_dir.parent, done.state.run_id)
    decision = next(
        item.value
        for item in report.approvals
        if isinstance(item.value, ApprovalDecision)
    )
    assert decision.note == "ok"


def test_langgraph_rejection_reverts_and_fails(tmp_path, lg):
    paused = _pause_at_approval(tmp_path, lg)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=False, note="no")
    resumed = lg(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"
    assert "len(values) - 1" in (run_dir / "workdir" / "mathutils.py").read_text()
    report = collect_run(run_dir.parent, resumed.state.run_id)
    decision = next(
        item.value
        for item in report.approvals
        if isinstance(item.value, ApprovalDecision)
    )
    assert decision.outcome == "rejected"
    assert decision.note == "no"


def test_langgraph_tampered_artifact_fails_closed(tmp_path, lg):
    paused = _pause_at_approval(tmp_path, lg)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")

    tampered = json.loads(_step_patch(run_dir).read_text())
    tampered["file_contents"] = {"mathutils.py": "def average(v):\n    return 0\n"}
    for path in (_step_patch(run_dir), run_dir / "patch.json"):
        path.write_text(json.dumps(tampered, indent=2))

    resumed = lg(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"
    src = (run_dir / "workdir" / "mathutils.py").read_text()
    assert "return 0" not in src
