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

import json
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.config import Config
from lha.harness import Harness
from lha.harness.approval import ApprovalDecision, HumanApprovalGate
from lha.harness.manifest import ArtifactManifest, sha256_bytes

APPROVAL_TASK = "data/tasks/fix_average_approval.yaml"


def _cfg(tmp_path: Path, **over) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
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


def test_decision_hash_tampered_fails(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")
    decision = json.loads((run_dir / "approval.json").read_text())
    decision["artifact_sha256"] = "0" * 64
    (run_dir / "approval.json").write_text(json.dumps(decision))

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"


def test_decision_without_step_or_hash_is_not_honored(tmp_path):
    paused = _pause_at_approval(tmp_path)
    run_dir = Path(paused.state.run_dir)
    (run_dir / "approval.json").write_text(ApprovalDecision(approved=True).model_dump_json())

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "AWAITING_APPROVAL"  # unusable decision was discarded


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


def test_langgraph_rejection_reverts_and_fails(tmp_path, lg):
    paused = _pause_at_approval(tmp_path, lg)
    run_dir = Path(paused.state.run_dir)
    HumanApprovalGate(run_dir).resolve(approved=False, note="no")
    resumed = lg(_cfg(tmp_path)).resume(paused.state.run_id)
    assert resumed.status == "FAILED"
    assert "len(values) - 1" in (run_dir / "workdir" / "mathutils.py").read_text()


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
