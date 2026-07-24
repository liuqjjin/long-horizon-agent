"""Human-approval gate. File-based so it survives pause/resume.

When a step requires approval the loop writes ``pending_approval.json`` —
including the SHA-256 of the exact artifact under review — and pauses (status
AWAITING_APPROVAL). A separate ``lha approve|reject <run_id>`` writes
``approval.json``, copying the step id and artifact hash from the pending
request; ``lha resume`` then continues, and only the artifact whose hash the
decision carries may be executed. This mirrors a LangGraph ``interrupt()`` /
``Command(resume=...)`` round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ApprovalDecision(BaseModel):
    approved: bool
    note: str = ""
    # Which step and which exact artifact this decision was made for. A decision
    # that names neither is unusable (fail closed); a mismatch on either must
    # never be honored.
    step_id: str | None = None
    artifact_sha256: str | None = None

    def binds(self, step_id: str, artifact_sha256: str | None) -> bool:
        """Whether this decision is bound to exactly this step + artifact."""
        if self.step_id != step_id:
            return False
        if self.artifact_sha256 is None or artifact_sha256 is None:
            return False
        return self.artifact_sha256 == artifact_sha256


class HumanApprovalGate:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.pending = self.run_dir / "pending_approval.json"
        self.decision_file = self.run_dir / "approval.json"

    def request(self, step, summary: str, *, artifact_sha256: str | None = None) -> None:
        self.pending.write_text(
            json.dumps(
                {
                    "step_id": step.step_id,
                    "goal": step.goal,
                    "summary": summary,
                    "artifact_sha256": artifact_sha256,
                },
                indent=2,
            )
        )

    def decision(self) -> ApprovalDecision | None:
        if self.decision_file.exists():
            try:
                return ApprovalDecision.model_validate_json(self.decision_file.read_text())
            except Exception:
                return None  # a corrupt decision is no decision (fail closed)
        return None

    def resolve(self, approved: bool, note: str = "") -> None:
        # Bind the decision to the step AND artifact it answers, read from the
        # pending request, so a stale decision can't be misattributed to a later
        # step or a regenerated patch.
        step_id = None
        artifact_sha256 = None
        if self.pending.exists():
            try:
                pending = json.loads(self.pending.read_text())
                step_id = pending.get("step_id")
                artifact_sha256 = pending.get("artifact_sha256")
            except (json.JSONDecodeError, OSError):
                pass
        self.decision_file.write_text(
            ApprovalDecision(
                approved=approved,
                note=note,
                step_id=step_id,
                artifact_sha256=artifact_sha256,
            ).model_dump_json(indent=2)
        )
        self.pending.unlink(missing_ok=True)

    def clear(self) -> None:
        self.pending.unlink(missing_ok=True)
        self.decision_file.unlink(missing_ok=True)
