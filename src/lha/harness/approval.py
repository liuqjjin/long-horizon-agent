"""Human-approval gate. File-based so it survives pause/resume.

When a step requires approval the loop writes ``pending_approval.json`` and
pauses (status AWAITING_APPROVAL). A separate ``lha approve|reject <run_id>``
writes ``approval.json``; ``lha resume`` then continues. This mirrors a
LangGraph ``interrupt()`` / ``Command(resume=...)`` round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ApprovalDecision(BaseModel):
    approved: bool
    note: str = ""


class HumanApprovalGate:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.pending = self.run_dir / "pending_approval.json"
        self.decision_file = self.run_dir / "approval.json"

    def request(self, step, summary: str) -> None:
        self.pending.write_text(
            json.dumps(
                {"step_id": step.step_id, "goal": step.goal, "summary": summary},
                indent=2,
            )
        )

    def decision(self) -> ApprovalDecision | None:
        if self.decision_file.exists():
            return ApprovalDecision.model_validate_json(self.decision_file.read_text())
        return None

    def resolve(self, approved: bool, note: str = "") -> None:
        self.decision_file.write_text(
            ApprovalDecision(approved=approved, note=note).model_dump_json(indent=2)
        )
        self.pending.unlink(missing_ok=True)

    def clear(self) -> None:
        self.pending.unlink(missing_ok=True)
        self.decision_file.unlink(missing_ok=True)
