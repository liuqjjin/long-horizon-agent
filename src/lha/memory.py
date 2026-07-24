"""Skill memory: distill each verified success into a note.

After a run is verified DONE, a markdown skill note is written under
``data/skills/`` (gitignored — it is accumulated runtime memory, not source).
``index_docs`` indexes it and the Context Engineer can retrieve it for similar
future tasks. Only genuine successes are recorded, so memory doesn't teach
failures.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import ExperimentResult, Patch
from .clock import now
from .verifiers.verdict import Verdict


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "skill"


class SkillMemory:
    def __init__(self, skills_dir: str | Path = "data/skills"):
        self.skills_dir = Path(skills_dir)

    def record(self, state: Any) -> Path | None:
        if getattr(state, "status", None) != "DONE":
            return None
        run_dir = Path(state.run_dir)

        verify_json = run_dir / "verify.json"
        if not verify_json.exists():
            return None
        verdict = Verdict.model_validate_json(verify_json.read_text())
        if not verdict.passed:
            return None  # only record genuine successes
        checks = [f"{c.name}: {c.detail.get('summary', 'passed')}" for c in verdict.checks]

        approach: list[str] = []
        files: list[str] = []
        patch_json = run_dir / "patch.json"
        if patch_json.exists():
            patch = Patch.model_validate_json(patch_json.read_text())
            if patch.rationale:
                approach.append(patch.rationale)
            files = patch.touched_files
        exp_json = run_dir / "experiment.json"
        if exp_json.exists():
            exp = ExperimentResult.model_validate_json(exp_json.read_text())
            approach.append("experiment command: " + " ".join(exp.command))

        task = state.task
        # Provenance: which exact verdict justified this note. A skill retrieved
        # later can be traced back to (and re-checked against) its evidence.
        verdict_sha = hashlib.sha256(verify_json.read_bytes()).hexdigest()
        body = self._render(task, approach, files, checks, state.run_id, verdict_sha)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        path = self.skills_dir / f"{_slug(task.title)}.md"  # stable per task -> updated in place
        path.write_text(body)
        return path

    @staticmethod
    def _render(task, approach, files, checks, skill_id, verdict_sha: str) -> str:
        lines = [
            "---",
            f'title: "{task.title}"',
            f"task_kind: {task.kind}",
            f"skill_id: {skill_id}",
            "verified: true",
            f"verdict_sha256: {verdict_sha}",
            f"harness_version: {__version__}",
            f'created: "{now():%Y-%m-%dT%H:%M:%SZ}"',
            "---",
            "",
            f"# Skill: {task.title}",
            "",
            f"**Task:** {task.description or task.title}",
            "",
            "**What worked:**",
        ]
        lines += [f"- {a}" for a in approach] or ["- (no rationale recorded)"]
        if files:
            lines += ["", "**Files changed:** " + ", ".join(f"`{f}`" for f in files)]
        if checks:
            lines += ["", "**Verification:**"] + [f"- {c}" for c in checks]
        return "\n".join(lines) + "\n"
