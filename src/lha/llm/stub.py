"""Deterministic, network-free LLM stub for the walking skeleton.

It produces a scripted Patch for the toy issue->PR task (the average() off-by-one
bug), so the loop fixes a REAL bug that a REAL pytest then verifies — with no LLM,
no API key, and fully reproducible CI.
"""

from __future__ import annotations

from pathlib import Path

from ..artifacts import Patch
from ..live_context.models import ContextBundle
from ..tools.patch import make_unified_diff
from .base import LLMClient

# The scripted fix: drop the stray "- 1" in average().
_BUGGY = "sum(values) / len(values) - 1"
_FIXED = "sum(values) / len(values)"


class DeterministicStub(LLMClient):
    name = "stub"

    def complete(self, system: str, prompt: str) -> str:  # not used by the stub
        return ""

    def propose_patch(self, step, bundle: ContextBundle, workdir: str | Path) -> Patch:
        workdir = Path(workdir)
        target_rel = "mathutils.py"
        target = workdir / target_rel
        if not target.exists():
            return Patch(step_id=step.step_id, rationale="stub: target not found")

        original = target.read_text()
        fixed = original.replace(_BUGGY, _FIXED)
        if fixed == original:
            return Patch(
                step_id=step.step_id,
                rationale="stub: nothing to fix (pattern not present)",
                based_on_context=bundle.locators(),
            )

        return Patch(
            step_id=step.step_id,
            unified_diff=make_unified_diff(original, fixed, target_rel),
            touched_files=[target_rel],
            file_contents={target_rel: fixed},
            rationale="Remove the stray '- 1' that made average() off-by-one.",
            based_on_context=bundle.locators(),
        )
