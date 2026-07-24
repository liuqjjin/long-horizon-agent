"""Context Engineer: gather provenance-carrying, fresh context via the facade.

Calls ONLY ``live_context`` — never CocoIndex/ccc directly.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..artifacts import Step
from ..config import Config
from ..live_context import (
    ContextBundle,
    StaleContextError,
    get_fresh_context,
    reject_stale,
)
from ..live_context.freshness import path_from_locator

_KINDS_BY_STEP = {
    "code": ("code",),
    "experiment": ("experiment", "paper"),
    "context": ("code", "paper", "experiment"),
}


class ContextEngineer:
    def __init__(self, config: Config):
        self.config = config

    def gather(self, step: Step, workdir: str | Path | None = None) -> ContextBundle:
        kinds = list(_KINDS_BY_STEP.get(step.kind, ("code", "paper", "experiment")))
        # Episodic memory: retrieve relevant past skills (cheap no-op until indexed).
        if self.config.use_skill_memory and "skill" not in kinds:
            kinds.append("skill")
        kinds = tuple(kinds)
        query = step.context_query or step.goal
        bundle = get_fresh_context(
            query, kinds=kinds, k=8, max_age_s=self.config.freshness_max_age_s
        )
        if bundle.freshness.is_stale():
            try:
                bundle = reject_stale(bundle)
            except StaleContextError as e:
                # Refresh failed: the bundle stays stale AND is marked so the
                # freshness verifier fails it closed with a diagnosable reason.
                bundle.status = "index_failed"
                bundle.status_notes.append(str(e))
        if workdir is not None and step.repair_of:
            # A repair must reason over the CURRENT sandbox, not the pristine
            # repo the index was built from — the failing state is the point.
            self._overlay_workdir(bundle, Path(workdir))
        if step.action == "answer_query":
            bundle.answer = self._synthesize(bundle)
        return bundle

    @staticmethod
    def _overlay_workdir(bundle: ContextBundle, workdir: Path) -> None:
        overlaid = 0
        for item in bundle.items:
            if item.provenance.source_kind != "code":
                continue
            rel = path_from_locator(item.provenance.locator)
            path = workdir / rel
            if not path.is_file():
                continue
            try:
                current = path.read_text(errors="replace")
            except OSError:
                continue
            if item.text and item.text in current:
                continue  # chunk unchanged in the sandbox
            item.text = _slice_by_locator(current, item.provenance.locator)
            overlaid += 1
        if overlaid:
            bundle.status_notes.append(f"{overlaid} code item(s) refreshed from the run sandbox")

    @staticmethod
    def _synthesize(bundle: ContextBundle) -> str:
        """A simple, deterministic, citation-anchored answer for v1."""
        if not bundle.items:
            return f"No indexed context found for: {bundle.query}"
        lines = [f"Answer for: {bundle.query}", ""]
        for item in bundle.items[:5]:
            snippet = " ".join(item.text.split())[:200]
            lines.append(f"- {snippet} [{item.provenance.locator}]")
        return "\n".join(lines)


def _slice_by_locator(text: str, locator: str) -> str:
    """The locator's line range from the current file (whole file if no range)."""
    m = re.search(r":(\d+)(?:-(\d+))?$", locator)
    if not m:
        return text
    start = int(m.group(1))
    end = int(m.group(2) or m.group(1))
    lines = text.splitlines()
    return "\n".join(lines[max(start - 1, 0) : end])
