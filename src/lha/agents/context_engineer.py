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
from ..live_context.freshness import content_hash, file_sha256, path_from_locator

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
        dropped = 0
        root = workdir.resolve()
        kept = []
        for item in bundle.items:
            if item.provenance.source_kind != "code":
                kept.append(item)
                continue
            rel = path_from_locator(item.provenance.locator)
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                dropped += 1
                continue
            path = root / rel_path
            try:
                path.relative_to(root)
            except ValueError:
                dropped += 1
                continue
            source_sha256 = file_sha256(rel_path, root=root)
            if source_sha256 is None:
                dropped += 1
                continue
            try:
                current = path.read_text(errors="replace")
            except OSError:
                dropped += 1
                continue
            item.text = _slice_by_locator(current, item.provenance.locator)
            item.provenance.source_root = str(root)
            item.provenance.content_hash = content_hash(item.text)
            item.provenance.source_sha256 = source_sha256
            kept.append(item)
            overlaid += 1
        bundle.items = kept
        if overlaid:
            bundle.status_notes.append(f"{overlaid} code item(s) refreshed from the run sandbox")
        if dropped:
            bundle.status_notes.append(
                f"{dropped} unsafe or missing code item(s) dropped during sandbox refresh"
            )
        if not bundle.items and bundle.status == "ok":
            bundle.status = "empty"

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
