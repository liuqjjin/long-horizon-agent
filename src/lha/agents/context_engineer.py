"""Context Engineer: gather provenance-carrying, fresh context via the facade.

Calls ONLY ``live_context`` — never CocoIndex/ccc directly.
"""

from __future__ import annotations

from ..artifacts import Step
from ..config import Config
from ..live_context import (
    ContextBundle,
    StaleContextError,
    get_fresh_context,
    reject_stale,
)

_KINDS_BY_STEP = {
    "code": ("code",),
    "experiment": ("experiment", "paper"),
    "context": ("code", "paper", "experiment"),
}


class ContextEngineer:
    def __init__(self, config: Config):
        self.config = config

    def gather(self, step: Step) -> ContextBundle:
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
            except StaleContextError:
                pass
        if step.action == "answer_query":
            bundle.answer = self._synthesize(bundle)
        return bundle

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
