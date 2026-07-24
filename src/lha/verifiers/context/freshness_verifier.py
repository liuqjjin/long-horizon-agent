"""Context verifier: the gathered context bundle must be fresh and available.

Fail-closed semantics: when a step requires context, "no bundle", "backend
unavailable", "index failed" and "empty result" are all failures — a check
that could not actually verify anything must not pass. A step may declare
``context_requirement="optional"`` to proceed without retrieval; even then, a
bundle that exists but is stale or unavailable-with-errors still fails.
"""

from __future__ import annotations

from typing import Any

from ..base import Verifier, VerifyContext
from ..verdict import Check


class FreshnessVerifier(Verifier):
    name = "freshness"
    family = "context"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        required = ctx.step.context_requirement == "required"
        bundle = ctx.bundle

        if bundle is None:
            return self._check(
                passed=not required,
                summary=(
                    "no context bundle was gathered"
                    + ("" if required else " (declared optional)")
                ),
            )

        indexed_at = bundle.freshness.indexed_at.isoformat()
        if bundle.status in ("backend_unavailable", "index_failed"):
            return self._check(
                passed=not required,
                summary=f"context {bundle.status}: {'; '.join(bundle.status_notes) or 'no detail'}"
                + ("" if required else " (declared optional)"),
                indexed_at=indexed_at,
            )

        if bundle.status == "empty":
            return self._check(
                passed=not required,
                summary="no context found"
                + (" — step requires context" if required else " (declared optional)"),
                indexed_at=indexed_at,
            )

        # status "ok" with items can still hide a dark backend: another kind
        # answered while one the step asked for could not be searched at all.
        # Skill memory is an optional augmentation and does not count.
        dark = [k for k in bundle.unavailable_kinds if k != "skill"]
        if dark:
            return self._check(
                passed=not required,
                summary=f"backend unavailable for kind(s): {', '.join(dark)}"
                + ("" if required else " (declared optional)"),
                indexed_at=indexed_at,
            )

        stale = bundle.freshness.is_stale()
        return self._check(
            passed=not stale,
            summary="fresh" if not stale else f"stale: {bundle.freshness.reasons}",
            indexed_at=indexed_at,
        )

    def _check(self, *, passed: bool, summary: str, indexed_at: str | None = None) -> Check:
        detail: dict[str, Any] = {"summary": summary}
        if indexed_at is not None:
            detail["indexed_at"] = indexed_at
        return Check(name=self.name, family=self.family, passed=passed, detail=detail)
