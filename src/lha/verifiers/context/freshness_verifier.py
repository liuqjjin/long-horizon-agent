"""Context verifier: the gathered context bundle must be fresh."""

from __future__ import annotations

from typing import Any

from ..base import Verifier, VerifyContext
from ..verdict import Check


class FreshnessVerifier(Verifier):
    name = "freshness"
    family = "context"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        bundle = ctx.bundle
        if bundle is None:
            return Check(name=self.name, family=self.family, passed=True, detail={"summary": "no bundle"})
        stale = bundle.freshness.is_stale()
        return Check(
            name=self.name,
            family=self.family,
            passed=not stale,
            detail={
                "summary": "fresh" if not stale else f"stale: {bundle.freshness.reasons}",
                "indexed_at": bundle.freshness.indexed_at.isoformat(),
            },
        )
