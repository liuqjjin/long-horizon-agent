"""Context verifier: claims/patches must cite resolvable provenance."""

from __future__ import annotations

from typing import Any

from ...artifacts import Patch
from ...live_context.models import ContextBundle
from ..base import Verifier, VerifyContext
from ..verdict import Check


class CitationVerifier(Verifier):
    name = "citation"
    family = "context"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        bundle = ctx.bundle
        known = set(bundle.locators()) if bundle else set()

        if isinstance(artifact, Patch):
            unresolved = [c for c in artifact.based_on_context if c not in known]
            return Check(
                name=self.name,
                family=self.family,
                passed=not unresolved,
                detail={
                    "summary": f"{len(artifact.based_on_context)} citations, {len(unresolved)} unresolved",
                    "unresolved": unresolved[:5],
                },
            )

        if isinstance(artifact, ContextBundle):
            have_prov = all(item.provenance.locator for item in artifact.items)
            return Check(
                name=self.name,
                family=self.family,
                passed=have_prov,
                detail={
                    "summary": f"{len(artifact.items)} items, all with provenance: {have_prov}"
                },
            )

        return Check(name=self.name, family=self.family, passed=True, detail={"summary": "n/a"})
