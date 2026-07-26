"""Code verifier: run ruff and report structured lint violations."""

from __future__ import annotations

import json
from typing import Any

from ..base import Verifier, VerifyContext
from ..verdict import Check


class RuffVerifier(Verifier):
    name = "ruff"
    family = "code"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        ruff = ctx.exec.tool("ruff")
        res = ctx.exec.run(
            [
                ruff,
                "check",
                "--no-cache",
                "--output-format",
                "json",
                "--exclude",
                ".cocoindex_code",
                ".",
            ],
            cwd=ctx.workdir,
        )
        # ruff exits 0 (clean) or 1 (violations found, JSON on stdout). Anything
        # else — 2 (internal/config error), 124 (timeout), 127 (not found) — means
        # ruff could not run a real lint pass. That must FAIL, never silently pass:
        # "couldn't verify" must not read as "verified".
        ran = res.returncode in (0, 1)
        violations: list = []
        if ran and res.stdout.strip():
            try:
                violations = json.loads(res.stdout)
            except json.JSONDecodeError:
                ran = False  # non-JSON on the success channel -> ruff did not run cleanly
        # rc=1 means "violations found"; an empty/unparsed result there is inconsistent,
        # so treat it as a failed run rather than a clean pass.
        if res.returncode == 1 and not violations:
            ran = False

        ok = ran and len(violations) == 0
        detail: dict[str, Any] = {
            "summary": (
                f"{len(violations)} violations"
                if ran
                else f"ruff failed to run (rc={res.returncode})"
            ),
            "codes": [v.get("code") for v in violations[:10]],
            "returncode": res.returncode,
        }
        if not ran:
            detail["stderr"] = (res.stderr or "")[-500:]
        return Check(
            name=self.name,
            family=self.family,
            passed=ok,
            score=float(len(violations)),
            threshold=0.0,
            detail=detail,
            duration_s=res.duration_s,
        )
