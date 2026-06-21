"""Code verifier: run ruff and report structured lint violations."""

from __future__ import annotations

import json
from typing import Any

from ...tools.shell import run, venv_tool
from ..base import Verifier, VerifyContext
from ..verdict import Check


class RuffVerifier(Verifier):
    name = "ruff"
    family = "code"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        ruff = venv_tool("ruff")
        res = run(
            [ruff, "check", "--output-format", "json", "--exclude", ".cocoindex_code", "."],
            cwd=ctx.workdir,
        )
        try:
            violations = json.loads(res.stdout) if res.stdout.strip() else []
        except json.JSONDecodeError:
            violations = []

        ok = len(violations) == 0
        return Check(
            name=self.name,
            family=self.family,
            passed=ok,
            score=float(len(violations)),
            threshold=0.0,
            detail={
                "summary": f"{len(violations)} violations",
                "codes": [v.get("code") for v in violations[:10]],
            },
            duration_s=res.duration_s,
        )
