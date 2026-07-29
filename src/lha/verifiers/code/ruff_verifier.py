"""Code verifier: run ruff and report structured lint violations."""

from __future__ import annotations

import json
from typing import Any

from ..base import Verifier, VerifyContext
from ..verdict import Check, process_cleanup_failure_detail


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
        violations: list[dict[str, Any]] = []
        parse_error: str | None = None
        if ran and res.stdout.strip():
            try:
                payload = json.loads(res.stdout)
            except json.JSONDecodeError as error:
                parse_error = f"invalid JSON: {error.msg}"
                ran = False
            else:
                if not isinstance(payload, list):
                    parse_error = "top-level Ruff output is not a list"
                    ran = False
                elif not all(
                    isinstance(item, dict)
                    and isinstance(item.get("code"), str)
                    and bool(item["code"])
                    and isinstance(item.get("message"), str)
                    and isinstance(item.get("filename"), str)
                    and isinstance(item.get("location"), dict)
                    and isinstance(item["location"].get("row"), int)
                    and isinstance(item["location"].get("column"), int)
                    for item in payload
                ):
                    parse_error = "Ruff violation output has an invalid schema"
                    ran = False
                else:
                    violations = payload
        elif ran:
            parse_error = "Ruff returned no JSON result"
            ran = False

        # Exit 0 requires the canonical empty list. Exit 1 requires at least one
        # schema-valid violation. An object such as ``{}`` has length zero too,
        # but is not evidence that Ruff completed a lint pass.
        if (res.returncode == 0 and violations) or (
            res.returncode == 1 and not violations
        ):
            parse_error = "Ruff result contradicts its exit code"
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
            if parse_error is not None:
                detail["parse_error"] = parse_error
        detail.update(
            process_cleanup_failure_detail(
                returncode=res.returncode,
                cleanup_unconfirmed=res.cleanup_unconfirmed,
                detail=(res.cleanup_detail or res.stderr or "")[-500:],
            )
        )
        return Check(
            name=self.name,
            family=self.family,
            passed=ok,
            score=float(len(violations)),
            threshold=0.0,
            detail=detail,
            duration_s=res.duration_s,
        )
