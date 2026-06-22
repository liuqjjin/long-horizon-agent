"""Verifier agent: run the step's selected verifiers and aggregate a Verdict.

Verifiers are independent, so they run concurrently (the slow ones — pytest, the
reproducibility re-run — are subprocess waits). Output order is preserved and a
crashing verifier becomes a failing Check, never an aborted verdict.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..artifacts import Step
from ..tools.shell import run
from ..verifiers import VerifyContext, select_verifiers
from ..verifiers.base import Verifier
from ..verifiers.verdict import Check, Verdict


def _env_record(workdir: str | Path) -> dict[str, Any]:
    pkgs: dict[str, str] = {}
    for name in ("pytest", "ruff", "pydantic", "scikit-image", "numpy", "cocoindex"):
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    rev = run(["git", "rev-parse", "HEAD"])
    commit = rev.stdout.strip() if rev.returncode == 0 else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "git_commit": commit,
    }


def _safe_verify(verifier: Verifier, artifact: Any, ctx: VerifyContext) -> Check:
    try:
        return verifier.verify(artifact, ctx)
    except Exception as e:  # a crashing verifier is a failure, not a lost verdict
        return Check(
            name=verifier.name,
            family=getattr(verifier, "family", "code"),
            passed=False,
            detail={"summary": f"verifier crashed: {type(e).__name__}: {e}"},
        )


class VerifierAgent:
    def __init__(self, parallel: bool = True):
        self.parallel = parallel

    def verify(self, step: Step, artifact: Any, ctx: VerifyContext) -> Verdict:
        verifiers = select_verifiers(step)
        if self.parallel and len(verifiers) > 1:
            with ThreadPoolExecutor(max_workers=min(len(verifiers), 4)) as ex:
                checks = list(ex.map(lambda v: _safe_verify(v, artifact, ctx), verifiers))
        else:
            checks = [_safe_verify(v, artifact, ctx) for v in verifiers]

        # A requested verifier that isn't registered is a failure, not a silent
        # pass — never let "couldn't verify" read as "verified".
        found = {v.name for v in verifiers}
        for name in step.verifiers:
            if name not in found:
                checks.append(
                    Check(
                        name=name,
                        family="context",
                        passed=False,
                        detail={"summary": f"verifier '{name}' is not registered"},
                    )
                )
        # Defense in depth: a step that produced zero checks verified nothing, so it
        # must not pass vacuously (Verdict.from_checks treats an empty list as pass).
        if not checks:
            checks.append(
                Check(
                    name="no-verifier",
                    family="context",
                    passed=False,
                    detail={"summary": "step declared no verifiers — nothing was verified"},
                )
            )
        return Verdict.from_checks(
            step.step_id,
            checks,
            artifact_ref=step.step_id,
            env=_env_record(ctx.workdir),
        )
