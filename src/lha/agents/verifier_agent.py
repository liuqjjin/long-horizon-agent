"""Verifier agent: run the step's selected verifiers and aggregate a Verdict.

Verifiers are independent, so they run concurrently (the slow ones — pytest, the
reproducibility re-run — are subprocess waits). Output order is preserved and a
crashing verifier becomes a failing Check, never an aborted verdict.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..artifacts import Step
from ..sandbox import ExecutionBackend
from ..tools.shell import run
from ..verifiers import VerifyContext, select_verifiers
from ..verifiers.base import Verifier
from ..verifiers.verdict import Check, Verdict


def _target_git_commit(workdir: str | Path) -> str | None:
    """Return HEAD only when the verified directory is itself a Git worktree."""
    root = Path(workdir).resolve()
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        # A run worktree may live below the harness checkout. Do not let Git walk
        # upward and accidentally attribute the harness commit to the target.
        "GIT_CEILING_DIRECTORIES": str(root.parent),
    }
    top = run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        env=environment,
    )
    if top.returncode != 0:
        return None
    try:
        discovered = Path(top.stdout.strip()).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if discovered != root:
        return None
    revision = run(["git", "rev-parse", "HEAD"], cwd=root, env=environment)
    value = revision.stdout.strip()
    return value if revision.returncode == 0 and len(value) == 40 else None


def _env_record(
    workdir: str | Path,
    execution_backend: ExecutionBackend,
) -> dict[str, Any]:
    pkgs: dict[str, str] = {}
    for name in ("pytest", "ruff", "pydantic", "scikit-image", "numpy", "cocoindex"):
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    harness_rev = run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
    backend_record: dict[str, Any] = {
        "name": execution_backend.name,
        "image": getattr(execution_backend, "image", None),
    }
    provenance_probe = getattr(execution_backend, "provenance", None)
    if callable(provenance_probe):
        try:
            provenance = provenance_probe()
        except Exception:
            # Provenance is diagnostic evidence. A broken Docker daemon must be
            # visible without turning an otherwise valid verifier result into a
            # lost verdict or persisting the daemon's potentially sensitive error.
            provenance = {
                "status": "unavailable",
                "reason": "probe_failed",
            }
        if not isinstance(provenance, dict):
            provenance = {
                "status": "unavailable",
                "reason": "invalid_probe_result",
            }
        backend_record["provenance"] = provenance

    return {
        # These values describe the harness process. A Docker verifier may use a
        # different interpreter and package set, so keep the boundary explicit.
        "control_plane": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": pkgs,
        },
        "execution_backend": backend_record,
        "target_git_commit": _target_git_commit(workdir),
        "harness_git_commit": harness_rev.stdout.strip() if harness_rev.returncode == 0 else None,
    }


def _safe_verify(verifier: Verifier, artifact: Any, ctx: VerifyContext) -> Check:
    try:
        source = Path(ctx.workdir).resolve()
        with tempfile.TemporaryDirectory(prefix="lha-verify-") as temporary:
            isolated = Path(temporary) / "workdir"
            shutil.copytree(
                source,
                isolated,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    ".pytest_cache",
                    ".lha_pytest.json",
                ),
            )
            return verifier.verify(
                artifact,
                replace(ctx, workdir=isolated),
            )
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
        # A step that produced zero checks verified nothing. Verdict.from_checks
        # already fails an empty list closed; this names the reason in the verdict
        # instead of leaving the repair loop with nothing to act on.
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
            attempt_id=ctx.attempt_id,
            env=_env_record(ctx.workdir, ctx.exec),
        )
