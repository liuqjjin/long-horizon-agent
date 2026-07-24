"""Runtime configuration, populated from environment variables (``LHA_*``).

A ``.env`` file in the project root is loaded automatically if python-dotenv is
available. Everything here has a sensible default so the walking skeleton runs
with zero configuration.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from pydantic import BaseModel

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float_opt(key: str) -> float | None:
    """Optional positive-float env var: ``None`` when unset/blank, parsed otherwise.

    Rejects non-finite (NaN/inf) and negative values so a misconfigured deadline
    can't silently disable the budget check.
    """
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as e:
        raise ValueError(f"{key} must be a number, got {raw!r}") from e
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be a finite, non-negative number, got {raw!r}")
    return value


class Config(BaseModel):
    """Harness configuration. Construct with ``Config.from_env()``."""

    # Loop budget
    max_steps: int = 20
    max_repairs: int = 3
    deadline_s: float | None = None

    # Run the selected verifiers concurrently
    parallel_verify: bool = True

    # Record verified successes as retrievable skills
    use_skill_memory: bool = True

    # Let the LLM backend (re)plan the task instead of the deterministic template.
    # Off by default so the stub/eval path stays deterministic; real backends only.
    dynamic_planning: bool = False

    # Freshness
    freshness_max_age_s: float = 3600.0

    # LLM backend: "stub" | "claude_cli" | "anthropic"
    llm_backend: str = "stub"
    claude_cli_path: str = "claude"
    anthropic_model_impl: str = "claude-opus-4-8"
    anthropic_model_orchestration: str = "claude-sonnet-4-6"

    # Code search backend: "ccc" | "null"  ("auto" picks ccc if available)
    code_backend: str = "auto"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Where target/model-influenced code executes: "trusted-local" (this repo's
    # own dev/self-eval only) or "docker" (external target repos).
    exec_backend: str = "trusted-local"
    exec_image: str = "python:3.12-slim"

    # Paths
    runs_dir: Path = Path("runs")
    data_dir: Path = Path("data")

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            max_steps=int(_env("LHA_MAX_STEPS", "20")),
            max_repairs=int(_env("LHA_MAX_REPAIRS", "3")),
            deadline_s=_env_float_opt("LHA_DEADLINE_S"),
            parallel_verify=_env("LHA_PARALLEL_VERIFY", "1") not in ("0", "false", "False"),
            use_skill_memory=_env("LHA_SKILL_MEMORY", "1") not in ("0", "false", "False"),
            dynamic_planning=_env("LHA_DYNAMIC_PLANNING", "0") not in ("0", "false", "False"),
            freshness_max_age_s=float(_env("LHA_FRESHNESS_MAX_AGE_S", "3600")),
            llm_backend=_env("LHA_LLM_BACKEND", "stub"),
            claude_cli_path=_env("LHA_CLAUDE_CLI", "claude"),
            anthropic_model_impl=_env("LHA_ANTHROPIC_MODEL_IMPL", "claude-opus-4-8"),
            anthropic_model_orchestration=_env("LHA_ANTHROPIC_MODEL_ORCH", "claude-sonnet-4-6"),
            code_backend=_env("LHA_CODE_BACKEND", "auto"),
            embedder_model=_env("LHA_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            exec_backend=_env("LHA_EXEC_BACKEND", "trusted-local"),
            exec_image=_env("LHA_EXEC_IMAGE", "python:3.12-slim"),
            runs_dir=Path(_env("LHA_RUNS_DIR", "runs")),
            data_dir=Path(_env("LHA_DATA_DIR", "data")),
        )
