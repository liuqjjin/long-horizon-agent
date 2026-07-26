"""Runtime configuration, populated from environment variables (``LHA_*``).

A ``.env`` file in the project root is loaded automatically if python-dotenv is
available. Everything here has a sensible default so the walking skeleton runs
with zero configuration.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat

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
    return value or None


def _env_int_opt(key: str) -> int | None:
    """Optional positive-int env var: ``None`` when unset/blank/0 (= unlimited).

    Rejects negatives so a typo bricks the run loudly at startup, not by raising
    BudgetExceeded before the first LLM call.
    """
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from e
    if value < 0:
        raise ValueError(f"{key} must be >= 0 (0 or unset = unlimited), got {raw!r}")
    return value or None


def _env_codex_sandbox() -> Literal["read-only", "workspace-write", "danger-full-access"]:
    value = _env("LHA_CODEX_SANDBOX", "read-only")
    if value == "read-only":
        return "read-only"
    if value == "workspace-write":
        return "workspace-write"
    if value == "danger-full-access":
        return "danger-full-access"
    raise ValueError(f"LHA_CODEX_SANDBOX has unsupported value {value!r}")


class Config(BaseModel):
    """Harness configuration. Construct with ``Config.from_env()``."""

    # Loop budget
    max_steps: int = Field(default=20, ge=1)
    max_repairs: int = Field(default=3, ge=0)
    deadline_s: FiniteFloat | None = Field(default=None, ge=0)
    # Model-call budget for the whole durable run (None = unbounded).
    max_llm_calls: int | None = Field(default=None, ge=1)

    # Run the selected verifiers concurrently
    parallel_verify: bool = True

    # Record verified successes as retrievable skills
    use_skill_memory: bool = True

    # Let the LLM backend (re)plan the task instead of the deterministic template.
    # Off by default so the stub/eval path stays deterministic; real backends only.
    dynamic_planning: bool = False

    # Freshness
    freshness_max_age_s: FiniteFloat = Field(default=3600.0, ge=0)

    # LLM backend: "stub" | "claude_cli" | "codex_cli" | "anthropic"
    llm_backend: str = "stub"
    claude_cli_path: str = "claude"
    # Pin a full model snapshot for reproducible runs; "" lets the CLI decide.
    claude_cli_model: str = ""
    codex_cli_path: str = "codex"
    codex_model: str = ""
    codex_reasoning_effort: str = "medium"
    codex_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    # danger-full-access is only valid when the whole process already runs in a
    # disposable external sandbox (for example a Harbor task container).
    codex_external_sandbox: bool = False
    # Only failures classified as transport/service-transient are retried.
    # Protocol violations and unsafe event streams fail on the first attempt.
    codex_max_retries: int = Field(default=2, ge=0, le=10)
    codex_retry_backoff_s: FiniteFloat = Field(default=1.0, ge=0)
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
            max_llm_calls=_env_int_opt("LHA_MAX_LLM_CALLS"),
            parallel_verify=_env("LHA_PARALLEL_VERIFY", "1") not in ("0", "false", "False"),
            use_skill_memory=_env("LHA_SKILL_MEMORY", "1") not in ("0", "false", "False"),
            dynamic_planning=_env("LHA_DYNAMIC_PLANNING", "0") not in ("0", "false", "False"),
            freshness_max_age_s=float(_env("LHA_FRESHNESS_MAX_AGE_S", "3600")),
            llm_backend=_env("LHA_LLM_BACKEND", "stub"),
            claude_cli_path=_env("LHA_CLAUDE_CLI", "claude"),
            claude_cli_model=_env("LHA_CLAUDE_MODEL", ""),
            codex_cli_path=_env("LHA_CODEX_CLI", "codex"),
            codex_model=_env("LHA_CODEX_MODEL", ""),
            codex_reasoning_effort=_env("LHA_CODEX_EFFORT", "medium"),
            codex_sandbox=_env_codex_sandbox(),
            codex_external_sandbox=_env("LHA_CODEX_EXTERNAL_SANDBOX", "0")
            not in ("0", "false", "False"),
            codex_max_retries=int(_env("LHA_CODEX_MAX_RETRIES", "2")),
            codex_retry_backoff_s=float(_env("LHA_CODEX_RETRY_BACKOFF_S", "1")),
            anthropic_model_impl=_env("LHA_ANTHROPIC_MODEL_IMPL", "claude-opus-4-8"),
            anthropic_model_orchestration=_env("LHA_ANTHROPIC_MODEL_ORCH", "claude-sonnet-4-6"),
            code_backend=_env("LHA_CODE_BACKEND", "auto"),
            embedder_model=_env("LHA_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            exec_backend=_env("LHA_EXEC_BACKEND", "trusted-local"),
            exec_image=_env("LHA_EXEC_IMAGE", "python:3.12-slim"),
            runs_dir=Path(_env("LHA_RUNS_DIR", "runs")),
            data_dir=Path(_env("LHA_DATA_DIR", "data")),
        )
