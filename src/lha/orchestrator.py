"""Run independent tasks concurrently in process-isolated workers.

The live-context facade is a process-global singleton, so cross-task parallelism
uses worker subprocesses (`lha run --json`) for clean isolation. This is the
only level that needs process isolation; verifiers within one run already use
threads where useful.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .sandbox.base import (
    DEFAULT_OUTPUT_BYTES,
    process_group_cleanup_supported,
    run_bounded_process,
    scrub_env,
    terminate_process_group,
)
from .tools.shell import ProcResult

_RESULT_PREFIX = "__LHA_RESULT__ "
_WORKER_OUTPUT_BYTES = DEFAULT_OUTPUT_BYTES

# Batch workers need the same explicit configuration as the parent CLI, not
# unrelated shell, cloud, GitHub, or SSH credentials.
_WORKER_CONFIG_ENV = (
    "LHA_MAX_STEPS",
    "LHA_MAX_REPAIRS",
    "LHA_DEADLINE_S",
    "LHA_MAX_LLM_CALLS",
    "LHA_PARALLEL_VERIFY",
    "LHA_SKILL_MEMORY",
    "LHA_DYNAMIC_PLANNING",
    "LHA_FRESHNESS_MAX_AGE_S",
    "LHA_LLM_BACKEND",
    "LHA_CLAUDE_CLI",
    "LHA_CLAUDE_MODEL",
    "LHA_CODEX_CLI",
    "LHA_CODEX_MODEL",
    "LHA_CODEX_EFFORT",
    "LHA_CODEX_SANDBOX",
    "LHA_CODEX_EXTERNAL_SANDBOX",
    "LHA_CODEX_MAX_RETRIES",
    "LHA_CODEX_RETRY_BACKOFF_S",
    "LHA_ANTHROPIC_MODEL_IMPL",
    "LHA_ANTHROPIC_MODEL_ORCH",
    "LHA_CODE_BACKEND",
    "LHA_EMBEDDER_MODEL",
    "LHA_EXEC_BACKEND",
    "LHA_EXEC_IMAGE",
    "LHA_RUNS_DIR",
    "LHA_DATA_DIR",
)
_BACKEND_AUTH_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "claude_cli": ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"),
    "codex_cli": ("CODEX_HOME",),
}


@dataclass
class TaskOutcome:
    task: str
    status: str
    verified: bool | None = None
    run_id: str | None = None
    detail: str = ""


def _worker_env(llm: str | None = None) -> dict[str, str]:
    selected_backend = llm or os.environ.get("LHA_LLM_BACKEND", "stub")
    allowed = {
        name: os.environ[name]
        for name in (
            *_WORKER_CONFIG_ENV,
            *_BACKEND_AUTH_ENV.get(selected_backend, ()),
        )
        if name in os.environ
    }
    env = scrub_env(allowed)
    extra = (str(Path.home() / ".local" / "bin"), "/opt/homebrew/bin")
    env["PATH"] = os.pathsep.join((*extra, env.get("PATH", os.defpath)))
    return env


# Statuses that the worker maps to a clean (exit 0) process.
_CLEAN_EXIT_STATUSES = {"DONE", "AWAITING_APPROVAL", "PAUSED"}


def _parse(task: str, proc: ProcResult) -> TaskOutcome:
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                d = json.loads(line[len(_RESULT_PREFIX) :])
            except json.JSONDecodeError:
                break
            status = d.get("status", "ERROR")
            # A clean status with a nonzero exit means the worker crashed after
            # emitting its result — don't report it as success.
            if proc.returncode != 0 and status in _CLEAN_EXIT_STATUSES:
                cleanup_detail = (proc.stderr or "").strip()[-300:]
                suffix = f": {cleanup_detail}" if cleanup_detail else ""
                return TaskOutcome(
                    task=task,
                    status="ERROR",
                    run_id=d.get("run_id"),
                    detail=(
                        f"worker exited {proc.returncode} despite status={status}"
                        f"{suffix}"
                    ),
                )
            return TaskOutcome(
                task=task,
                status=status,
                verified=d.get("verified"),
                run_id=d.get("run_id"),
            )
    tail = (proc.stderr or proc.stdout or "")[-300:]
    return TaskOutcome(task=task, status="ERROR", detail=tail)


def run_tasks(
    task_paths: list[str],
    *,
    llm: str | None = None,
    max_workers: int = 4,
    timeout: float = 1800.0,
) -> list[TaskOutcome]:
    def worker(task_path: str) -> TaskOutcome:
        if not process_group_cleanup_supported():
            return TaskOutcome(
                task=str(task_path),
                status="ERROR",
                detail=(
                    "batch execution requires POSIX process-group cleanup; "
                    "use Linux, macOS, or WSL2"
                ),
            )
        cmd = [sys.executable, "-m", "lha.cli"]
        if llm:
            cmd += ["--llm", llm]
        cmd += ["run", "--json", str(task_path)]
        try:
            proc = run_bounded_process(
                cmd,
                timeout=timeout,
                output_bytes=_WORKER_OUTPUT_BYTES,
                env=_worker_env(llm),
                start_new_session=True,
                on_exit=terminate_process_group,
            )
        except OSError as e:
            # A spawn failure (ENOMEM, EMFILE, bad interpreter, ...) for one task must
            # not propagate through ex.map and discard every sibling result.
            return TaskOutcome(task=str(task_path), status="ERROR", detail=f"spawn failed: {e}")
        if proc.returncode == 124:
            detail = (proc.stderr or proc.stdout or "")[-300:]
            return TaskOutcome(task=str(task_path), status="TIMEOUT", detail=detail)
        return _parse(str(task_path), proc)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(worker, [str(t) for t in task_paths]))
