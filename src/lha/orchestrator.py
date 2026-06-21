"""Orchestrator: run many tasks in parallel via process-isolated workers.

The live-context facade is a process-global singleton, so cross-task parallelism
uses worker subprocesses (`lha run --json`) for clean isolation. This is the
orchestrator-worker pattern, applied only where parallelism actually helps:
independent tasks. (Within a single run, verifiers already run concurrently.)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

_RESULT_PREFIX = "__LHA_RESULT__ "


@dataclass
class TaskOutcome:
    task: str
    status: str
    verified: bool | None = None
    run_id: str | None = None
    detail: str = ""


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    extra = f"{Path.home()}/.local/bin:/opt/homebrew/bin"
    env["PATH"] = extra + ":" + env.get("PATH", "")
    return env


# Statuses that the worker maps to a clean (exit 0) process.
_CLEAN_EXIT_STATUSES = {"DONE", "AWAITING_APPROVAL", "PAUSED"}


def _parse(task: str, proc: subprocess.CompletedProcess) -> TaskOutcome:
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                d = json.loads(line[len(_RESULT_PREFIX):])
            except json.JSONDecodeError:
                break
            status = d.get("status", "ERROR")
            # A clean status with a nonzero exit means the worker crashed after
            # emitting its result — don't report it as success.
            if proc.returncode != 0 and status in _CLEAN_EXIT_STATUSES:
                return TaskOutcome(
                    task=task,
                    status="ERROR",
                    run_id=d.get("run_id"),
                    detail=f"worker exited {proc.returncode} despite status={status}",
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
        cmd = [sys.executable, "-m", "lha.cli"]
        if llm:
            cmd += ["--llm", llm]
        cmd += ["run", "--json", str(task_path)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=_worker_env()
            )
        except subprocess.TimeoutExpired:
            return TaskOutcome(task=str(task_path), status="TIMEOUT")
        return _parse(str(task_path), proc)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(worker, [str(t) for t in task_paths]))
