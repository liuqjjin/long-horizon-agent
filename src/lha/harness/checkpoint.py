"""Checkpoint I/O: atomic ``state.json`` + append-only ``ledger.jsonl``.

The ``FileCheckpointer`` wrapper mirrors the shape of a LangGraph
``BaseCheckpointSaver`` (get_tuple/put keyed by thread_id) so the migration to
``SqliteSaver`` later is mechanical.
"""

from __future__ import annotations

import os
from pathlib import Path

from .state import RunState, StepRecord

STATE_FILE = "state.json"
LEDGER_FILE = "ledger.jsonl"


def save_state(state: RunState) -> None:
    run_dir = Path(state.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / STATE_FILE
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    os.replace(tmp, target)  # atomic


def load_state(run_dir: str | Path) -> RunState:
    path = Path(run_dir) / STATE_FILE
    return RunState.model_validate_json(path.read_text())


def load_state_by_id(runs_dir: str | Path, run_id: str) -> RunState:
    return load_state(Path(runs_dir) / run_id)


def append_ledger(state: RunState, record: StepRecord) -> None:
    path = Path(state.run_dir) / LEDGER_FILE
    with open(path, "a") as f:
        f.write(record.model_dump_json() + "\n")


def read_ledger(run_dir: str | Path) -> list[StepRecord]:
    path = Path(run_dir) / LEDGER_FILE
    if not path.exists():
        return []
    return [StepRecord.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


class FileCheckpointer:
    """LangGraph-shaped facade over the file checkpoint (thread_id == run_id)."""

    def __init__(self, runs_dir: str | Path):
        self.runs_dir = Path(runs_dir)

    def get_tuple(self, thread_id: str) -> RunState | None:
        path = self.runs_dir / thread_id / STATE_FILE
        if not path.exists():
            return None
        return load_state(self.runs_dir / thread_id)

    def put(self, state: RunState) -> None:
        save_state(state)
