"""Checkpoint I/O: atomic, checksummed ``state.json`` + append-only ``ledger.jsonl``.

Durability protocol:
  - ``state.json`` is a versioned envelope ``{schema_version, sha256, payload}``
    written to a temp file, fsynced, and atomically renamed (plus a directory
    fsync), so a crash leaves either the old or the new checkpoint — never a
    torn one. The checksum is verified on load; a mismatch fails closed with a
    diagnosable error instead of resuming from corrupt state.
  - Ledger records carry a unique ``event_id`` and are fsynced on append. A
    torn final line (the expected artifact of a crash mid-append) is dropped
    with a warning; a corrupt line anywhere else raises — that is damage, not
    a crash artifact.

The ``FileCheckpointer`` wrapper mirrors the shape of a LangGraph
``BaseCheckpointSaver`` (get_tuple/put keyed by thread_id).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from .errors import CheckpointCorrupt
from .state import RunState, StepRecord

logger = logging.getLogger(__name__)

STATE_FILE = "state.json"
LEDGER_FILE = "ledger.jsonl"
_ENVELOPE_VERSION = 2


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fsync_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # fsync the directory so the rename itself is durable
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:  # pragma: no cover - platform without dir fsync
        pass


def save_state(state: RunState) -> None:
    run_dir = Path(state.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = state.model_dump(mode="json")
    envelope = {
        "schema_version": _ENVELOPE_VERSION,
        "sha256": hashlib.sha256(_canonical(payload).encode()).hexdigest(),
        "payload": payload,
    }
    _fsync_write(run_dir / STATE_FILE, json.dumps(envelope, indent=2))


def load_state(run_dir: str | Path) -> RunState:
    path = Path(run_dir) / STATE_FILE
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CheckpointCorrupt(f"unreadable checkpoint {path}: {e}") from e

    if isinstance(raw, dict) and "payload" in raw and "sha256" in raw:
        payload = raw["payload"]
        digest = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        if digest != raw["sha256"]:
            raise CheckpointCorrupt(
                f"checkpoint {path} failed its integrity check "
                f"(stored {raw['sha256'][:12]}…, computed {digest[:12]}…) — refusing to resume"
            )
    else:
        # pre-envelope checkpoint (schema 1): accepted for old runs, no checksum
        payload = raw

    try:
        state = RunState.model_validate(payload)
    except Exception as e:
        raise CheckpointCorrupt(f"checkpoint {path} does not validate: {e}") from e
    if state.plan is not None and not (0 <= state.cursor <= len(state.plan.steps)):
        raise CheckpointCorrupt(
            f"checkpoint {path} is inconsistent: cursor {state.cursor} outside plan "
            f"of {len(state.plan.steps)} steps"
        )
    return state


def load_state_by_id(runs_dir: str | Path, run_id: str) -> RunState:
    return load_state(Path(runs_dir) / run_id)


def append_ledger(state: RunState, record: StepRecord) -> None:
    path = Path(state.run_dir) / LEDGER_FILE
    # A crash can leave a torn final line. Appending after it would merge two
    # records into one corrupt mid-file line (which read_ledger rightly refuses),
    # so drop the fragment first — it was never durable, and read_ledger already
    # treats it as lost.
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            with open(path, "rb+") as f:
                f.truncate(raw.rfind(b"\n") + 1)
    with open(path, "a") as f:
        f.write(record.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_ledger(run_dir: str | Path) -> list[StepRecord]:
    path = Path(run_dir) / LEDGER_FILE
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records: list[StepRecord] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(StepRecord.model_validate_json(line))
        except Exception as e:
            if i == len(lines) - 1:
                # a torn final line is the expected residue of a crash mid-append
                logger.warning("dropping torn final ledger line in %s", path)
                continue
            raise CheckpointCorrupt(
                f"ledger {path} line {i + 1} is corrupt (not a torn tail): {e}"
            ) from e
    return records


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
