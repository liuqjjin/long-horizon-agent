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
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import CheckpointCorrupt, RunLocked
from .state import RunState, StepRecord

logger = logging.getLogger(__name__)

STATE_FILE = "state.json"
LEDGER_FILE = "ledger.jsonl"
_ENVELOPE_VERSION = 2
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


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

    legacy_checkpoint = not (
        isinstance(raw, dict) and "payload" in raw and "sha256" in raw
    )
    if not legacy_checkpoint:
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

    if not isinstance(payload, dict):
        raise CheckpointCorrupt(f"checkpoint {path} payload must be an object")
    # RunState did not carry its own schema marker before v2. Pydantic defaults
    # are for newly constructed states, so allowing that default while loading
    # would silently upgrade an old, transaction-less run and make it resumable.
    payload = dict(payload)
    if legacy_checkpoint or "schema_version" not in payload:
        payload["schema_version"] = 1

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
    validate_run_id(run_id)
    root = Path(runs_dir).resolve()
    expected = root / run_id
    if expected.is_symlink():
        raise CheckpointCorrupt(f"run directory is a symlink: {expected}")
    state = load_state(expected)
    if state.run_id != run_id:
        raise CheckpointCorrupt(
            f"checkpoint run_id {state.run_id!r} does not match directory {run_id!r}"
        )
    recorded_run_dir = Path(state.run_dir).resolve()
    recorded_workdir = Path(state.workdir).resolve()
    if recorded_run_dir != expected:
        raise CheckpointCorrupt(
            f"checkpoint run_dir {state.run_dir!r} does not match {expected}"
        )
    expected_workdir = expected / "workdir"
    if recorded_workdir != expected_workdir:
        raise CheckpointCorrupt(
            f"checkpoint workdir {state.workdir!r} does not match {expected_workdir}"
        )
    return state


def validate_run_id(run_id: str) -> None:
    if _RUN_ID.fullmatch(run_id) is None or run_id in (".", ".."):
        raise ValueError(f"invalid run id: {run_id!r}")


def _ledger_sha256(record: StepRecord) -> str:
    payload = record.model_dump(mode="json")
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def append_ledger(state: RunState, record: StepRecord) -> None:
    path = Path(state.run_dir) / LEDGER_FILE
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CheckpointCorrupt(f"ledger path is unsafe: {path}")
    # A crash can leave a torn final line. Appending after it would merge two
    # records into one corrupt mid-file line (which read_ledger rightly refuses),
    # so drop the fragment first — it was never durable, and read_ledger already
    # treats it as lost.
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            durable_end = raw.rfind(b"\n") + 1
            with open(path, "rb+") as f:
                f.truncate(durable_end)
                f.flush()
                os.fsync(f.fileno())
            raw = raw[:durable_end]
        existing = read_ledger(path.parent)
        for current in existing:
            if (
                record.idempotency_key
                and current.idempotency_key == record.idempotency_key
            ):
                if (
                    current.step_id,
                    current.phase,
                    current.attempt_id,
                    current.artifact_ref,
                    current.verdict_ref,
                    current.evidence_sha256,
                ) != (
                    record.step_id,
                    record.phase,
                    record.attempt_id,
                    record.artifact_ref,
                    record.verdict_ref,
                    record.evidence_sha256,
                ):
                    raise CheckpointCorrupt(
                        f"ledger idempotency key {record.idempotency_key!r} "
                        "was reused for a different event"
                    )
                return
    else:
        existing = []
    previous = _ledger_sha256(existing[-1]) if existing else None
    record = record.model_copy(update={"prev_event_sha256": previous})
    with open(path, "a") as f:
        f.write(record.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:  # pragma: no cover - platform without directory fsync
        pass


def read_ledger(run_dir: str | Path) -> list[StepRecord]:
    path = Path(run_dir) / LEDGER_FILE
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CheckpointCorrupt(f"ledger path is unsafe: {path}")
    if not path.exists():
        return []
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CheckpointCorrupt(f"unreadable ledger {path}: {error}") from error
    if raw and not raw.endswith(b"\n"):
        logger.warning("dropping torn final ledger line in %s", path)
        raw = raw[: raw.rfind(b"\n") + 1]
    records: list[StepRecord] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(StepRecord.model_validate_json(line))
        except Exception as e:
            raise CheckpointCorrupt(
                f"ledger {path} line {line_number} is corrupt (not a torn tail): {e}"
            ) from e
    seen_events: set[str] = set()
    seen_keys: set[str] = set()
    previous: StepRecord | None = None
    previous_seq = -1
    for record in records:
        expected_hash = _ledger_sha256(previous) if previous is not None else None
        if record.prev_event_sha256 != expected_hash:
            raise CheckpointCorrupt(
                f"ledger {path} hash chain is broken at seq {record.seq}"
            )
        if record.seq <= previous_seq:
            raise CheckpointCorrupt(
                f"ledger {path} sequence is not increasing at {record.seq}"
            )
        if record.event_id in seen_events:
            raise CheckpointCorrupt(
                f"ledger {path} repeats event_id {record.event_id}"
            )
        if record.idempotency_key and record.idempotency_key in seen_keys:
            raise CheckpointCorrupt(
                f"ledger {path} repeats idempotency key {record.idempotency_key}"
            )
        seen_events.add(record.event_id)
        if record.idempotency_key:
            seen_keys.add(record.idempotency_key)
        previous = record
        previous_seq = record.seq
    return records


class FileCheckpointer:
    """LangGraph-shaped facade over the file checkpoint (thread_id == run_id)."""

    def __init__(self, runs_dir: str | Path):
        self.runs_dir = Path(runs_dir)

    def get_tuple(self, thread_id: str) -> RunState | None:
        validate_run_id(thread_id)
        path = self.runs_dir / thread_id / STATE_FILE
        if not path.exists():
            return None
        return load_state_by_id(self.runs_dir, thread_id)

    def put(self, state: RunState) -> None:
        save_state(state)


@contextmanager
def run_lock(run_dir: str | Path) -> Iterator[None]:
    """Hold an OS-backed exclusive lock for one run.

    The lock file remains as harmless metadata, while the kernel lock is
    released automatically on process exit.
    """
    path = Path(run_dir) / ".run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CheckpointCorrupt(f"run lock path is unsafe: {path}")
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as error:
        raise CheckpointCorrupt(f"cannot open run lock safely: {path}: {error}") from error
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise CheckpointCorrupt(f"run lock path is not a regular file: {path}")
    try:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as e:
            raise RunLocked(f"run is already active: {Path(run_dir).name}") from e
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.fsync(fd)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
