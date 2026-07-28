"""Durable transactions for model-authored code changes."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..clock import now
from ..durable_io import (
    AnchoredAtomicReplaceTemp,
    anchored_atomic_replace_bytes,
    anchored_read_bytes,
    anchored_update_bytes,
    atomic_replace_temp_target_name,
    durable_mkdir_chain,
    inspect_anchored_atomic_replace_temp,
    remove_anchored_atomic_replace_temp,
)
from ..step_ids import canonical_artifact_segment
from ..tools.patch import ResolvedPatch
from .errors import TransactionCorrupt
from .manifest import FileState, file_state, saved_file_state

TransactionStatus = Literal["PREPARED", "APPLIED", "VERIFIED", "REVERTED"]
_TRANSITIONS: dict[TransactionStatus, frozenset[TransactionStatus]] = {
    "PREPARED": frozenset({"PREPARED", "APPLIED", "REVERTED"}),
    "APPLIED": frozenset({"APPLIED", "VERIFIED", "REVERTED"}),
    "VERIFIED": frozenset({"VERIFIED", "REVERTED"}),
    "REVERTED": frozenset({"REVERTED"}),
}


class PatchTransaction(BaseModel):
    schema_version: Literal[2, 3] = 3
    sequence: int | None = Field(default=None, ge=1)
    step_id: str
    attempt_id: str
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_paths: list[str] = Field(default_factory=list)
    backup_ref: str
    backup_mirror_ref: str
    backup_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: TransactionStatus = "PREPARED"
    applied_state: dict[str, FileState] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: now().isoformat())
    updated_at: str = Field(default_factory=lambda: now().isoformat())

    @model_validator(mode="after")
    def _state_matches_phase(self) -> "PatchTransaction":
        if self.schema_version == 3 and self.sequence is None:
            raise ValueError("schema-3 transaction requires a sequence")
        if self.schema_version == 2 and self.sequence is not None:
            raise ValueError("schema-2 transaction cannot carry a sequence")
        checked_paths = [_canonical_transaction_path(path) for path in self.resolved_paths]
        if checked_paths != sorted(set(checked_paths)):
            raise ValueError("resolved_paths must be canonical, unique, and sorted")
        aliases = [unicodedata.normalize("NFC", path).casefold() for path in checked_paths]
        if len(aliases) != len(set(aliases)):
            raise ValueError("resolved_paths contain case-insensitive aliases")
        expected_backup = (
            Path("backups")
            / _safe_seg(self.step_id)
            / f"{_safe_seg(self.attempt_id)}.json"
        ).as_posix()
        expected_mirror = (
            Path("steps")
            / _safe_seg(self.step_id)
            / "attempts"
            / _safe_seg(self.attempt_id)
            / "backup.json"
        ).as_posix()
        if self.backup_ref != expected_backup:
            raise ValueError("backup_ref does not match the transaction identity")
        if self.backup_mirror_ref != expected_mirror:
            raise ValueError(
                "backup_mirror_ref does not match the transaction identity"
            )
        if self.status in ("APPLIED", "VERIFIED"):
            if set(self.applied_state) != set(self.resolved_paths):
                raise ValueError("applied transaction must record every resolved path")
        elif self.applied_state:
            raise ValueError(f"{self.status} transaction must not carry applied_state")
        return self

    def transition(
        self,
        status: TransactionStatus,
        *,
        workdir: Path | None = None,
    ) -> "PatchTransaction":
        if status not in _TRANSITIONS[self.status]:
            raise ValueError(f"invalid transaction transition: {self.status} -> {status}")
        if status == "APPLIED" and workdir is None:
            raise ValueError("APPLIED transition requires a workdir")
        update: dict = {"status": status, "updated_at": now().isoformat()}
        if status == "APPLIED" and workdir is not None:
            update["applied_state"] = state_for_paths(workdir, self.resolved_paths)
        elif status == "VERIFIED":
            # APPLIED is the one point that captures the bytes produced by this
            # attempt. A later repair may supersede one of those paths before the
            # whole step verifies, so VERIFIED is a status-only transition.
            update["applied_state"] = self.applied_state
        elif status == "REVERTED":
            update["applied_state"] = {}
        return self.model_copy(update=update)


class PatchTransactionEvent(BaseModel):
    """One checksummed, append-only transaction phase record."""

    schema_version: Literal[1, 2] = 2
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_sequence: int | None = Field(default=None, ge=1)
    transaction_sequence: int | None = Field(default=None, ge=1)
    at: str
    step_id: str
    attempt_id: str
    status: TransactionStatus
    transaction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _sequence_matches_schema(self) -> "PatchTransactionEvent":
        if self.schema_version == 2:
            if self.event_sequence is None or self.transaction_sequence is None:
                raise ValueError("schema-2 transaction event requires both sequences")
        elif self.event_sequence is not None or self.transaction_sequence is not None:
            raise ValueError("schema-1 transaction event cannot carry sequences")
        return self


def state_for_paths(workdir: Path, paths: list[str]) -> dict[str, FileState]:
    return {
        rel: file_state(resolve_worktree_target(workdir, rel))
        for rel in paths
    }


def resolve_worktree_target(workdir: Path, relative: str) -> Path:
    """Resolve one transaction path without following a symlink component."""
    relative = _canonical_transaction_path(relative)
    if workdir.is_symlink() or not workdir.is_dir():
        raise TransactionCorrupt(f"worktree root is missing or unsafe: {workdir}")
    root = workdir.resolve()
    target = root / relative
    probe = target.parent
    while probe != root:
        if probe.is_symlink():
            raise TransactionCorrupt(
                f"worktree path contains a symlink: {relative}"
            )
        if probe.exists() and not probe.is_dir():
            raise TransactionCorrupt(
                f"worktree path has a non-directory parent: {relative}"
            )
        probe = probe.parent
    return target


def _canonical_transaction_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe transaction path: {value!r}")
    normalized = unicodedata.normalize("NFC", value)
    path = Path(normalized)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"unsafe transaction path: {value!r}")
    return value


def resolve_transaction_evidence(run_dir: Path, reference: str) -> Path:
    """Resolve a persisted transaction reference without following links."""
    relative = Path(_canonical_transaction_path(reference))
    root = run_dir.resolve()
    candidate = root / relative
    probe = candidate
    while probe != root:
        if probe.is_symlink():
            raise TransactionCorrupt(
                f"transaction evidence path contains a symlink: {reference}"
            )
        probe = probe.parent
    return candidate


def transaction_dir(run_dir: Path, step_id: str) -> Path:
    return run_dir / "transactions" / _safe_seg(step_id)


def transaction_path(run_dir: Path, step_id: str, attempt_id: str) -> Path:
    return transaction_dir(run_dir, step_id) / f"{_safe_seg(attempt_id)}.json"


def transaction_log_path(run_dir: Path, step_id: str, attempt_id: str) -> Path:
    return transaction_dir(run_dir, step_id) / f"{_safe_seg(attempt_id)}.events.jsonl"


def attempt_artifact_dir(run_dir: Path, step_id: str, attempt_id: str) -> Path:
    return (
        run_dir
        / "steps"
        / _safe_seg(step_id)
        / "attempts"
        / _safe_seg(attempt_id)
    )


def _safe_seg(value: str) -> str:
    return canonical_artifact_segment(value)


def _read_transaction_state(
    run_dir: Path,
    path: Path,
) -> tuple[PatchTransaction, str]:
    encoded = anchored_read_bytes(path, anchor=run_dir)
    assert encoded is not None
    return _decode_transaction_state_bytes(encoded, path)


def _decode_transaction_state_bytes(
    encoded: bytes,
    path: Path,
) -> tuple[PatchTransaction, str]:
    """Decode and authenticate one transaction state snapshot."""
    raw = json.loads(encoded)
    payload = raw["payload"]
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if digest != raw["sha256"]:
        raise ValueError("checksum mismatch")
    return PatchTransaction.model_validate(payload), digest


def _persisted_transactions(
    run_dir: Path,
) -> list[tuple[Path, PatchTransaction, str]]:
    root = run_dir / "transactions"
    if not root.exists():
        return []
    records: list[tuple[Path, PatchTransaction, str]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            transaction, digest = _read_transaction_state(run_dir, path)
        except Exception as error:
            raise TransactionCorrupt(
                f"invalid patch transaction {path}: {error}"
            ) from error
        expected = transaction_path(
            run_dir, transaction.step_id, transaction.attempt_id
        )
        if path != expected:
            raise TransactionCorrupt(
                f"transaction path does not match its identity: {path}"
            )
        records.append((path, transaction, digest))
    return records


def _ordered_transactions(
    transactions: list[PatchTransaction],
) -> list[PatchTransaction]:
    """Order current transactions without trusting wall-clock timestamps.

    Schema 2 predates durable ordering. A single old transaction is still safe
    to load or recover because no ordering choice exists. More than one remains
    individually inspectable, but ordered recovery fails closed: timestamps
    cannot prove which write happened first.
    """
    legacy = [
        transaction
        for transaction in transactions
        if transaction.schema_version == 2
    ]
    current = [
        transaction
        for transaction in transactions
        if transaction.schema_version == 3
    ]
    if legacy and current:
        raise TransactionCorrupt(
            "cannot mix legacy and sequenced transaction histories"
        )
    if legacy:
        if len(legacy) > 1:
            raise TransactionCorrupt(
                "legacy transaction history has no durable ordering"
            )
        return legacy

    sequences = [int(transaction.sequence or 0) for transaction in current]
    expected = list(range(1, len(current) + 1))
    if sorted(sequences) != expected:
        raise TransactionCorrupt(
            "transaction sequences must be unique and contiguous from 1"
        )
    return sorted(current, key=lambda transaction: int(transaction.sequence or 0))


def _next_transaction_sequence(run_dir: Path) -> int:
    existing = [
        transaction
        for _path, transaction, _digest in _persisted_transactions(run_dir)
    ]
    if any(transaction.schema_version == 2 for transaction in existing):
        raise TransactionCorrupt(
            "cannot append a sequenced transaction to a legacy history"
        )
    return len(_ordered_transactions(existing)) + 1


def build_transaction(
    *,
    run_dir: Path,
    step_id: str,
    attempt_id: str,
    resolved: ResolvedPatch,
    backup_sha256: str,
) -> PatchTransaction:
    backup = (
        Path("backups")
        / transaction_dir(run_dir, step_id).name
        / f"{transaction_path(run_dir, step_id, attempt_id).stem}.json"
    )
    mirror = attempt_artifact_dir(run_dir, step_id, attempt_id).relative_to(run_dir) / "backup.json"
    return PatchTransaction(
        sequence=_next_transaction_sequence(run_dir),
        step_id=step_id,
        attempt_id=attempt_id,
        patch_sha256=resolved.patch_sha256,
        resolved_paths=resolved.paths,
        backup_ref=backup.as_posix(),
        backup_mirror_ref=mirror.as_posix(),
        backup_sha256=backup_sha256,
    )


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fsync_replace(
    path: Path,
    data: str,
    *,
    anchor: Path | None = None,
) -> None:
    anchored_atomic_replace_bytes(
        path,
        data.encode("utf-8"),
        anchor=anchor or path.parent,
    )


def durable_artifact_write(path: Path, data: bytes) -> None:
    """Atomically persist transaction evidence before PREPARED is recorded."""
    anchor = path.parent
    while True:
        try:
            anchor.lstat()
            break
        except FileNotFoundError:
            if anchor.parent == anchor:
                raise
            anchor = anchor.parent
    anchored_atomic_replace_bytes(path, data, anchor=anchor)


def save_transaction(run_dir: Path, tx: PatchTransaction) -> None:
    try:
        tx = PatchTransaction.model_validate(tx.model_dump(mode="json"))
    except Exception as error:
        raise TransactionCorrupt(f"invalid patch transaction: {error}") from error
    path = transaction_path(run_dir, tx.step_id, tx.attempt_id)
    # A previous state-before-event crash must be repaired explicitly by the
    # locked resume path.  Never replace the main record while its current
    # journal binding is missing or corrupt.
    validate_transaction_journals(run_dir)
    records = _persisted_transactions(run_dir)
    _ordered_transactions(
        [transaction for _path, transaction, _digest in records]
    )
    persisted = next(
        (
            transaction
            for existing_path, transaction, _digest in records
            if existing_path == path
        ),
        None,
    )
    if persisted is None:
        if tx.schema_version == 3:
            existing = [transaction for _path, transaction, _digest in records]
            if any(transaction.schema_version == 2 for transaction in existing):
                raise TransactionCorrupt(
                    "cannot append a sequenced transaction to a legacy history"
                )
            if tx.sequence != len(_ordered_transactions(existing)) + 1:
                raise TransactionCorrupt(
                    "new transaction does not use the next durable sequence"
                )
        elif any(
            transaction.schema_version == 3
            for _path, transaction, _digest in records
        ):
            raise TransactionCorrupt(
                "cannot append a legacy transaction to a sequenced history"
            )
    else:
        _validate_transaction_update(persisted, tx)
        if tx.status not in _TRANSITIONS[persisted.status]:
            raise TransactionCorrupt(
                f"invalid transaction transition: {persisted.status} -> {tx.status}"
            )
    durable_mkdir_chain(path.parent, anchor=run_dir)
    payload = tx.model_dump(mode="json")
    envelope = {
        "schema_version": 1,
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }
    _fsync_replace(path, json.dumps(envelope, indent=2), anchor=run_dir)
    _append_transaction_event(run_dir, tx, envelope["sha256"])


def _validate_transaction_update(
    persisted: PatchTransaction,
    candidate: PatchTransaction,
) -> None:
    """Freeze the patch and recovery bindings after PREPARED is durable."""
    immutable_fields = (
        "schema_version",
        "sequence",
        "step_id",
        "attempt_id",
        "patch_sha256",
        "resolved_paths",
        "backup_ref",
        "backup_mirror_ref",
        "backup_sha256",
        "created_at",
    )
    changed = [
        field
        for field in immutable_fields
        if getattr(persisted, field) != getattr(candidate, field)
    ]
    if changed:
        raise TransactionCorrupt(
            f"transaction binding changed for "
            f"{persisted.step_id}/{persisted.attempt_id}: {', '.join(changed)}"
        )

    # PREPARED -> APPLIED captures the resulting bytes once.  Verification may
    # confirm that evidence, but must not replace it with a different snapshot.
    if (
        persisted.status in ("APPLIED", "VERIFIED")
        and candidate.status in ("APPLIED", "VERIFIED")
        and persisted.applied_state != candidate.applied_state
    ):
        raise TransactionCorrupt(
            f"applied transaction evidence changed for "
            f"{persisted.step_id}/{persisted.attempt_id}"
        )


def _append_transaction_event(
    run_dir: Path,
    tx: PatchTransaction,
    transaction_sha256: str,
) -> None:
    path = transaction_log_path(run_dir, tx.step_id, tx.attempt_id)
    durable_mkdir_chain(path.parent, anchor=run_dir)
    events = read_transaction_events(run_dir, tx.step_id, tx.attempt_id)
    _validate_transaction_event_history(tx, events, path)
    if not _transaction_event_needs_append(tx, transaction_sha256, events):
        return
    event = PatchTransactionEvent(
        schema_version=2 if tx.schema_version == 3 else 1,
        event_sequence=len(events) + 1 if tx.schema_version == 3 else None,
        transaction_sequence=tx.sequence,
        at=tx.updated_at,
        step_id=tx.step_id,
        attempt_id=tx.attempt_id,
        status=tx.status,
        transaction_sha256=transaction_sha256,
    )
    payload = event.model_dump(mode="json")
    envelope = {
        "schema_version": 1,
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }
    line = (json.dumps(envelope, sort_keys=True) + "\n").encode()
    try:
        anchored_update_bytes(
            path,
            lambda current: (current or b"") + line,
            anchor=run_dir,
        )
    except OSError as error:
        raise TransactionCorrupt(f"transaction log is unsafe: {path}: {error}") from error


def _validate_transaction_event_history(
    tx: PatchTransaction,
    events: list[PatchTransactionEvent],
    path: Path,
) -> None:
    if tx.schema_version == 3:
        for event in events:
            if (
                event.schema_version != 2
                or event.transaction_sequence != tx.sequence
            ):
                raise TransactionCorrupt(
                    f"transaction event sequence does not match "
                    f"{tx.step_id}/{tx.attempt_id}"
                )
    elif any(event.schema_version != 1 for event in events):
        raise TransactionCorrupt(
            f"legacy transaction has sequenced events: {tx.step_id}/{tx.attempt_id}"
        )
    for event in events:
        if event.step_id != tx.step_id or event.attempt_id != tx.attempt_id:
            raise TransactionCorrupt(
                f"transaction log identity does not match persisted state: {path}"
            )
    for previous, current in zip(events, events[1:]):
        if current.status not in _TRANSITIONS[previous.status]:
            raise TransactionCorrupt(
                f"invalid transaction log transition: "
                f"{previous.status} -> {current.status}"
            )


def _transaction_event_needs_append(
    tx: PatchTransaction,
    transaction_sha256: str,
    events: list[PatchTransactionEvent],
) -> bool:
    matching = [
        index
        for index, event in enumerate(events)
        if event.transaction_sha256 == transaction_sha256
    ]
    if matching:
        if matching == [len(events) - 1]:
            if events[-1].status != tx.status:
                raise TransactionCorrupt(
                    f"transaction event status does not match persisted state for "
                    f"{tx.step_id}/{tx.attempt_id}"
                )
            return False
        raise TransactionCorrupt(
            f"transaction state repeats an earlier journal event for "
            f"{tx.step_id}/{tx.attempt_id}"
        )
    if not events and tx.status != "PREPARED":
        raise TransactionCorrupt(
            f"transaction history for {tx.step_id}/{tx.attempt_id} "
            "does not start at PREPARED"
        )
    if events and tx.status not in _TRANSITIONS[events[-1].status]:
        raise TransactionCorrupt(
            f"invalid transaction log transition: "
            f"{events[-1].status} -> {tx.status}"
        )
    return True


def read_transaction_events(
    run_dir: Path,
    step_id: str,
    attempt_id: str,
) -> list[PatchTransactionEvent]:
    """Read the durable phase history without changing recovery evidence."""
    _path, events, _committed_size = _read_transaction_events(
        run_dir,
        step_id,
        attempt_id,
        allow_torn_tail=False,
    )
    return events


def _read_transaction_events(
    run_dir: Path,
    step_id: str,
    attempt_id: str,
    *,
    allow_torn_tail: bool,
) -> tuple[Path, list[PatchTransactionEvent], int | None]:
    path = transaction_log_path(run_dir, step_id, attempt_id)
    try:
        value = anchored_read_bytes(path, anchor=run_dir, missing_ok=True)
    except OSError as error:
        raise TransactionCorrupt(f"transaction log is unsafe: {path}: {error}") from error
    if value is None:
        return path, [], None
    events, committed_size = _decode_transaction_event_bytes(
        value,
        path,
        allow_torn_tail=allow_torn_tail,
    )
    return path, events, committed_size


def _decode_transaction_event_bytes(
    value: bytes,
    path: Path,
    *,
    allow_torn_tail: bool,
) -> tuple[list[PatchTransactionEvent], int | None]:
    """Decode a journal snapshot without trusting the name that supplied it."""
    raw = value
    committed_size: int | None = None
    if raw and not raw.endswith(b"\n"):
        if not allow_torn_tail:
            raise TransactionCorrupt(
                f"transaction log has a torn final append: {path}"
            )
        committed_size = raw.rfind(b"\n") + 1
        raw = raw[:committed_size]
    events: list[PatchTransactionEvent] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            envelope = json.loads(line)
            payload = envelope["payload"]
            digest = hashlib.sha256(_canonical(payload)).hexdigest()
            if digest != envelope["sha256"]:
                raise ValueError("checksum mismatch")
            events.append(PatchTransactionEvent.model_validate(payload))
        except Exception as e:
            raise TransactionCorrupt(
                f"invalid transaction log {path} line {line_number}: {e}"
            ) from e
    if events and events[0].status != "PREPARED":
        raise TransactionCorrupt(
            f"transaction log {path} does not start at PREPARED"
        )
    versions = {event.schema_version for event in events}
    if len(versions) > 1:
        raise TransactionCorrupt(
            f"transaction log {path} mixes event schemas"
        )
    if versions == {2}:
        event_sequences = [event.event_sequence for event in events]
        expected = list(range(1, len(events) + 1))
        if event_sequences != expected:
            raise TransactionCorrupt(
                f"transaction log {path} event sequences are not contiguous"
            )
        transaction_sequences = {
            event.transaction_sequence for event in events
        }
        if len(transaction_sequences) != 1:
            raise TransactionCorrupt(
                f"transaction log {path} changes transaction sequence"
            )
    return events, committed_size


def _truncate_transaction_log(path: Path, committed_size: int) -> None:
    """Discard one uncommitted append while the caller holds the run lock."""
    try:
        run_dir = path.parents[2]
    except IndexError as error:
        raise TransactionCorrupt(f"transaction log path is invalid: {path}") from error
    try:
        anchored_update_bytes(
            path,
            lambda current: _truncate_bytes(current, committed_size, path),
            anchor=run_dir,
        )
    except OSError as error:
        raise TransactionCorrupt(f"transaction log is unsafe: {path}: {error}") from error


def _truncate_bytes(
    current: bytes | None,
    committed_size: int,
    path: Path,
) -> bytes:
    if current is None or committed_size < 0 or committed_size > len(current):
        raise TransactionCorrupt(f"transaction log truncation is invalid: {path}")
    return current[:committed_size]


def load_transaction(
    run_dir: Path,
    step_id: str,
    attempt_id: str,
) -> PatchTransaction | None:
    path = transaction_path(run_dir, step_id, attempt_id)
    try:
        encoded = anchored_read_bytes(path, anchor=run_dir, missing_ok=True)
        if encoded is None:
            return None
        raw = json.loads(encoded)
        payload = raw["payload"]
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        if digest != raw["sha256"]:
            raise ValueError("checksum mismatch")
        tx = PatchTransaction.model_validate(payload)
        if (
            tx.step_id != step_id
            or tx.attempt_id != attempt_id
            or path != transaction_path(run_dir, tx.step_id, tx.attempt_id)
        ):
            raise ValueError("transaction path does not match its identity")
        events = read_transaction_events(run_dir, tx.step_id, tx.attempt_id)
        _validate_transaction_event_history(tx, events, path)
        if (
            not events
            or events[-1].status != tx.status
            or events[-1].transaction_sha256 != digest
        ):
            raise ValueError("transaction log does not end at the persisted state")
        return tx
    except Exception as e:
        raise TransactionCorrupt(f"invalid patch transaction {path}: {e}") from e


def list_transactions(run_dir: Path, step_id: str) -> list[PatchTransaction]:
    validate_transaction_journals(run_dir)
    records = _persisted_transactions(run_dir)
    ordered = _ordered_transactions(
        [transaction for _path, transaction, _digest in records]
    )
    return [
        transaction for transaction in ordered if transaction.step_id == step_id
    ]


def _transaction_temp_target(
    run_dir: Path,
    root: Path,
    temporary: Path,
) -> tuple[Path, Literal["state", "events"], str, str]:
    """Bind one exact temporary name to a legal transaction evidence target."""
    try:
        relative = temporary.relative_to(root)
    except ValueError as error:
        raise TransactionCorrupt(
            f"transaction temporary file escapes its root: {temporary}"
        ) from error
    if len(relative.parts) != 2:
        raise TransactionCorrupt(
            f"transaction temporary file has an invalid location: {temporary}"
        )
    try:
        step_id = _safe_seg(relative.parts[0])
    except ValueError as error:
        raise TransactionCorrupt(
            f"transaction temporary file has an invalid step: {temporary}"
        ) from error

    target_name = atomic_replace_temp_target_name(temporary.name)
    if target_name is None:
        raise TransactionCorrupt(
            f"unknown transaction temporary file: {temporary}"
        )
    if target_name.endswith(".events.jsonl"):
        attempt_id = target_name[: -len(".events.jsonl")]
        kind: Literal["state", "events"] = "events"
    elif target_name.endswith(".json"):
        attempt_id = target_name[: -len(".json")]
        kind = "state"
    else:
        raise TransactionCorrupt(
            f"transaction temporary file targets unknown evidence: {temporary}"
        )
    try:
        attempt_id = _safe_seg(attempt_id)
    except ValueError as error:
        raise TransactionCorrupt(
            f"transaction temporary file has an invalid attempt: {temporary}"
        ) from error
    expected = (
        transaction_path(run_dir, step_id, attempt_id)
        if kind == "state"
        else transaction_log_path(run_dir, step_id, attempt_id)
    )
    if expected.parent != temporary.parent or expected.name != target_name:
        raise TransactionCorrupt(
            f"transaction temporary target does not match its location: {temporary}"
        )
    return expected, kind, step_id, attempt_id


def _validate_transaction_state_temp(
    run_dir: Path,
    target: Path,
    record: AnchoredAtomicReplaceTemp,
    records: list[tuple[Path, PatchTransaction, str]],
) -> None:
    try:
        candidate, _digest = _decode_transaction_state_bytes(record.data, target)
    except Exception as error:
        raise TransactionCorrupt(
            f"invalid transaction atomic-replace temporary file {record.path}: {error}"
        ) from error
    if target != transaction_path(
        run_dir,
        candidate.step_id,
        candidate.attempt_id,
    ):
        raise TransactionCorrupt(
            f"transaction temporary payload does not own its target: {record.path}"
        )

    persisted = next(
        (
            transaction
            for path, transaction, _digest in records
            if path == target
        ),
        None,
    )
    if persisted is not None:
        _validate_transaction_update(persisted, candidate)
        if candidate.status not in _TRANSITIONS[persisted.status]:
            raise TransactionCorrupt(
                f"invalid temporary transaction transition: "
                f"{persisted.status} -> {candidate.status}"
            )
        return

    if candidate.status != "PREPARED":
        raise TransactionCorrupt(
            f"new transaction temporary file is not PREPARED: {record.path}"
        )
    existing = [transaction for _path, transaction, _digest in records]
    if candidate.schema_version == 3:
        if any(transaction.schema_version == 2 for transaction in existing):
            raise TransactionCorrupt(
                "new transaction temporary file follows a legacy history"
            )
        if candidate.sequence != len(_ordered_transactions(existing)) + 1:
            raise TransactionCorrupt(
                f"transaction temporary file has an invalid sequence: {record.path}"
            )
    elif any(transaction.schema_version == 3 for transaction in existing):
        raise TransactionCorrupt(
            "legacy transaction temporary file follows a sequenced history"
        )


def _validate_transaction_event_temp(
    run_dir: Path,
    target: Path,
    record: AnchoredAtomicReplaceTemp,
    step_id: str,
    attempt_id: str,
    records: list[tuple[Path, PatchTransaction, str]],
) -> None:
    state_target = transaction_path(run_dir, step_id, attempt_id)
    persisted = next(
        (
            transaction
            for path, transaction, _digest in records
            if path == state_target
        ),
        None,
    )
    if persisted is None:
        raise TransactionCorrupt(
            f"transaction journal temporary file has no durable state: {record.path}"
        )
    try:
        events, _committed_size = _decode_transaction_event_bytes(
            record.data,
            target,
            allow_torn_tail=False,
        )
        _validate_transaction_event_history(persisted, events, state_target)
    except Exception as error:
        raise TransactionCorrupt(
            f"invalid transaction atomic-replace temporary file {record.path}: {error}"
        ) from error
    if events:
        return

    # An empty candidate is only produced when recovery truncates a torn first
    # append. It is not a valid stand-alone journal temporary file.
    _path, _current, committed_size = _read_transaction_events(
        run_dir,
        step_id,
        attempt_id,
        allow_torn_tail=True,
    )
    if committed_size != 0:
        raise TransactionCorrupt(
            f"empty transaction journal temporary file is not recoverable: {record.path}"
        )


def _recover_transaction_atomic_temps(run_dir: Path) -> None:
    """Remove one proven uncommitted replace file while the run lock is held."""
    root = run_dir / "transactions"
    if not root.exists() and not root.is_symlink():
        return
    if root.is_symlink() or not root.is_dir():
        raise TransactionCorrupt(f"transaction directory is unsafe: {root}")
    run_metadata = run_dir.lstat()
    root_metadata = root.lstat()
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise TransactionCorrupt(f"run directory is unsafe: {run_dir}")
    if root_metadata.st_uid != run_metadata.st_uid:
        raise TransactionCorrupt(f"transaction directory owner is unsafe: {root}")

    temporary_paths: list[Path] = []
    actual_logs: set[Path] = set()
    state_paths: set[Path] = set()
    for descendant in sorted(root.rglob("*")):
        if descendant.is_symlink():
            raise TransactionCorrupt(
                f"refusing symlink transaction evidence: {descendant}"
            )
        if descendant.is_dir():
            continue
        if descendant.name.endswith(".tmp"):
            temporary_paths.append(descendant)
        elif descendant.name.endswith(".events.jsonl"):
            actual_logs.add(descendant)
        elif descendant.suffix == ".json":
            state_paths.add(descendant)
        else:
            raise TransactionCorrupt(f"unknown transaction evidence: {descendant}")
    if not temporary_paths:
        return
    if len(temporary_paths) != 1:
        raise TransactionCorrupt(
            "multiple transaction atomic-replace temporary files require manual review"
        )

    records = _persisted_transactions(run_dir)
    _ordered_transactions(
        [transaction for _path, transaction, _digest in records]
    )
    if {path for path, _transaction, _digest in records} != state_paths:
        raise TransactionCorrupt("transaction state inventory changed while reading")
    expected_logs = {
        transaction_log_path(
            run_dir,
            transaction.step_id,
            transaction.attempt_id,
        )
        for _path, transaction, _digest in records
    }
    orphaned = actual_logs - expected_logs
    if orphaned:
        raise TransactionCorrupt(f"orphaned transaction log: {sorted(orphaned)[0]}")
    for path, transaction, _digest in records:
        _log_path, events, _committed_size = _read_transaction_events(
            run_dir,
            transaction.step_id,
            transaction.attempt_id,
            allow_torn_tail=True,
        )
        _validate_transaction_event_history(transaction, events, path)

    temporary = temporary_paths[0]
    target, kind, step_id, attempt_id = _transaction_temp_target(
        run_dir,
        root,
        temporary,
    )
    try:
        record = inspect_anchored_atomic_replace_temp(
            temporary,
            anchor=run_dir,
            expected_target_name=target.name,
            owner_uid=run_metadata.st_uid,
            mode=0o600,
        )
    except OSError as error:
        raise TransactionCorrupt(
            f"transaction atomic-replace temporary file is unsafe: "
            f"{temporary}: {error}"
        ) from error
    if kind == "state":
        _validate_transaction_state_temp(run_dir, target, record, records)
    else:
        _validate_transaction_event_temp(
            run_dir,
            target,
            record,
            step_id,
            attempt_id,
            records,
        )
    try:
        remove_anchored_atomic_replace_temp(
            record,
            anchor=run_dir,
            owner_uid=run_metadata.st_uid,
            mode=0o600,
        )
    except OSError as error:
        raise TransactionCorrupt(
            f"could not remove transaction atomic-replace temporary file "
            f"{temporary}: {error}"
        ) from error


def _transaction_journal_inventory(
    run_dir: Path,
) -> list[tuple[Path, PatchTransaction, str]]:
    root = run_dir / "transactions"
    if not root.exists() and not root.is_symlink():
        return []
    if root.is_symlink() or not root.is_dir():
        raise TransactionCorrupt(f"transaction directory is unsafe: {root}")

    expected_logs: set[Path] = set()
    actual_logs: set[Path] = set()
    state_paths: set[Path] = set()
    for descendant in root.rglob("*"):
        if descendant.is_symlink():
            raise TransactionCorrupt(
                f"refusing symlink transaction evidence: {descendant}"
            )
        if descendant.is_dir():
            continue
        if descendant.name.endswith(".events.jsonl"):
            actual_logs.add(descendant)
        elif descendant.suffix == ".json":
            state_paths.add(descendant)
        else:
            raise TransactionCorrupt(f"unknown transaction evidence: {descendant}")

    records = _persisted_transactions(run_dir)
    _ordered_transactions(
        [transaction for _path, transaction, _digest in records]
    )
    if {path for path, _transaction, _digest in records} != state_paths:
        raise TransactionCorrupt("transaction state inventory changed while reading")

    for _path, transaction, _digest in records:
        expected_logs.add(
            transaction_log_path(
                run_dir, transaction.step_id, transaction.attempt_id
            )
        )

    orphaned = actual_logs - expected_logs
    if orphaned:
        raise TransactionCorrupt(f"orphaned transaction log: {sorted(orphaned)[0]}")
    return records


def recover_transaction_journals(run_dir: Path) -> None:
    """Repair only the state-before-event crash window.

    This function writes recovery evidence.  Its caller must hold ``run_lock``
    for the run, and it must only be used by a resume entry point.  Inspection
    and retention paths call ``validate_transaction_journals`` instead.
    """
    repairs: list[tuple[PatchTransaction, str]] = []
    truncations: list[tuple[Path, int]] = []
    _recover_transaction_atomic_temps(run_dir)
    records = _transaction_journal_inventory(run_dir)

    # Validate every committed prefix before changing any journal.  A damaged
    # second transaction must not cause an otherwise valid first one to be
    # silently rewritten during inspection.
    for path, transaction, digest in records:
        log_path, events, committed_size = _read_transaction_events(
            run_dir,
            transaction.step_id,
            transaction.attempt_id,
            allow_torn_tail=True,
        )
        _validate_transaction_event_history(transaction, events, path)
        if _transaction_event_needs_append(transaction, digest, events):
            repairs.append((transaction, digest))
        if committed_size is not None:
            truncations.append((log_path, committed_size))

    for log_path, committed_size in truncations:
        _truncate_transaction_log(log_path, committed_size)
    for transaction, digest in repairs:
        _append_transaction_event(run_dir, transaction, digest)
    validate_transaction_journals(run_dir)


def validate_transaction_journals(run_dir: Path) -> None:
    """Validate durable transaction journals without changing any file.

    Backups and reviewed artifacts are deliberately not inspected here.  The
    step recovery path owns those checks because it can still restore from the
    mirrored backup and return a terminal FAILED result.
    """
    for path, transaction, digest in _transaction_journal_inventory(run_dir):
        events = read_transaction_events(
            run_dir, transaction.step_id, transaction.attempt_id
        )
        _validate_transaction_event_history(transaction, events, path)
        if (
            not events
            or events[-1].status != transaction.status
            or events[-1].transaction_sha256 != digest
        ):
            raise TransactionCorrupt(
                f"transaction log does not end at the persisted state: {path}"
            )


def validate_applied_state(tx: PatchTransaction, workdir: Path) -> None:
    actual = state_for_paths(workdir, tx.resolved_paths)
    if actual != tx.applied_state:
        raise TransactionCorrupt(
            f"worktree drift for {tx.step_id}/{tx.attempt_id}: "
            "current files do not match the applied transaction"
        )


def validate_terminal_transaction_state(
    run_dir: Path,
    workdir: Path,
    status: Literal["DONE", "FAILED"],
) -> None:
    """Prove that a terminal run has no unresolved patch side effects.

    The transaction journal proves which phase was durably recorded, while the
    two backups and the worktree prove the resulting bytes.  This check is
    intentionally usable by both resume and reporting so a terminal label
    cannot bypass the recovery boundary.
    """
    from ..tools.patch import backup_sha256, load_backup

    validate_transaction_journals(run_dir)
    root = run_dir / "transactions"
    if not root.exists():
        return

    transactions = _ordered_transactions(
        [
            transaction
            for _path, transaction, _digest in _persisted_transactions(run_dir)
        ]
    )
    bases: dict[tuple[str, str], dict[str, FileState]] = {}
    for transaction in transactions:
        backups = []
        for reference in (
            transaction.backup_ref,
            transaction.backup_mirror_ref,
        ):
            try:
                backup = load_backup(
                    resolve_transaction_evidence(run_dir, reference),
                    run_dir=run_dir,
                    required=True,
                )
                assert backup is not None
            except Exception as error:
                raise TransactionCorrupt(
                    f"terminal transaction backup is unusable for "
                    f"{transaction.step_id}/{transaction.attempt_id}: {error}"
                ) from error
            if backup_sha256(backup) != transaction.backup_sha256:
                raise TransactionCorrupt(
                    f"terminal transaction backup checksum mismatch for "
                    f"{transaction.step_id}/{transaction.attempt_id}"
                )
            if (
                set(backup.originals) != set(transaction.resolved_paths)
                or set(backup.modes) != set(transaction.resolved_paths)
            ):
                raise TransactionCorrupt(
                    f"terminal transaction backup write set mismatch for "
                    f"{transaction.step_id}/{transaction.attempt_id}"
                )
            backups.append(backup)
        bases[(transaction.step_id, transaction.attempt_id)] = {
            relative: saved_file_state(
                backups[0].originals[relative],
                backups[0].modes[relative],
            )
            for relative in transaction.resolved_paths
        }

    unresolved = [
        transaction
        for transaction in transactions
        if transaction.status in ("PREPARED", "APPLIED")
    ]
    if unresolved:
        first = unresolved[0]
        raise TransactionCorrupt(
            f"{status} run contains unresolved transaction "
            f"{first.step_id}/{first.attempt_id}: {first.status}"
        )

    by_path: dict[str, list[PatchTransaction]] = {}
    for transaction in transactions:
        for relative in transaction.resolved_paths:
            by_path.setdefault(relative, []).append(transaction)

    expected: dict[str, FileState] = {}
    for relative, history in by_path.items():
        confirmed: FileState | None = None
        rollback_open = False
        for transaction in history:
            if transaction.status == "VERIFIED":
                if rollback_open:
                    restarted_base = bases[
                        (transaction.step_id, transaction.attempt_id)
                    ][relative]
                    if restarted_base != confirmed:
                        raise TransactionCorrupt(
                            f"transaction after rollback does not start from "
                            f"the restored state for {relative}"
                        )
                    rollback_open = False
                confirmed = transaction.applied_state[relative]
                continue
            if transaction.status == "REVERTED" and not rollback_open:
                restored = bases[
                    (transaction.step_id, transaction.attempt_id)
                ][relative]
                if confirmed is not None and restored != confirmed:
                    raise TransactionCorrupt(
                        f"rollback base does not match the preceding verified "
                        f"state for {relative}"
                    )
                confirmed = restored
                rollback_open = True
        if confirmed is not None:
            expected[relative] = confirmed

    if not expected:
        return
    actual = state_for_paths(workdir, sorted(expected))
    mismatches = [
        relative
        for relative, expected_state in expected.items()
        if actual.get(relative) != expected_state
    ]
    if mismatches:
        raise TransactionCorrupt(
            "terminal worktree does not match the confirmed transaction state: "
            + ", ".join(sorted(mismatches))
        )
