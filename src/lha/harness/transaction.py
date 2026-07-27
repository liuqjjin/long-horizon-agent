"""Durable transactions for model-authored code changes."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..clock import now
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
        if status in ("APPLIED", "VERIFIED") and workdir is None:
            raise ValueError(f"{status} transition requires a workdir")
        update: dict = {"status": status, "updated_at": now().isoformat()}
        if status in ("APPLIED", "VERIFIED") and workdir is not None:
            update["applied_state"] = state_for_paths(workdir, self.resolved_paths)
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


def _read_transaction_state(path: Path) -> tuple[PatchTransaction, str]:
    raw = json.loads(path.read_text())
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
            transaction, digest = _read_transaction_state(path)
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


def _fsync_replace(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # pragma: no cover - platform without directory fsync
        pass


def durable_artifact_write(path: Path, data: bytes) -> None:
    """Atomically persist transaction evidence before PREPARED is recorded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:  # pragma: no cover - platform without directory fsync
        pass


def save_transaction(run_dir: Path, tx: PatchTransaction) -> None:
    try:
        tx = PatchTransaction.model_validate(tx.model_dump(mode="json"))
    except Exception as error:
        raise TransactionCorrupt(f"invalid patch transaction: {error}") from error
    path = transaction_path(run_dir, tx.step_id, tx.attempt_id)
    path.parent.mkdir(parents=True, exist_ok=True)
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
        if (
            persisted.schema_version != tx.schema_version
            or persisted.sequence != tx.sequence
        ):
            raise TransactionCorrupt(
                f"transaction sequence changed for {tx.step_id}/{tx.attempt_id}"
            )
        if tx.status not in _TRANSITIONS[persisted.status]:
            raise TransactionCorrupt(
                f"invalid transaction transition: {persisted.status} -> {tx.status}"
            )
    payload = tx.model_dump(mode="json")
    envelope = {
        "schema_version": 1,
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }
    _fsync_replace(path, json.dumps(envelope, indent=2))
    _append_transaction_event(run_dir, tx, envelope["sha256"])


def _append_transaction_event(
    run_dir: Path,
    tx: PatchTransaction,
    transaction_sha256: str,
) -> None:
    path = transaction_log_path(run_dir, tx.step_id, tx.attempt_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = read_transaction_events(run_dir, tx.step_id, tx.attempt_id)
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
    matching = [
        index
        for index, event in enumerate(events)
        if event.transaction_sha256 == transaction_sha256
    ]
    if matching:
        if matching == [len(events) - 1]:
            return
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
    with open(path, "a") as f:
        f.write(json.dumps(envelope, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_transaction_events(
    run_dir: Path,
    step_id: str,
    attempt_id: str,
) -> list[PatchTransactionEvent]:
    """Read the durable phase history, dropping only a torn final append."""
    path = transaction_log_path(run_dir, step_id, attempt_id)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        last_newline = raw.rfind(b"\n")
        raw = raw[: last_newline + 1]
        with open(path, "rb+") as f:
            f.truncate(last_newline + 1)
            f.flush()
            os.fsync(f.fileno())
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
    return events


def load_transaction(
    run_dir: Path,
    step_id: str,
    attempt_id: str,
) -> PatchTransaction | None:
    path = transaction_path(run_dir, step_id, attempt_id)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        payload = raw["payload"]
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        if digest != raw["sha256"]:
            raise ValueError("checksum mismatch")
        tx = PatchTransaction.model_validate(payload)
        # save_transaction writes the state before the append-only event. If a
        # process died between those fsyncs, repair the missing audit record
        # from the checksummed state before recovery continues.
        _append_transaction_event(run_dir, tx, digest)
        return tx
    except Exception as e:
        raise TransactionCorrupt(f"invalid patch transaction {path}: {e}") from e


def list_transactions(run_dir: Path, step_id: str) -> list[PatchTransaction]:
    records = _persisted_transactions(run_dir)
    ordered = _ordered_transactions(
        [transaction for _path, transaction, _digest in records]
    )
    digests = {
        (transaction.step_id, transaction.attempt_id): digest
        for _path, transaction, digest in records
    }
    for transaction in ordered:
        _append_transaction_event(
            run_dir,
            transaction,
            digests[(transaction.step_id, transaction.attempt_id)],
        )
    return [
        transaction for transaction in ordered if transaction.step_id == step_id
    ]


def validate_transaction_journals(run_dir: Path) -> None:
    """Validate the durable transaction journal without pre-empting recovery.

    Backups and reviewed artifacts are deliberately not inspected here.  The
    step recovery path owns those checks because it can still restore from the
    mirrored backup and return a terminal FAILED result.  Rejecting them before
    recovery would strand an applied change in the worktree.
    """
    root = run_dir / "transactions"
    if not root.exists() and not root.is_symlink():
        return
    if root.is_symlink() or not root.is_dir():
        raise TransactionCorrupt(f"transaction directory is unsafe: {root}")

    expected_logs: set[Path] = set()
    actual_logs: set[Path] = set()
    state_paths: list[Path] = []
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
            state_paths.append(descendant)
        else:
            raise TransactionCorrupt(f"unknown transaction evidence: {descendant}")

    records = _persisted_transactions(run_dir)
    _ordered_transactions(
        [transaction for _path, transaction, _digest in records]
    )

    for path in state_paths:
        try:
            raw = json.loads(path.read_text())
            payload = raw["payload"]
            digest = hashlib.sha256(_canonical(payload)).hexdigest()
            if digest != raw["sha256"]:
                raise ValueError("checksum mismatch")
            transaction = PatchTransaction.model_validate(payload)
        except Exception as error:
            raise TransactionCorrupt(
                f"invalid patch transaction {path}: {error}"
            ) from error

        expected_path = transaction_path(
            run_dir, transaction.step_id, transaction.attempt_id
        )
        if path != expected_path:
            raise TransactionCorrupt(
                f"transaction path does not match its identity: {path}"
            )

        log_path = transaction_log_path(
            run_dir, transaction.step_id, transaction.attempt_id
        )
        expected_logs.add(log_path)
        events = read_transaction_events(
            run_dir, transaction.step_id, transaction.attempt_id
        )
        for event in events:
            if (
                event.step_id != transaction.step_id
                or event.attempt_id != transaction.attempt_id
            ):
                raise TransactionCorrupt(
                    f"transaction log identity does not match persisted state: {path}"
                )
            if transaction.schema_version == 3 and (
                event.schema_version != 2
                or event.transaction_sequence != transaction.sequence
            ):
                raise TransactionCorrupt(
                    f"transaction log sequence does not match persisted state: {path}"
                )
            if transaction.schema_version == 2 and event.schema_version != 1:
                raise TransactionCorrupt(
                    f"legacy transaction has sequenced events: {path}"
                )
        for previous, current in zip(events, events[1:]):
            if current.status not in _TRANSITIONS[previous.status]:
                raise TransactionCorrupt(
                    f"invalid transaction log transition: "
                    f"{previous.status} -> {current.status}"
                )
        # A crash can occur after the checksummed state rename but before the
        # matching log append.  The state is authoritative in this narrow
        # window, so restore the missing audit event before continuing.
        if not events or events[-1].transaction_sha256 != digest:
            if events and transaction.status not in _TRANSITIONS[events[-1].status]:
                raise TransactionCorrupt(
                    f"invalid transaction recovery transition: "
                    f"{events[-1].status} -> {transaction.status}"
                )
            _append_transaction_event(run_dir, transaction, digest)
            events = read_transaction_events(
                run_dir, transaction.step_id, transaction.attempt_id
            )
        if (
            not events
            or events[-1].status != transaction.status
            or events[-1].transaction_sha256 != digest
        ):
            raise TransactionCorrupt(
                f"transaction log does not end at the persisted state: {path}"
            )

    orphaned = actual_logs - expected_logs
    if orphaned:
        raise TransactionCorrupt(f"orphaned transaction log: {sorted(orphaned)[0]}")


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
