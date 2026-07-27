"""Host-only control records for formal Terminal-Bench runs.

Harbor bind-mounts its ``trial/agent`` directory into the task container.  It is
therefore suitable for display logs, but not for evidence that the task must not
be able to replace.  This module stores the authoritative markers in a sibling
directory that is never mounted into the task.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..clock import now

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_HEX_32_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_DEFAULT_MAX_FILE_BYTES = 32 * 1024 * 1024


class ControlStoreError(RuntimeError):
    """A host-only control record is missing, unsafe, or inconsistent."""


class ControlRecordExists(ControlStoreError):
    """An immutable control record already exists."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegisteredAttempt(_FrozenModel):
    attempt_id: str
    run_kind: Literal["smoke", "scored"]
    instance_id: str
    command_sha256: str

    @field_validator("attempt_id", "command_sha256")
    @classmethod
    def _sha256_hex(cls, value: str) -> str:
        if _HEX_64_RE.fullmatch(value) is None:
            raise ValueError("attempt and command identifiers must be SHA-256 hex")
        return value


class EvaluationRegistration(_FrozenModel):
    schema_version: Literal[1] = 1
    evaluation_id: str
    protocol_sha256: str
    output_root: str
    created_at: str
    attempts: tuple[RegisteredAttempt, ...]

    @field_validator("evaluation_id")
    @classmethod
    def _evaluation_id_is_random_hex(cls, value: str) -> str:
        if _HEX_32_RE.fullmatch(value) is None:
            raise ValueError("evaluation_id must be 128-bit lowercase hex")
        return value

    @field_validator("protocol_sha256")
    @classmethod
    def _protocol_digest_is_sha256(cls, value: str) -> str:
        if _HEX_64_RE.fullmatch(value) is None:
            raise ValueError("protocol_sha256 must be lowercase SHA-256 hex")
        return value

    @field_validator("output_root")
    @classmethod
    def _output_root_is_absolute(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError("output_root must be an absolute normalized path")
        return str(path)

    @model_validator(mode="after")
    def _attempts_are_unique(self) -> EvaluationRegistration:
        ids = [item.attempt_id for item in self.attempts]
        instances = [(item.run_kind, item.instance_id) for item in self.attempts]
        if len(ids) != len(set(ids)) or len(instances) != len(set(instances)):
            raise ValueError("registered attempts must be unique")
        if len(self.attempts) != 23:
            raise ValueError("a formal evaluation must register 3 smoke and 20 scored attempts")
        if sum(item.run_kind == "smoke" for item in self.attempts) != 3:
            raise ValueError("a formal evaluation must register exactly 3 smoke attempts")
        if sum(item.run_kind == "scored" for item in self.attempts) != 20:
            raise ValueError("a formal evaluation must register exactly 20 scored attempts")
        return self


class ModelStartedMarker(_FrozenModel):
    schema_version: Literal[1] = 1
    evaluation_id: str
    attempt_id: str
    protocol_sha256: str
    run_kind: Literal["smoke", "scored"]
    instance_id: str
    container_id: str
    started_at: str

    @field_validator("evaluation_id")
    @classmethod
    def _evaluation_id_is_hex(cls, value: str) -> str:
        if _HEX_32_RE.fullmatch(value) is None:
            raise ValueError("evaluation_id must be 128-bit lowercase hex")
        return value

    @field_validator("attempt_id", "protocol_sha256")
    @classmethod
    def _digests_are_sha256(cls, value: str) -> str:
        if _HEX_64_RE.fullmatch(value) is None:
            raise ValueError("marker digests must be lowercase SHA-256 hex")
        return value

    @field_validator("container_id")
    @classmethod
    def _container_id_is_full(cls, value: str) -> str:
        if _CONTAINER_ID_RE.fullmatch(value) is None:
            raise ValueError("container_id must be a full lowercase Docker ID")
        return value


class CommandStartedMarker(_FrozenModel):
    """Immutable proof that the host spent this attempt's one command slot."""

    schema_version: Literal[1] = 1
    evaluation_id: str
    attempt_id: str
    run_kind: Literal["smoke", "scored"]
    instance_id: str
    command_sha256: str
    started_at: str

    @field_validator("evaluation_id")
    @classmethod
    def _evaluation_id_is_hex(cls, value: str) -> str:
        if _HEX_32_RE.fullmatch(value) is None:
            raise ValueError("evaluation_id must be 128-bit lowercase hex")
        return value

    @field_validator("attempt_id", "command_sha256")
    @classmethod
    def _digests_are_sha256(cls, value: str) -> str:
        if _HEX_64_RE.fullmatch(value) is None:
            raise ValueError("command marker digests must be lowercase SHA-256 hex")
        return value


class CommandEnvelope(_FrozenModel):
    """Host-side outcome even when Harbor never writes ``result.json``."""

    schema_version: Literal[1] = 1
    evaluation_id: str
    attempt_id: str
    run_kind: Literal["smoke", "scored"]
    instance_id: str
    command_sha256: str
    started_at: str
    finished_at: str
    process_return_code: int | None
    outcome: Literal["completed", "error", "interrupted"]
    failure_stage: Literal[
        "harbor_start",
        "environment_setup",
        "agent_setup",
        "model",
        "verification",
        "result_persistence",
        "unknown",
    ] | None = None
    exception_sha256: str | None = None
    model_started: bool

    @field_validator("evaluation_id")
    @classmethod
    def _evaluation_id_is_hex(cls, value: str) -> str:
        if _HEX_32_RE.fullmatch(value) is None:
            raise ValueError("evaluation_id must be 128-bit lowercase hex")
        return value

    @field_validator("attempt_id", "command_sha256", "exception_sha256")
    @classmethod
    def _digests_are_sha256(cls, value: str | None) -> str | None:
        if value is not None and _HEX_64_RE.fullmatch(value) is None:
            raise ValueError("envelope digests must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def _error_fields_match_outcome(self) -> CommandEnvelope:
        if self.outcome == "completed":
            if self.process_return_code != 0 or self.failure_stage is not None:
                raise ValueError("a completed command must exit zero without a failure stage")
        elif self.failure_stage is None:
            raise ValueError("an unsuccessful command must record its failure stage")
        return self


class SmokeSeal(_FrozenModel):
    schema_version: Literal[2] = 2
    evaluation_id: str
    protocol_sha256: str
    manifest_sha256: str
    smoke_instance_ids: tuple[str, str, str]
    terminal_record_sha256: dict[str, str]
    sealed_at: str

    @field_validator("evaluation_id")
    @classmethod
    def _evaluation_id_is_hex(cls, value: str) -> str:
        if _HEX_32_RE.fullmatch(value) is None:
            raise ValueError("evaluation_id must be 128-bit lowercase hex")
        return value

    @field_validator("protocol_sha256", "manifest_sha256")
    @classmethod
    def _protocol_digest_is_sha256(cls, value: str) -> str:
        if _HEX_64_RE.fullmatch(value) is None:
            raise ValueError("smoke-seal digests must be lowercase SHA-256 hex")
        return value

    @field_validator("terminal_record_sha256")
    @classmethod
    def _record_digests_are_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        if any(_HEX_64_RE.fullmatch(item) is None for item in value.values()):
            raise ValueError("smoke terminal records must use SHA-256 hex")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def _records_cover_smoke_set(self) -> SmokeSeal:
        if set(self.terminal_record_sha256) != set(self.smoke_instance_ids):
            raise ValueError("the smoke seal must cover exactly the three smoke instances")
        return self


def terminal_attempt_id(
    evaluation_id: str,
    run_kind: Literal["smoke", "scored"],
    instance_id: str,
) -> str:
    """Return the stable attempt ID bound to one registered task."""
    if _HEX_32_RE.fullmatch(evaluation_id) is None:
        raise ValueError("evaluation_id must be 128-bit lowercase hex")
    if not instance_id.strip():
        raise ValueError("instance_id may not be empty")
    payload = f"{evaluation_id}\0{run_kind}\0{instance_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def terminal_control_root(output_root: str | Path, evaluation_id: str) -> Path:
    """Derive the host-only root; it is a sibling, never a Harbor child."""
    output = Path(output_root)
    if not output.is_absolute() or output != output.resolve():
        raise ValueError("output_root must be an absolute normalized path")
    if _HEX_32_RE.fullmatch(evaluation_id) is None:
        raise ValueError("evaluation_id must be 128-bit lowercase hex")
    return output.parent / ".lha-control" / evaluation_id


def command_digest(argv: Sequence[str]) -> str:
    """Hash argv without shell re-parsing ambiguities."""
    payload = json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class SecureDirectory:
    """A directory-FD based store that refuses links and unsafe permissions."""

    def __init__(self, path: str | Path, *, expected_mode: int = _DIRECTORY_MODE) -> None:
        self.path = Path(path)
        self._expected_mode = expected_mode
        self._fd = self._open_verified_directory(self.path, expected_mode)

    @classmethod
    def _from_open_fd(
        cls,
        path: Path,
        descriptor: int,
        *,
        expected_mode: int,
    ) -> SecureDirectory:
        value = cls.__new__(cls)
        value.path = path
        value._expected_mode = expected_mode
        value._fd = descriptor
        return value

    @staticmethod
    def _open_verified_directory(path: Path, expected_mode: int) -> int:
        try:
            before = os.lstat(path)
        except OSError as exc:
            raise ControlStoreError(f"control directory is unavailable: {path}") from exc
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != expected_mode
        ):
            raise ControlStoreError(f"control directory is not private and owner-only: {path}")
        flags = os.O_RDONLY | os.O_CLOEXEC
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ControlStoreError(f"control directory cannot be opened safely: {path}") from exc
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or not stat.S_ISDIR(after.st_mode)
            or after.st_uid != os.getuid()
            or stat.S_IMODE(after.st_mode) != expected_mode
        ):
            os.close(descriptor)
            raise ControlStoreError("control directory changed while it was opened")
        return descriptor

    def close(self) -> None:
        descriptor = getattr(self, "_fd", -1)
        if descriptor >= 0:
            os.close(descriptor)
            self._fd = -1

    def __enter__(self) -> SecureDirectory:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._fd < 0:
            raise ControlStoreError("control directory is closed")

    @staticmethod
    def _safe_name(name: str) -> str:
        if _SAFE_NAME_RE.fullmatch(name) is None or name in {".", ".."}:
            raise ControlStoreError(f"unsafe control-record name: {name!r}")
        return name

    def create_directory(self, name: str) -> Path:
        self._check_open()
        name = self._safe_name(name)
        try:
            os.mkdir(name, _DIRECTORY_MODE, dir_fd=self._fd)
            os.fsync(self._fd)
        except FileExistsError as exc:
            raise ControlRecordExists(f"control directory already exists: {name}") from exc
        except OSError as exc:
            raise ControlStoreError(f"control directory could not be created: {name}") from exc
        return self.path / name

    def open_directory(self, name: str) -> SecureDirectory:
        """Open a child relative to this already-verified directory FD."""
        self._check_open()
        name = self._safe_name(name)
        try:
            before = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        except OSError as exc:
            raise ControlStoreError(f"control child directory is unavailable: {name}") from exc
        flags = os.O_RDONLY | os.O_CLOEXEC
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=self._fd)
        except OSError as exc:
            raise ControlStoreError(
                f"control child directory cannot be opened safely: {name}"
            ) from exc
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or not stat.S_ISDIR(after.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or after.st_uid != os.getuid()
            or stat.S_IMODE(after.st_mode) != _DIRECTORY_MODE
        ):
            os.close(descriptor)
            raise ControlStoreError(f"control child directory is unsafe: {name}")
        return self._from_open_fd(
            self.path / name,
            descriptor,
            expected_mode=_DIRECTORY_MODE,
        )

    def has(self, name: str) -> bool:
        self._check_open()
        name = self._safe_name(name)
        try:
            info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ControlStoreError(f"control record could not be inspected: {name}") from exc
        self._validate_file_info(info, name, _DEFAULT_MAX_FILE_BYTES)
        return True

    def write_once(self, name: str, payload: bytes) -> str:
        self._check_open()
        name = self._safe_name(name)
        if len(payload) > _DEFAULT_MAX_FILE_BYTES:
            raise ControlStoreError("control record exceeds the maximum size")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, _FILE_MODE, dir_fd=self._fd)
        except FileExistsError as exc:
            raise ControlRecordExists(f"control record already exists: {name}") from exc
        except OSError as exc:
            raise ControlStoreError(f"control record could not be created: {name}") from exc
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ControlStoreError("control-record write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            info = os.fstat(descriptor)
            self._validate_file_info(info, name, len(payload))
            if info.st_size != len(payload):
                raise ControlStoreError("control-record size changed during write")
        finally:
            os.close(descriptor)
        os.fsync(self._fd)
        return hashlib.sha256(payload).hexdigest()

    def write_json_once(self, name: str, value: BaseModel | Mapping[str, Any]) -> str:
        if isinstance(value, BaseModel):
            plain = value.model_dump(mode="json")
        else:
            plain = dict(value)
        payload = (
            json.dumps(plain, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        return self.write_once(name, payload)

    def read(self, name: str, *, max_bytes: int = _DEFAULT_MAX_FILE_BYTES) -> bytes:
        self._check_open()
        name = self._safe_name(name)
        flags = os.O_RDONLY | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=self._fd)
        except OSError as exc:
            raise ControlStoreError(f"control record is unavailable: {name}") from exc
        try:
            before = os.fstat(descriptor)
            self._validate_file_info(before, name, max_bytes)
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise ControlStoreError("control record ended before its stated size")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ControlStoreError("control record grew while it was read")
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ControlStoreError("control record changed while it was read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def read_json(self, name: str, model: type[BaseModel]) -> BaseModel:
        try:
            return model.model_validate_json(self.read(name))
        except ValueError as exc:
            raise ControlStoreError(f"control record is invalid: {name}") from exc

    @staticmethod
    def _validate_file_info(info: os.stat_result, name: str, max_bytes: int) -> None:
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != _FILE_MODE
            or info.st_size < 0
            or info.st_size > max_bytes
        ):
            raise ControlStoreError(f"control record is not a private regular file: {name}")


def initialize_control_store(
    *,
    evaluation_id: str,
    protocol_sha256: str,
    output_root: str | Path,
    attempts: Sequence[RegisteredAttempt],
) -> EvaluationRegistration:
    """Create the output and host-control roots exactly once."""
    output = Path(output_root)
    if not output.is_absolute() or output != output.resolve():
        raise ValueError("output_root must be an absolute normalized path")
    control = terminal_control_root(output, evaluation_id)
    if output.exists():
        raise ControlRecordExists(f"Terminal-Bench output root already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output, _DIRECTORY_MODE)

    control_parent = control.parent
    try:
        os.mkdir(control_parent, _DIRECTORY_MODE)
    except FileExistsError:
        with SecureDirectory(control_parent):
            pass
    try:
        os.mkdir(control, _DIRECTORY_MODE)
    except FileExistsError as exc:
        raise ControlRecordExists(
            f"Terminal-Bench control root already exists: {control}"
        ) from exc

    registration = EvaluationRegistration(
        evaluation_id=evaluation_id,
        protocol_sha256=protocol_sha256,
        output_root=str(output),
        created_at=now().isoformat(),
        attempts=tuple(attempts),
    )
    with SecureDirectory(control) as store:
        store.write_json_once("registration.json", registration)
        for attempt in registration.attempts:
            store.create_directory(attempt.attempt_id)
    return registration


def open_attempt_store(
    output_root: str | Path,
    evaluation_id: str,
    attempt_id: str,
) -> SecureDirectory:
    if _HEX_64_RE.fullmatch(attempt_id) is None:
        raise ValueError("attempt_id must be lowercase SHA-256 hex")
    control = terminal_control_root(output_root, evaluation_id)
    parent = SecureDirectory(control)
    try:
        child = parent.open_directory(attempt_id)
    finally:
        parent.close()
    return child


@contextmanager
def evaluation_lock(
    output_root: str | Path,
    evaluation_id: str,
) -> Iterator[int]:
    """Hold one kernel lease for the evaluation and expose it to Harbor.

    The Harbor process inherits the descriptor. If the controller is killed,
    the lease therefore remains held until that in-flight command exits.
    """
    import fcntl

    control = terminal_control_root(output_root, evaluation_id)
    parent = SecureDirectory(control)
    name = "evaluation.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(name, flags, _FILE_MODE, dir_fd=parent._fd)
        except OSError as exc:
            raise ControlStoreError("evaluation lock could not be opened safely") from exc
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != _FILE_MODE
            ):
                raise ControlStoreError("evaluation lock is not a private regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                raise ControlStoreError("formal evaluation is already active") from exc
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(descriptor)
            yield descriptor
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)
    finally:
        parent.close()


def write_model_started(
    store: SecureDirectory,
    *,
    evaluation_id: str,
    attempt_id: str,
    protocol_sha256: str,
    run_kind: Literal["smoke", "scored"],
    instance_id: str,
    container_id: str,
) -> ModelStartedMarker:
    marker = ModelStartedMarker(
        evaluation_id=evaluation_id,
        attempt_id=attempt_id,
        protocol_sha256=protocol_sha256,
        run_kind=run_kind,
        instance_id=instance_id,
        container_id=container_id,
        started_at=now().isoformat(),
    )
    store.write_json_once("MODEL_STARTED.json", marker)
    return marker


def write_command_started(
    store: SecureDirectory,
    *,
    evaluation_id: str,
    attempt_id: str,
    run_kind: Literal["smoke", "scored"],
    instance_id: str,
    command_sha256: str,
) -> CommandStartedMarker:
    marker = CommandStartedMarker(
        evaluation_id=evaluation_id,
        attempt_id=attempt_id,
        run_kind=run_kind,
        instance_id=instance_id,
        command_sha256=command_sha256,
        started_at=now().isoformat(),
    )
    store.write_json_once("COMMAND_STARTED.json", marker)
    return marker
