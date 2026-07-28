"""Host-only control records for formal Terminal-Bench runs.

Harbor bind-mounts its ``trial/agent`` directory into the task container.  It is
therefore suitable for display logs, but not for evidence that the task must not
be able to replace.  This module stores the authoritative markers in a sibling
directory that is never mounted into the task.

The store rejects unsafe path types, survives controller crashes, and
serializes normal runners. It is not a security boundary against another
process running as the same host user; such a process can modify these files or
trace the controller directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
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
_INITIALIZATION_ANCHOR_DIRECTORY = ".lha-terminal-initialization"
_REGISTRATION = "registration.json"
_EVALUATION_LOCK = "evaluation.lock"
_CONTROL_LIFECYCLE_RECORDS = {
    _EVALUATION_LOCK,
    "smoke_manifest.json",
    "smoke_seal.json",
}


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


class _FileIdentity(_FrozenModel):
    device: int
    inode: int


class _OutputBinding(_FrozenModel):
    schema_version: Literal[1] = 1
    evaluation_id: str
    protocol_sha256: str
    output_root: str
    attempts: tuple[RegisteredAttempt, ...]
    created_at: str


class _RegistrationCommit(_FrozenModel):
    schema_version: Literal[1] = 1
    evaluation_id: str
    protocol_sha256: str
    output_root: str
    binding_sha256: str
    registration_sha256: str
    evaluation_lock_identity: _FileIdentity
    committed_at: str

    @field_validator("binding_sha256", "registration_sha256")
    @classmethod
    def _digests_are_sha256(cls, value: str) -> str:
        if _HEX_64_RE.fullmatch(value) is None:
            raise ValueError("registration-commit digests must be lowercase SHA-256 hex")
        return value


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

    @staticmethod
    def _pending_name(name: str) -> str:
        digest = hashlib.sha256(name.encode()).hexdigest()
        return f"pending-{digest}"

    def _entry_info(self, name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ControlStoreError(f"control record could not be inspected: {name}") from exc

    def _open_locked_pending(self, name: str, *, create: bool) -> tuple[int, str] | None:
        import fcntl

        pending = self._pending_name(name)
        flags = os.O_RDWR | os.O_CLOEXEC
        if create:
            flags |= os.O_CREAT
        flags |= getattr(os, "O_NOFOLLOW", 0)
        for _attempt in range(3):
            try:
                descriptor = os.open(pending, flags, _FILE_MODE, dir_fd=self._fd)
            except FileNotFoundError:
                if not create:
                    return None
                continue
            except OSError as exc:
                raise ControlStoreError(
                    f"pending control record could not be opened: {name}"
                ) from exc
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                info = os.fstat(descriptor)
                self._validate_file_info(
                    info,
                    pending,
                    _DEFAULT_MAX_FILE_BYTES,
                    allowed_links={1, 2},
                )
                current = self._entry_info(pending)
                if current is not None and (
                    current.st_dev,
                    current.st_ino,
                ) == (
                    info.st_dev,
                    info.st_ino,
                ):
                    return descriptor, pending
            except BaseException:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
                raise
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            if self._entry_info(name) is not None:
                return None
        raise ControlStoreError(f"pending control record changed repeatedly: {name}")

    @staticmethod
    def _close_locked_pending(descriptor: int) -> None:
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def pending_exists(self, name: str) -> bool:
        """Validate and report an unpublished crash-recovery record."""
        self._check_open()
        name = self._safe_name(name)
        locked = self._open_locked_pending(name, create=False)
        if locked is None:
            return False
        descriptor, _pending = locked
        self._close_locked_pending(descriptor)
        return True

    def _recover_linked_pending(self, name: str) -> None:
        """Finish the only safe crash state after a no-replace link publish."""
        locked = self._open_locked_pending(name, create=False)
        if locked is None:
            return
        descriptor, pending = locked
        try:
            pending_info = os.fstat(descriptor)
            final_info = self._entry_info(name)
            if final_info is None:
                if pending_info.st_nlink != 1:
                    raise ControlStoreError(
                        f"pending control record has an invalid link count: {name}"
                    )
                return
            self._validate_file_info(
                final_info,
                name,
                _DEFAULT_MAX_FILE_BYTES,
                allowed_links={2},
            )
            if (
                pending_info.st_nlink != 2
                or (pending_info.st_dev, pending_info.st_ino)
                != (final_info.st_dev, final_info.st_ino)
            ):
                raise ControlStoreError(
                    f"pending control record does not match its published record: {name}"
                )
            try:
                os.unlink(pending, dir_fd=self._fd)
                os.fsync(self._fd)
            except OSError as exc:
                raise ControlStoreError(
                    f"published control record could not be finalized: {name}"
                ) from exc
        finally:
            self._close_locked_pending(descriptor)

    def _read_entry(self, name: str, *, max_bytes: int) -> bytes:
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
        self._recover_linked_pending(name)
        info = self._entry_info(name)
        if info is None:
            return False
        self._validate_file_info(info, name, _DEFAULT_MAX_FILE_BYTES)
        return True

    def write_once(self, name: str, payload: bytes) -> str:
        self._check_open()
        name = self._safe_name(name)
        if len(payload) > _DEFAULT_MAX_FILE_BYTES:
            raise ControlStoreError("control record exceeds the maximum size")
        self._recover_linked_pending(name)
        if self._entry_info(name) is not None:
            raise ControlRecordExists(f"control record already exists: {name}")

        locked = self._open_locked_pending(name, create=True)
        if locked is None:
            raise ControlRecordExists(f"control record already exists: {name}")
        descriptor, pending = locked
        try:
            pending_info = os.fstat(descriptor)
            final_info = self._entry_info(name)
            if final_info is not None:
                if (
                    pending_info.st_nlink == 2
                    and (pending_info.st_dev, pending_info.st_ino)
                    == (final_info.st_dev, final_info.st_ino)
                ):
                    os.unlink(pending, dir_fd=self._fd)
                    os.fsync(self._fd)
                elif pending_info.st_nlink == 1:
                    os.unlink(pending, dir_fd=self._fd)
                    os.fsync(self._fd)
                else:
                    raise ControlStoreError(
                        f"pending control record has an invalid link count: {name}"
                    )
                raise ControlRecordExists(f"control record already exists: {name}")
            if pending_info.st_nlink != 1:
                raise ControlStoreError(
                    f"pending control record has an invalid link count: {name}"
                )
            try:
                os.ftruncate(descriptor, 0)
                os.lseek(descriptor, 0, os.SEEK_SET)
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise ControlStoreError("control-record write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
                info = os.fstat(descriptor)
                self._validate_file_info(info, pending, len(payload))
                if info.st_size != len(payload):
                    raise ControlStoreError("control-record size changed during write")
            except OSError as exc:
                raise ControlStoreError(
                    f"pending control record could not be written: {name}"
                ) from exc
            try:
                os.link(
                    pending,
                    name,
                    src_dir_fd=self._fd,
                    dst_dir_fd=self._fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise ControlRecordExists(
                    f"control record already exists: {name}"
                ) from exc
            except OSError as exc:
                raise ControlStoreError(
                    f"control record could not be published atomically: {name}"
                ) from exc
            try:
                os.fsync(self._fd)
                os.unlink(pending, dir_fd=self._fd)
                os.fsync(self._fd)
            except OSError as exc:
                raise ControlStoreError(
                    f"control record publication could not be finalized: {name}"
                ) from exc
            final_info = self._entry_info(name)
            if final_info is None:  # pragma: no cover - only pending was unlinked
                raise ControlStoreError(f"published control record disappeared: {name}")
            self._validate_file_info(final_info, name, len(payload))
            return hashlib.sha256(payload).hexdigest()
        finally:
            self._close_locked_pending(descriptor)

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
        self._recover_linked_pending(name)
        return self._read_entry(name, max_bytes=max_bytes)

    def read_json(self, name: str, model: type[BaseModel]) -> BaseModel:
        try:
            return model.model_validate_json(self.read(name))
        except ValueError as exc:
            raise ControlStoreError(f"control record is invalid: {name}") from exc

    @staticmethod
    def _validate_file_info(
        info: os.stat_result,
        name: str,
        max_bytes: int,
        *,
        allowed_links: set[int] | None = None,
    ) -> None:
        links = {1} if allowed_links is None else allowed_links
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink not in links
            or stat.S_IMODE(info.st_mode) != _FILE_MODE
            or info.st_size < 0
            or info.st_size > max_bytes
        ):
            raise ControlStoreError(f"control record is not a private regular file: {name}")


def _verify_directory_path(
    store: SecureDirectory,
    path: Path,
    *,
    label: str,
) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ControlStoreError(f"{label} disappeared during initialization") from exc
    opened = os.fstat(store._fd)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != _DIRECTORY_MODE
        or (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ControlStoreError(f"{label} was replaced during initialization")


def _file_identity(descriptor: int) -> _FileIdentity:
    info = os.fstat(descriptor)
    return _FileIdentity(device=info.st_dev, inode=info.st_ino)


def _verify_file_path(
    store: SecureDirectory,
    name: str,
    descriptor: int,
    *,
    label: str,
) -> None:
    try:
        info = os.stat(name, dir_fd=store._fd, follow_symlinks=False)
    except OSError as exc:
        raise ControlStoreError(f"{label} disappeared") from exc
    opened = os.fstat(descriptor)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != _FILE_MODE
        or (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ControlStoreError(f"{label} was replaced")


def _read_model(
    store: SecureDirectory,
    name: str,
    model: type[BaseModel],
) -> tuple[BaseModel, str]:
    payload = store.read(name)
    try:
        value = model.model_validate_json(payload)
    except ValueError as exc:
        raise ControlStoreError(f"control record is invalid: {name}") from exc
    return value, hashlib.sha256(payload).hexdigest()


@contextmanager
def _store_lock(
    store: SecureDirectory,
    name: str,
    *,
    blocking: bool,
    create: bool,
    busy_message: str,
) -> Iterator[int]:
    import fcntl

    flags = os.O_RDWR | os.O_CLOEXEC
    if create:
        flags |= os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    attempts = 2 if create else 1
    for attempt in range(attempts):
        try:
            descriptor = os.open(name, flags, _FILE_MODE, dir_fd=store._fd)
            break
        except FileNotFoundError:
            if not create or attempt == attempts - 1:
                raise ControlStoreError(f"control lock is missing: {name}") from None
        except OSError as exc:
            raise ControlStoreError(f"control lock could not be opened: {name}") from exc
    if descriptor < 0:  # pragma: no cover - loop either opens or raises
        raise ControlStoreError(f"control lock could not be opened: {name}")
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != _FILE_MODE
        ):
            raise ControlStoreError(f"control lock is not a private regular file: {name}")
        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, operation)
        except (BlockingIOError, OSError) as exc:
            raise ControlStoreError(busy_message) from exc
        try:
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _output_key(output: Path) -> str:
    return hashlib.sha256(str(output).encode()).hexdigest()


def _anchor_record_names(output: Path) -> tuple[str, str, str]:
    key = _output_key(output)
    return (
        f"{key}.binding.json",
        f"{key}.commit.json",
        f"{key}.lock",
    )


def _output_binding_matches(
    value: _OutputBinding,
    registration: EvaluationRegistration,
) -> bool:
    return (
        value.evaluation_id == registration.evaluation_id
        and value.protocol_sha256 == registration.protocol_sha256
        and value.output_root == registration.output_root
        and value.attempts == registration.attempts
    )


def _registration_binding_matches(
    recorded: EvaluationRegistration,
    expected: EvaluationRegistration,
) -> bool:
    return (
        recorded.evaluation_id == expected.evaluation_id
        and recorded.protocol_sha256 == expected.protocol_sha256
        and recorded.output_root == expected.output_root
        and recorded.attempts == expected.attempts
    )


def _registration_matches_output_binding(
    registration: EvaluationRegistration,
    binding: _OutputBinding,
) -> bool:
    return (
        registration.evaluation_id == binding.evaluation_id
        and registration.protocol_sha256 == binding.protocol_sha256
        and registration.output_root == binding.output_root
        and registration.attempts == binding.attempts
        and registration.created_at == binding.created_at
    )


def _initialization_anchor(output: Path, *, create: bool) -> SecureDirectory:
    path = output.parent / _INITIALIZATION_ANCHOR_DIRECTORY
    if create:
        try:
            os.mkdir(path, _DIRECTORY_MODE)
        except FileExistsError:
            pass
    return SecureDirectory(path)


def _open_attempt_directories(
    stack: ExitStack,
    store: SecureDirectory,
    attempts: Sequence[RegisteredAttempt],
) -> dict[str, SecureDirectory]:
    opened: dict[str, SecureDirectory] = {}
    for attempt in attempts:
        opened[attempt.attempt_id] = stack.enter_context(
            store.open_directory(attempt.attempt_id)
        )
    return opened


def _registration_commit_matches(
    commit: _RegistrationCommit,
    *,
    binding: _OutputBinding,
    binding_sha256: str,
    registration_sha256: str,
    evaluation_lock_descriptor: int,
) -> bool:
    return (
        commit.evaluation_id == binding.evaluation_id
        and commit.protocol_sha256 == binding.protocol_sha256
        and commit.output_root == binding.output_root
        and commit.binding_sha256 == binding_sha256
        and commit.registration_sha256 == registration_sha256
        and commit.evaluation_lock_identity == _file_identity(evaluation_lock_descriptor)
    )


def _control_entries(
    store: SecureDirectory,
    attempts: Sequence[RegisteredAttempt],
    *,
    allow_pending_registration: bool,
) -> set[str]:
    expected_attempts = {attempt.attempt_id for attempt in attempts}
    allowed = {
        _REGISTRATION,
        _EVALUATION_LOCK,
        *_CONTROL_LIFECYCLE_RECORDS,
        *expected_attempts,
    }
    pending_sources = {"smoke_manifest.json", "smoke_seal.json"}
    if allow_pending_registration:
        pending_sources.add(_REGISTRATION)
    allowed.update(SecureDirectory._pending_name(name) for name in pending_sources)

    entries = set(os.listdir(store._fd))
    for name in _CONTROL_LIFECYCLE_RECORDS & entries:
        store.has(name)
    for name in pending_sources:
        if SecureDirectory._pending_name(name) in entries:
            store.pending_exists(name)

    # A writer can finish while pending_exists waits for its kernel lock.
    entries = set(os.listdir(store._fd))
    unexpected = entries - allowed
    if unexpected:
        raise ControlStoreError("Terminal-Bench control root contains unexpected records")
    for name in _CONTROL_LIFECYCLE_RECORDS & entries:
        store.has(name)
    return entries


def _validate_lifecycle_records(
    store: SecureDirectory,
    entries: set[str],
    registration: EvaluationRegistration,
) -> None:
    manifest_name = "smoke_manifest.json"
    seal_name = "smoke_seal.json"
    manifest_exists = manifest_name in entries
    seal_exists = seal_name in entries
    manifest_pending = SecureDirectory._pending_name(manifest_name) in entries
    seal_pending = SecureDirectory._pending_name(seal_name) in entries
    if not manifest_exists:
        if seal_exists or seal_pending:
            raise ControlStoreError("smoke seal exists without its manifest")
        if manifest_pending:
            return
        return

    # Imported only for a completed smoke phase, after terminal_bench finished
    # importing this module.
    from .terminal_bench import HarborExecutionManifest

    loaded_manifest, manifest_sha256 = _read_model(
        store,
        manifest_name,
        HarborExecutionManifest,
    )
    manifest: Any = loaded_manifest
    smoke_ids = tuple(
        attempt.instance_id
        for attempt in registration.attempts
        if attempt.run_kind == "smoke"
    )
    if (
        manifest.run_kind != "smoke"
        or manifest.protocol_sha256 != registration.protocol_sha256
        or manifest.expected_instance_ids != smoke_ids
        or manifest.observed_instance_ids != smoke_ids
    ):
        raise ControlStoreError("smoke manifest changed its registration binding")
    if not seal_exists:
        # The controller may have stopped after publishing the manifest or
        # while writing the seal. seal_smoke_phase validates and completes it.
        return
    loaded_seal, _seal_sha256 = _read_model(store, seal_name, SmokeSeal)
    if not isinstance(loaded_seal, SmokeSeal):
        raise ControlStoreError("smoke seal has an unexpected model type")
    terminal_records = {
        instance_id: manifest.terminal_record_sha256[instance_id]
        for instance_id in smoke_ids
    }
    if (
        any(value is None for value in terminal_records.values())
        or loaded_seal.evaluation_id != registration.evaluation_id
        or loaded_seal.protocol_sha256 != registration.protocol_sha256
        or loaded_seal.manifest_sha256 != manifest_sha256
        or loaded_seal.smoke_instance_ids != smoke_ids
        or loaded_seal.terminal_record_sha256 != terminal_records
    ):
        raise ControlStoreError("smoke lifecycle records changed their registration binding")


def _real_directory_exists(path: Path, *, label: str) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ControlStoreError(f"{label} could not be inspected") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ControlStoreError(f"{label} is not a real directory")
    return True


def initialize_control_store(
    *,
    evaluation_id: str,
    protocol_sha256: str,
    output_root: str | Path,
    attempts: Sequence[RegisteredAttempt],
) -> EvaluationRegistration:
    """Initialize once, or validate an already committed control topology."""
    output = Path(output_root)
    if not output.is_absolute() or output != output.resolve():
        raise ValueError("output_root must be an absolute normalized path")
    control = terminal_control_root(output, evaluation_id)
    expected_registration = EvaluationRegistration(
        evaluation_id=evaluation_id,
        protocol_sha256=protocol_sha256,
        output_root=str(output),
        created_at=now().isoformat(),
        attempts=tuple(attempts),
    )
    binding_name, commit_name, initialization_lock_name = _anchor_record_names(output)
    control_parent = control.parent

    output.parent.mkdir(parents=True, exist_ok=True)
    with _initialization_anchor(output, create=True) as anchor_store:
        with _store_lock(
            anchor_store,
            initialization_lock_name,
            blocking=True,
            create=True,
            busy_message="Terminal-Bench initialization is already active",
        ):
            binding: _OutputBinding | None = None
            binding_sha256: str | None = None
            if anchor_store.has(binding_name):
                loaded, binding_sha256 = _read_model(
                    anchor_store,
                    binding_name,
                    _OutputBinding,
                )
                if not isinstance(loaded, _OutputBinding):
                    raise ControlStoreError("output binding has an unexpected model type")
                binding = loaded
                if not _output_binding_matches(binding, expected_registration):
                    raise ControlStoreError(
                        "Terminal-Bench output root is registered to another evaluation"
                    )

            commit: _RegistrationCommit | None = None
            if anchor_store.has(commit_name):
                loaded, _commit_sha256 = _read_model(
                    anchor_store,
                    commit_name,
                    _RegistrationCommit,
                )
                if not isinstance(loaded, _RegistrationCommit):
                    raise ControlStoreError("registration commit has an unexpected model type")
                commit = loaded
                if binding is None:
                    raise ControlStoreError("registration commit has no output binding")

            output_exists = _real_directory_exists(
                output,
                label="Terminal-Bench output root",
            )
            control_exists = _real_directory_exists(
                control,
                label="Terminal-Bench control root",
            )
            if commit is not None and not output_exists:
                raise ControlStoreError("committed Terminal-Bench output root was removed")
            if commit is not None and not control_exists:
                raise ControlStoreError("committed Terminal-Bench control root was removed")

            if not control_exists:
                if output_exists:
                    with SecureDirectory(output) as existing_output:
                        if os.listdir(existing_output._fd):
                            raise ControlStoreError(
                                "cannot create control state after output started"
                            )
                try:
                    os.mkdir(control_parent, _DIRECTORY_MODE)
                except FileExistsError:
                    with SecureDirectory(control_parent):
                        pass
                os.mkdir(control, _DIRECTORY_MODE)

            with ExitStack() as stack:
                control_store = stack.enter_context(SecureDirectory(control))
                registration: EvaluationRegistration | None = None
                registration_sha256: str | None = None
                if control_store.has(_REGISTRATION):
                    loaded, registration_sha256 = _read_model(
                        control_store,
                        _REGISTRATION,
                        EvaluationRegistration,
                    )
                    if not isinstance(loaded, EvaluationRegistration):
                        raise ControlStoreError(
                            "registration has an unexpected model type"
                        )
                    registration = loaded
                    if not _registration_binding_matches(
                        registration,
                        expected_registration,
                    ):
                        raise ControlStoreError(
                            "existing Terminal-Bench registration changed its binding"
                        )
                if commit is not None and registration is None:
                    raise ControlStoreError("committed Terminal-Bench registration was removed")

                if not output_exists:
                    if registration is not None:
                        raise ControlStoreError(
                            "Terminal-Bench output root was removed after registration"
                        )
                    os.mkdir(output, _DIRECTORY_MODE)
                output_store = stack.enter_context(SecureDirectory(output))

                entries = _control_entries(
                    control_store,
                    expected_registration.attempts,
                    allow_pending_registration=registration is None,
                )
                expected_attempt_ids = {
                    attempt.attempt_id for attempt in expected_registration.attempts
                }
                existing_attempt_ids = entries & expected_attempt_ids
                if registration is None:
                    if entries & (_CONTROL_LIFECYCLE_RECORDS - {_EVALUATION_LOCK}):
                        raise ControlStoreError(
                            "unregistered control root contains lifecycle evidence"
                        )
                    if os.listdir(output_store._fd):
                        raise ControlStoreError(
                            "unregistered Terminal-Bench output root is not empty"
                        )
                    for attempt_id in existing_attempt_ids:
                        with control_store.open_directory(attempt_id) as attempt_store:
                            if os.listdir(attempt_store._fd):
                                raise ControlStoreError(
                                    "unregistered attempt directory is not empty"
                                )
                    if binding is None:
                        binding = _OutputBinding(
                            evaluation_id=evaluation_id,
                            protocol_sha256=protocol_sha256,
                            output_root=str(output),
                            attempts=expected_registration.attempts,
                            created_at=expected_registration.created_at,
                        )
                        binding_sha256 = anchor_store.write_json_once(
                            binding_name,
                            binding,
                        )

                lock_exists = _EVALUATION_LOCK in entries
                if registration is not None and not lock_exists:
                    raise ControlStoreError(
                        "registered Terminal-Bench evaluation lock was removed"
                    )
                with _store_lock(
                    control_store,
                    _EVALUATION_LOCK,
                    blocking=False,
                    create=registration is None and not lock_exists,
                    busy_message="formal evaluation is already active",
                ) as evaluation_lock_descriptor:
                    entries = _control_entries(
                        control_store,
                        expected_registration.attempts,
                        allow_pending_registration=registration is None,
                    )
                    existing_attempt_ids = entries & expected_attempt_ids

                    if registration is None:
                        for attempt in expected_registration.attempts:
                            if attempt.attempt_id not in existing_attempt_ids:
                                control_store.create_directory(attempt.attempt_id)
                        attempt_stores = _open_attempt_directories(
                            stack,
                            control_store,
                            expected_registration.attempts,
                        )
                        for attempt_store in attempt_stores.values():
                            os.fsync(attempt_store._fd)
                        if binding is None or binding_sha256 is None:
                            raise ControlStoreError("output binding was not persisted")
                        registration = EvaluationRegistration(
                            evaluation_id=binding.evaluation_id,
                            protocol_sha256=binding.protocol_sha256,
                            output_root=binding.output_root,
                            created_at=binding.created_at,
                            attempts=binding.attempts,
                        )
                        registration_sha256 = control_store.write_json_once(
                            _REGISTRATION,
                            registration,
                        )
                    else:
                        attempt_stores = _open_attempt_directories(
                            stack,
                            control_store,
                            registration.attempts,
                        )
                        _validate_lifecycle_records(
                            control_store,
                            entries,
                            registration,
                        )
                        if binding is None:
                            binding = _OutputBinding(
                                evaluation_id=registration.evaluation_id,
                                protocol_sha256=registration.protocol_sha256,
                                output_root=registration.output_root,
                                attempts=registration.attempts,
                                created_at=registration.created_at,
                            )
                            binding_sha256 = anchor_store.write_json_once(
                                binding_name,
                                binding,
                            )

                    if (
                        binding is None
                        or binding_sha256 is None
                        or registration is None
                        or registration_sha256 is None
                    ):
                        raise ControlStoreError("Terminal-Bench registration is incomplete")
                    if not _registration_matches_output_binding(registration, binding):
                        raise ControlStoreError(
                            "registration changed its output binding"
                        )

                    if commit is None:
                        commit = _RegistrationCommit(
                            evaluation_id=registration.evaluation_id,
                            protocol_sha256=registration.protocol_sha256,
                            output_root=registration.output_root,
                            binding_sha256=binding_sha256,
                            registration_sha256=registration_sha256,
                            evaluation_lock_identity=_file_identity(
                                evaluation_lock_descriptor
                            ),
                            committed_at=now().isoformat(),
                        )
                        anchor_store.write_json_once(commit_name, commit)
                    elif not _registration_commit_matches(
                        commit,
                        binding=binding,
                        binding_sha256=binding_sha256,
                        registration_sha256=registration_sha256,
                        evaluation_lock_descriptor=evaluation_lock_descriptor,
                    ):
                        raise ControlStoreError(
                            "registration commit changed its immutable binding"
                        )

                    _control_entries(
                        control_store,
                        registration.attempts,
                        allow_pending_registration=False,
                    )
                    _verify_directory_path(
                        anchor_store,
                        output.parent / _INITIALIZATION_ANCHOR_DIRECTORY,
                        label="initialization anchor",
                    )
                    _verify_directory_path(
                        output_store,
                        output,
                        label="Terminal-Bench output root",
                    )
                    _verify_directory_path(
                        control_store,
                        control,
                        label="Terminal-Bench control root",
                    )
                    _verify_file_path(
                        control_store,
                        _EVALUATION_LOCK,
                        evaluation_lock_descriptor,
                        label="Terminal-Bench evaluation lock",
                    )
                    for attempt_id, attempt_store in attempt_stores.items():
                        _verify_directory_path(
                            attempt_store,
                            control / attempt_id,
                            label=f"Terminal-Bench attempt directory {attempt_id}",
                        )
                    # Recheck the roots after the per-attempt pass. Consumers
                    # repeat this validation before using any descriptor.
                    _verify_directory_path(
                        output_store,
                        output,
                        label="Terminal-Bench output root",
                    )
                    _verify_directory_path(
                        control_store,
                        control,
                        label="Terminal-Bench control root",
                    )
                    for attempt_id, attempt_store in attempt_stores.items():
                        _verify_directory_path(
                            attempt_store,
                            control / attempt_id,
                            label=f"Terminal-Bench attempt directory {attempt_id}",
                        )
                    return registration


@dataclass(frozen=True)
class _CommittedTopology:
    registration: EvaluationRegistration
    anchor_store: SecureDirectory
    output_store: SecureDirectory
    control_store: SecureDirectory
    attempt_stores: dict[str, SecureDirectory]
    evaluation_lock_descriptor: int


def _open_committed_topology(
    stack: ExitStack,
    *,
    output: Path,
    evaluation_id: str,
    lock_readwrite: bool,
    validate_lifecycle: bool,
) -> _CommittedTopology:
    control = terminal_control_root(output, evaluation_id)
    binding_name, commit_name, _initialization_lock_name = _anchor_record_names(output)
    anchor_store = stack.enter_context(_initialization_anchor(output, create=False))
    loaded, binding_sha256 = _read_model(
        anchor_store,
        binding_name,
        _OutputBinding,
    )
    if not isinstance(loaded, _OutputBinding):
        raise ControlStoreError("output binding has an unexpected model type")
    binding = loaded
    if binding.evaluation_id != evaluation_id or binding.output_root != str(output):
        raise ControlStoreError("Terminal-Bench output binding does not match this evaluation")
    loaded, _commit_sha256 = _read_model(
        anchor_store,
        commit_name,
        _RegistrationCommit,
    )
    if not isinstance(loaded, _RegistrationCommit):
        raise ControlStoreError("registration commit has an unexpected model type")
    commit = loaded

    output_store = stack.enter_context(SecureDirectory(output))
    control_store = stack.enter_context(SecureDirectory(control))
    loaded, registration_sha256 = _read_model(
        control_store,
        _REGISTRATION,
        EvaluationRegistration,
    )
    if not isinstance(loaded, EvaluationRegistration):
        raise ControlStoreError("registration has an unexpected model type")
    registration = loaded
    if not _registration_matches_output_binding(registration, binding):
        raise ControlStoreError("registration changed its output binding")

    entries = _control_entries(
        control_store,
        registration.attempts,
        allow_pending_registration=False,
    )
    if validate_lifecycle:
        _validate_lifecycle_records(control_store, entries, registration)

    flags = (os.O_RDWR if lock_readwrite else os.O_RDONLY) | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        evaluation_lock_descriptor = os.open(
            _EVALUATION_LOCK,
            flags,
            dir_fd=control_store._fd,
        )
    except OSError as exc:
        raise ControlStoreError("registered Terminal-Bench evaluation lock is missing") from exc
    stack.callback(os.close, evaluation_lock_descriptor)
    SecureDirectory._validate_file_info(
        os.fstat(evaluation_lock_descriptor),
        _EVALUATION_LOCK,
        _DEFAULT_MAX_FILE_BYTES,
    )
    if not _registration_commit_matches(
        commit,
        binding=binding,
        binding_sha256=binding_sha256,
        registration_sha256=registration_sha256,
        evaluation_lock_descriptor=evaluation_lock_descriptor,
    ):
        raise ControlStoreError("registration commit changed its immutable binding")

    attempt_stores = _open_attempt_directories(
        stack,
        control_store,
        registration.attempts,
    )
    topology = _CommittedTopology(
        registration=registration,
        anchor_store=anchor_store,
        output_store=output_store,
        control_store=control_store,
        attempt_stores=attempt_stores,
        evaluation_lock_descriptor=evaluation_lock_descriptor,
    )
    _verify_committed_topology(topology, output=output, evaluation_id=evaluation_id)
    return topology


def _verify_committed_topology(
    topology: _CommittedTopology,
    *,
    output: Path,
    evaluation_id: str,
) -> None:
    control = terminal_control_root(output, evaluation_id)
    _verify_directory_path(
        topology.anchor_store,
        output.parent / _INITIALIZATION_ANCHOR_DIRECTORY,
        label="initialization anchor",
    )
    _verify_directory_path(
        topology.output_store,
        output,
        label="Terminal-Bench output root",
    )
    _verify_directory_path(
        topology.control_store,
        control,
        label="Terminal-Bench control root",
    )
    for attempt_id, attempt_store in topology.attempt_stores.items():
        _verify_directory_path(
            attempt_store,
            control / attempt_id,
            label=f"Terminal-Bench attempt directory {attempt_id}",
        )
    _verify_file_path(
        topology.control_store,
        _EVALUATION_LOCK,
        topology.evaluation_lock_descriptor,
        label="Terminal-Bench evaluation lock",
    )
    for attempt_id, attempt_store in topology.attempt_stores.items():
        _verify_directory_path(
            attempt_store,
            control / attempt_id,
            label=f"Terminal-Bench attempt directory {attempt_id}",
        )
    _verify_directory_path(
        topology.output_store,
        output,
        label="Terminal-Bench output root",
    )
    _verify_directory_path(
        topology.control_store,
        control,
        label="Terminal-Bench control root",
    )


def open_attempt_store(
    output_root: str | Path,
    evaluation_id: str,
    attempt_id: str,
) -> SecureDirectory:
    if _HEX_64_RE.fullmatch(attempt_id) is None:
        raise ValueError("attempt_id must be lowercase SHA-256 hex")
    output = Path(output_root)
    if not output.is_absolute() or output != output.resolve():
        raise ValueError("output_root must be an absolute normalized path")
    with ExitStack() as stack:
        topology = _open_committed_topology(
            stack,
            output=output,
            evaluation_id=evaluation_id,
            lock_readwrite=False,
            validate_lifecycle=True,
        )
        if attempt_id not in topology.attempt_stores:
            raise ControlStoreError("attempt is not present in the committed registration")
        _verify_committed_topology(
            topology,
            output=output,
            evaluation_id=evaluation_id,
        )
        descriptor = os.dup(topology.attempt_stores[attempt_id]._fd)
    return SecureDirectory._from_open_fd(
        terminal_control_root(output, evaluation_id) / attempt_id,
        descriptor,
        expected_mode=_DIRECTORY_MODE,
    )


def open_control_store(
    output_root: str | Path,
    evaluation_id: str,
) -> SecureDirectory:
    """Open the committed control root after validating its full topology."""
    output = Path(output_root)
    if not output.is_absolute() or output != output.resolve():
        raise ValueError("output_root must be an absolute normalized path")
    with ExitStack() as stack:
        topology = _open_committed_topology(
            stack,
            output=output,
            evaluation_id=evaluation_id,
            lock_readwrite=False,
            validate_lifecycle=True,
        )
        _verify_committed_topology(
            topology,
            output=output,
            evaluation_id=evaluation_id,
        )
        descriptor = os.dup(topology.control_store._fd)
    return SecureDirectory._from_open_fd(
        terminal_control_root(output, evaluation_id),
        descriptor,
        expected_mode=_DIRECTORY_MODE,
    )


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

    output = Path(output_root)
    if not output.is_absolute() or output != output.resolve():
        raise ValueError("output_root must be an absolute normalized path")
    with ExitStack() as stack:
        topology = _open_committed_topology(
            stack,
            output=output,
            evaluation_id=evaluation_id,
            lock_readwrite=True,
            validate_lifecycle=True,
        )
        descriptor = topology.evaluation_lock_descriptor
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                raise ControlStoreError("formal evaluation is already active") from exc
            _verify_committed_topology(
                topology,
                output=output,
                evaluation_id=evaluation_id,
            )
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(descriptor)
            yield descriptor
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass


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
