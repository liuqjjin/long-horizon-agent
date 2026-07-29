"""Durable ownership records for commands that can outlive the harness.

The normal checkpoint says which workflow step is active, but it cannot reap a
process after the harness itself is killed.  An operation lease records the
kernel or daemon identity needed for that cleanup before target code is allowed
to run.

Local commands use a small launcher.  The parent first writes ``PREPARING``,
then the launcher writes its PID/PGID and ``ACTIVE`` before waiting for a
one-byte release from the parent.  Docker commands can write ``ACTIVE`` before
starting because their container name and identity label are deterministic.
Each concurrent command gets its own file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .clock import now
from .sandbox.base import (
    ProcessCleanupResult,
    read_process_group_census,
)

LEASE_DIRECTORY = "active-operations"
_SCHEMA_VERSION = 1
_OPERATION_ID = re.compile(r"[0-9a-f]{32}")
_KERNEL_BOOT_TIME = re.compile(
    r"\bsec\s*=\s*(?P<sec>[0-9]+)\s*,\s*usec\s*=\s*(?P<usec>[0-9]+)\b"
)
_ACTIVATION_TIMEOUT_S = 5.0
_RECOVERY_CONFIRMATION_S = 2.0
_IDENTITY_RETRY_S = 0.5
_IDENTITY_RETRY_INTERVAL_S = 0.01


class OperationLeaseError(RuntimeError):
    """An operation ownership record is missing, corrupt, or contradictory."""


class ActiveOperationLease(BaseModel):
    """Identity needed to stop one command after its original parent dies."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    operation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    backend: Literal["trusted-local", "docker"]
    phase: Literal["PREPARING", "ACTIVE"]
    created_at: datetime
    cwd_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    boot_identity: str
    pid: int | None = Field(default=None, ge=1)
    pgid: int | None = Field(default=None, ge=1)
    process_identity: str | None = None
    container_name: str | None = None
    container_identity: str | None = None
    container_id: str | None = None

    @field_validator("created_at")
    @classmethod
    def _created_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _identity_matches_backend(self) -> ActiveOperationLease:
        if self.backend == "trusted-local":
            if self.container_name is not None or self.container_identity is not None:
                raise ValueError("local operation cannot carry container identity")
            if self.phase == "PREPARING":
                if any(
                    value is not None
                    for value in (self.pid, self.pgid, self.process_identity)
                ):
                    raise ValueError("preparing local operation cannot carry a PID")
            elif None in (self.pid, self.pgid, self.process_identity):
                raise ValueError("active local operation requires PID/PGID identity")
        else:
            if any(
                value is not None
                for value in (self.pid, self.pgid, self.process_identity)
            ):
                raise ValueError("Docker operation cannot carry process identity")
            if self.phase != "ACTIVE":
                raise ValueError("Docker operation identity must be active before spawn")
            if self.container_name is None or self.container_identity is None:
                raise ValueError("Docker operation requires a name and identity label")
        return self


@dataclass(frozen=True)
class OperationRecoveryResult:
    """Typed result used by resume to decide whether mutation may continue."""

    confirmed: bool
    recovered_operation_ids: tuple[str, ...] = ()
    quarantined_operation_ids: tuple[str, ...] = ()
    detail: str = ""

    @property
    def requires_quarantine(self) -> bool:
        return not self.confirmed


@dataclass(frozen=True)
class _ProcessSnapshot:
    """Process birth marker and group read from one kernel-backed snapshot."""

    birth_identity: str
    pgid: int


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _command_sha256(command: list[str]) -> str:
    return hashlib.sha256(_canonical(command)).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OperationLeaseError(
            f"operation lease directory could not be opened durably: {path}"
        ) from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise OperationLeaseError(
            f"operation lease directory could not be synced durably: {path}"
        ) from error
    finally:
        os.close(descriptor)


def _linux_boot_identity() -> str | None:
    linux_boot_id = Path("/proc/sys/kernel/random/boot_id")
    try:
        value = linux_boot_id.read_text().strip()
    except OSError:
        return None
    if re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        value,
    ) is None:
        return None
    return f"linux:{value.lower()}"


def _kernel_boot_time_identity() -> str | None:
    """Read macOS/BSD boot time from a fixed system sysctl executable."""
    for candidate in (Path("/usr/sbin/sysctl"), Path("/sbin/sysctl")):
        try:
            metadata = candidate.stat()
        except OSError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or not os.access(candidate, os.X_OK)
        ):
            continue
        try:
            result = subprocess.run(
                [str(candidate), "-n", "kern.boottime"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
                env={"PATH": "/usr/sbin:/sbin:/usr/bin:/bin", "LANG": "C"},
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if (
            result.returncode != 0
            or len(result.stdout.encode("utf-8", errors="replace")) > 4096
        ):
            continue
        match = _KERNEL_BOOT_TIME.search(result.stdout)
        if match is None:
            continue
        seconds = int(match.group("sec"))
        microseconds = int(match.group("usec"))
        if seconds <= 0 or not 0 <= microseconds < 1_000_000:
            continue
        return f"kern-boottime:{sys.platform}:{seconds}:{microseconds}"
    return None


def _boot_identity() -> str | None:
    """Return a kernel-backed identity for the current host boot."""
    linux = _linux_boot_identity()
    if linux is not None:
        return linux
    if sys.platform == "darwin" or "bsd" in sys.platform:
        return _kernel_boot_time_identity()
    return None


def _boot_identity_scheme(value: str) -> str | None:
    if value.startswith("linux:"):
        return "linux"
    match = re.fullmatch(
        r"kern-boottime:(?P<platform>[^:]+):[0-9]+:[0-9]+",
        value,
    )
    if match is not None:
        return f"kern-boottime:{match.group('platform')}"
    return None


def _linux_process_snapshot(pid: int) -> _ProcessSnapshot | None:
    """Read Linux start ticks and process group from one /proc record."""
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        record = stat_path.read_text()
    except OSError:
        return None
    # The parenthesized comm field may contain spaces and parentheses. Linux
    # places the remaining fixed-position fields after its final right paren.
    close = record.rfind(")")
    if close < 0:
        return None
    fields = record[close + 1 :].split()
    if len(fields) < 20:
        return None
    try:
        pgid = int(fields[2])
        start_ticks = int(fields[19])
    except (TypeError, ValueError):
        return None
    if pgid <= 0 or start_ticks <= 0:
        return None
    return _ProcessSnapshot(
        birth_identity=f"proc-start:{start_ticks}",
        pgid=pgid,
    )


def _darwin_process_snapshot(pid: int) -> _ProcessSnapshot | None:
    """Read microsecond process birth time and PGID through macOS libproc."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        class ProcBSDInfo(ctypes.Structure):
            _fields_ = [
                ("pbi_flags", ctypes.c_uint32),
                ("pbi_status", ctypes.c_uint32),
                ("pbi_xstatus", ctypes.c_uint32),
                ("pbi_pid", ctypes.c_uint32),
                ("pbi_ppid", ctypes.c_uint32),
                ("pbi_uid", ctypes.c_uint32),
                ("pbi_gid", ctypes.c_uint32),
                ("pbi_ruid", ctypes.c_uint32),
                ("pbi_rgid", ctypes.c_uint32),
                ("pbi_svuid", ctypes.c_uint32),
                ("pbi_svgid", ctypes.c_uint32),
                ("rfu_1", ctypes.c_uint32),
                ("pbi_comm", ctypes.c_char * 16),
                ("pbi_name", ctypes.c_char * 32),
                ("pbi_nfiles", ctypes.c_uint32),
                ("pbi_pgid", ctypes.c_uint32),
                ("pbi_pjobc", ctypes.c_uint32),
                ("e_tdev", ctypes.c_uint32),
                ("e_tpgid", ctypes.c_uint32),
                ("pbi_nice", ctypes.c_int32),
                ("pbi_start_tvsec", ctypes.c_uint64),
                ("pbi_start_tvusec", ctypes.c_uint64),
            ]

        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = library.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        info = ProcBSDInfo()
        written = proc_pidinfo(
            pid,
            3,  # PROC_PIDTBSDINFO
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
    except (AttributeError, OSError, ValueError):
        return None
    if (
        written != ctypes.sizeof(info)
        or info.pbi_pid != pid
        or info.pbi_pgid <= 0
        or info.pbi_start_tvsec <= 0
        or not 0 <= info.pbi_start_tvusec < 1_000_000
    ):
        return None
    return _ProcessSnapshot(
        birth_identity=(
            f"darwin-start:{info.pbi_start_tvsec}:"
            f"{info.pbi_start_tvusec:06d}"
        ),
        pgid=int(info.pbi_pgid),
    )


def _bsd_process_snapshot(pid: int) -> _ProcessSnapshot | None:
    """Best available fixed-command snapshot for non-Darwin BSD hosts."""
    ps = Path("/bin/ps")
    if not ps.is_file():
        ps = Path("/usr/bin/ps")
    if not ps.is_file():
        return None
    try:
        result = subprocess.run(
            [str(ps), "-p", str(pid), "-o", "lstart=", "-o", "pgid="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            env={"PATH": "/usr/bin:/bin", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        return None
    try:
        marker, raw_pgid = output.rsplit(maxsplit=1)
        pgid = int(raw_pgid)
    except (TypeError, ValueError):
        return None
    if not marker or pgid <= 0:
        return None
    return _ProcessSnapshot(
        birth_identity=f"ps-start:{marker}",
        pgid=pgid,
    )


def _process_snapshot(pid: int) -> _ProcessSnapshot | None:
    linux = _linux_process_snapshot(pid)
    if linux is not None:
        return linux
    if sys.platform == "darwin":
        return _darwin_process_snapshot(pid)
    if "bsd" in sys.platform:
        return _bsd_process_snapshot(pid)
    return None


class OperationLeaseStore:
    """Checksummed, atomically replaced operation records below one run dir."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()
        self.directory = self.run_dir / LEASE_DIRECTORY

    def _path(self, operation_id: str) -> Path:
        if _OPERATION_ID.fullmatch(operation_id) is None:
            raise OperationLeaseError(f"invalid operation id: {operation_id!r}")
        return self.directory / f"{operation_id}.json"

    def _write(self, lease: ActiveOperationLease, *, replace: bool) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise OperationLeaseError(
                f"operation lease directory is unsafe: {self.directory}"
            )
        # The directory entry itself must be durable before a target can be
        # released. Syncing only the child directory does not persist its
        # creation in the run directory after a crash.
        _fsync_directory(self.run_dir)
        path = self._path(lease.operation_id)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise OperationLeaseError(f"operation lease path is unsafe: {path}")
        if not replace and path.exists():
            raise OperationLeaseError(
                f"operation lease already exists: {lease.operation_id}"
            )
        payload = lease.model_dump(mode="json")
        envelope = {
            "schema_version": _SCHEMA_VERSION,
            "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
            "payload": payload,
        }
        temporary = self.directory / f".{lease.operation_id}.{uuid.uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            data = json.dumps(envelope, indent=2).encode()
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(self.directory)
        finally:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def prepare_local(
        self,
        command: list[str],
        *,
        cwd: str | Path,
        operation_id: str | None = None,
    ) -> ActiveOperationLease:
        operation_id = operation_id or uuid.uuid4().hex
        boot_identity = _boot_identity()
        if boot_identity is None:
            raise OperationLeaseError(
                "kernel boot identity is unavailable; refusing durable operation"
            )
        lease = ActiveOperationLease(
            operation_id=operation_id,
            backend="trusted-local",
            phase="PREPARING",
            created_at=now(),
            cwd_sha256=_sha256_text(str(Path(cwd).resolve())),
            command_sha256=_command_sha256(command),
            boot_identity=boot_identity,
        )
        self._write(lease, replace=False)
        return lease

    def activate_local(
        self,
        operation_id: str,
        *,
        pid: int,
        pgid: int,
    ) -> ActiveOperationLease:
        lease = self.load(operation_id)
        if lease.backend != "trusted-local" or lease.phase != "PREPARING":
            raise OperationLeaseError(
                f"local operation {operation_id} is not preparing"
            )
        snapshot = _process_snapshot(pid)
        if snapshot is None:
            raise OperationLeaseError(
                f"could not read process identity for operation {operation_id}"
            )
        if snapshot.pgid != pgid:
            raise OperationLeaseError(
                f"process group changed while activating operation {operation_id}"
            )
        active = lease.model_copy(
            update={
                "phase": "ACTIVE",
                "pid": pid,
                "pgid": pgid,
                "process_identity": snapshot.birth_identity,
            }
        )
        self._write(active, replace=True)
        return active

    def activate_docker(
        self,
        command: list[str],
        *,
        cwd: str | Path,
        operation_id: str | None = None,
    ) -> ActiveOperationLease:
        operation_id = operation_id or uuid.uuid4().hex
        boot_identity = _boot_identity()
        if boot_identity is None:
            raise OperationLeaseError(
                "kernel boot identity is unavailable; refusing durable operation"
            )
        lease = ActiveOperationLease(
            operation_id=operation_id,
            backend="docker",
            phase="ACTIVE",
            created_at=now(),
            cwd_sha256=_sha256_text(str(Path(cwd).resolve())),
            command_sha256=_command_sha256(command),
            boot_identity=boot_identity,
            container_name=f"lha-{operation_id}",
            container_identity=operation_id,
        )
        self._write(lease, replace=False)
        return lease

    def bind_container_id(
        self,
        operation_id: str,
        container_id: str,
    ) -> ActiveOperationLease:
        lease = self.load(operation_id)
        if lease.backend != "docker" or lease.phase != "ACTIVE":
            raise OperationLeaseError(
                f"Docker operation {operation_id} is not active"
            )
        if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
            raise OperationLeaseError("Docker returned an invalid container id")
        active = lease.model_copy(update={"container_id": container_id})
        self._write(active, replace=True)
        return active

    def load(self, operation_id: str) -> ActiveOperationLease:
        path = self._path(operation_id)
        if path.is_symlink() or not path.is_file():
            raise OperationLeaseError(f"operation lease is missing or unsafe: {path}")
        try:
            raw = json.loads(path.read_text())
            payload = raw["payload"]
            digest = hashlib.sha256(_canonical(payload)).hexdigest()
            if (
                raw.get("schema_version") != _SCHEMA_VERSION
                or raw.get("sha256") != digest
            ):
                raise OperationLeaseError(
                    f"operation lease failed its integrity check: {path}"
                )
            return ActiveOperationLease.model_validate(payload)
        except OperationLeaseError:
            raise
        except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise OperationLeaseError(
                f"operation lease is unreadable: {path}: {error}"
            ) from error

    def list(self) -> list[ActiveOperationLease]:
        if not self.directory.exists():
            return []
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise OperationLeaseError(
                f"operation lease directory is unsafe: {self.directory}"
            )
        leases: list[ActiveOperationLease] = []
        for path in sorted(self.directory.iterdir()):
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                # A crash before replace cannot authorize target execution.
                continue
            match = re.fullmatch(r"([0-9a-f]{32})\.json", path.name)
            if match is None:
                raise OperationLeaseError(
                    f"unexpected entry in operation lease directory: {path.name}"
                )
            leases.append(self.load(match.group(1)))
        return leases

    def clear(self, operation_id: str) -> None:
        path = self._path(operation_id)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise OperationLeaseError(f"operation lease path is unsafe: {path}")
        try:
            path.unlink()
        except FileNotFoundError:
            return
        _fsync_directory(self.directory)


def operation_store_for_workdir(cwd: str | Path) -> OperationLeaseStore | None:
    """Find the enclosing run without trusting arbitrary target directories."""
    workdir = Path(cwd).resolve()
    # Verification may run in a disposable copy below the run directory rather
    # than in the canonical ``workdir``.  The managed sibling plus checkpoint
    # distinguish that layout from a repository that merely contains a file
    # named ``state.json``.
    for run_dir in workdir.parents:
        state = run_dir / "state.json"
        managed_workdir = run_dir / "workdir"
        if (
            not run_dir.is_symlink()
            and not state.is_symlink()
            and state.is_file()
            and not managed_workdir.is_symlink()
            and managed_workdir.is_dir()
        ):
            return OperationLeaseStore(run_dir)
    return None


def build_local_launcher(
    store: OperationLeaseStore,
    lease: ActiveOperationLease,
    command: list[str],
    *,
    release_fd: int,
) -> list[str]:
    if lease.backend != "trusted-local" or lease.phase != "PREPARING":
        raise OperationLeaseError("local launcher requires a preparing lease")
    return [
        sys.executable,
        "-I",
        "-m",
        "lha.operation_lease",
        "--lease-run-dir",
        str(store.run_dir),
        "--operation-id",
        lease.operation_id,
        "--release-fd",
        str(release_fd),
        "--",
        *command,
    ]


def wait_until_local_active(
    store: OperationLeaseStore,
    operation_id: str,
    *,
    pid: int,
    timeout_s: float = _ACTIVATION_TIMEOUT_S,
) -> ActiveOperationLease:
    deadline = time.monotonic() + timeout_s
    while True:
        lease = store.load(operation_id)
        if lease.phase == "ACTIVE":
            if lease.pid != pid or lease.pgid != pid:
                raise OperationLeaseError(
                    f"launcher identity mismatch for operation {operation_id}"
                )
            return lease
        if time.monotonic() >= deadline:
            raise OperationLeaseError(
                f"launcher did not activate operation {operation_id}"
            )
        time.sleep(0.01)


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_local_operation(
    store: OperationLeaseStore,
    lease: ActiveOperationLease,
    *,
    confirmation_timeout_s: float = _RECOVERY_CONFIRMATION_S,
) -> ProcessCleanupResult:
    """Reap one orphaned local process group and independently confirm absence."""
    if lease.backend != "trusted-local":
        return ProcessCleanupResult(False, "operation is not a local process")
    current_boot_identity = _boot_identity()
    if current_boot_identity is None:
        return ProcessCleanupResult(
            False,
            "kernel boot identity is unavailable; operation remains quarantined",
        )
    lease_boot_scheme = _boot_identity_scheme(lease.boot_identity)
    current_boot_scheme = _boot_identity_scheme(current_boot_identity)
    if (
        lease_boot_scheme is None
        or current_boot_scheme is None
        or lease_boot_scheme != current_boot_scheme
    ):
        return ProcessCleanupResult(
            False,
            "operation boot identity cannot be compared safely",
        )
    if lease.boot_identity != current_boot_identity:
        store.clear(lease.operation_id)
        return ProcessCleanupResult(True, "operation belongs to an earlier host boot")
    if lease.phase == "PREPARING":
        deadline = time.monotonic() + min(confirmation_timeout_s, 0.5)
        while time.monotonic() < deadline:
            current = store.load(lease.operation_id)
            if current.phase == "ACTIVE":
                return recover_local_operation(
                    store,
                    current,
                    confirmation_timeout_s=confirmation_timeout_s,
                )
            time.sleep(0.01)
        return ProcessCleanupResult(
            False,
            (
                f"operation {lease.operation_id} remained PREPARING; "
                "launcher absence cannot be confirmed"
            ),
        )

    assert lease.pid is not None
    assert lease.pgid is not None
    identity_deadline = time.monotonic() + min(
        confirmation_timeout_s,
        _IDENTITY_RETRY_S,
    )
    while True:
        if not _group_exists(lease.pgid):
            store.clear(lease.operation_id)
            return ProcessCleanupResult(True, "process group is absent")
        census = read_process_group_census(lease.pgid)
        if census.error is not None:
            unavailable_detail = (
                f"process group {lease.pgid} could not be inspected: {census.error}"
            )
        elif not census.members:
            if not _group_exists(lease.pgid):
                store.clear(lease.operation_id)
                return ProcessCleanupResult(
                    True,
                    "process leader and process group are absent",
                )
            unavailable_detail = (
                f"process group {lease.pgid} is absent from the process table "
                "but the kernel still reports it present"
            )
        elif not census.runnable_members:
            store.clear(lease.operation_id)
            return ProcessCleanupResult(
                True,
                f"process group {lease.pgid} has only zombie members",
            )
        else:
            leader = next(
                (member for member in census.members if member.pid == lease.pid),
                None,
            )
            if leader is None:
                runnable_pids = ", ".join(
                    str(member.pid) for member in census.runnable_members[:10]
                )
                return ProcessCleanupResult(
                    False,
                    (
                        f"leased process leader PID {lease.pid} is absent while "
                        f"process group {lease.pgid} has runnable members: "
                        f"{runnable_pids}"
                    ),
                )
            current_snapshot = _process_snapshot(lease.pid)
            if current_snapshot is not None:
                if current_snapshot.pgid != lease.pgid:
                    return ProcessCleanupResult(
                        False,
                        (
                            f"PID {lease.pid} now belongs to process group "
                            f"{current_snapshot.pgid}, expected {lease.pgid}"
                        ),
                    )
                if current_snapshot.birth_identity != lease.process_identity:
                    return ProcessCleanupResult(
                        False,
                        (
                            f"PID {lease.pid} was reused while process group "
                            f"{lease.pgid} still exists"
                        ),
                    )
                break
            unavailable_detail = (
                f"PID {lease.pid} is a zombie with runnable descendants but "
                "its birth identity is unavailable"
                if leader.is_zombie
                else f"live PID identity is unavailable for PID {lease.pid} after retry"
            )
        if time.monotonic() >= identity_deadline:
            return ProcessCleanupResult(False, unavailable_detail)
        time.sleep(_IDENTITY_RETRY_INTERVAL_S)

    permission_error: PermissionError | None = None
    try:
        os.killpg(lease.pgid, signal.SIGKILL)
    except ProcessLookupError:
        store.clear(lease.operation_id)
        return ProcessCleanupResult(True, "process group is absent")
    except PermissionError as error:
        # Darwin can report EPERM after delivering SIGKILL to members that are
        # now zombies. The bounded census below decides whether anything can
        # still execute.
        permission_error = error
    except OSError as error:
        return ProcessCleanupResult(
            False,
            f"could not kill process group {lease.pgid}: {error}",
        )
    deadline = time.monotonic() + confirmation_timeout_s
    while True:
        if not _group_exists(lease.pgid):
            store.clear(lease.operation_id)
            return ProcessCleanupResult(True, "process group killed and confirmed absent")
        census = read_process_group_census(lease.pgid)
        if census.error is not None:
            cleanup_detail = (
                f"process group {lease.pgid} could not be inspected after SIGKILL: "
                f"{census.error}"
            )
        elif not census.members:
            if not _group_exists(lease.pgid):
                store.clear(lease.operation_id)
                return ProcessCleanupResult(
                    True,
                    "process group killed and confirmed absent",
                )
            cleanup_detail = (
                f"process group {lease.pgid} is absent from the process table "
                "but the kernel still reports it present after SIGKILL"
            )
        elif not census.runnable_members:
            store.clear(lease.operation_id)
            return ProcessCleanupResult(
                True,
                f"process group {lease.pgid} has only zombie members after SIGKILL",
            )
        else:
            runnable_pids = ", ".join(
                str(member.pid) for member in census.runnable_members[:10]
            )
            cleanup_detail = (
                f"process group {lease.pgid} still has runnable members after "
                f"SIGKILL: {runnable_pids}"
            )
        if time.monotonic() >= deadline:
            if permission_error is not None:
                cleanup_detail = (
                    f"could not kill process group {lease.pgid}: "
                    f"{permission_error}; {cleanup_detail}"
                )
            return ProcessCleanupResult(
                False,
                cleanup_detail,
            )
        time.sleep(_IDENTITY_RETRY_INTERVAL_S)


def recover_active_operations(
    run_dir: str | Path,
    *,
    docker_recover: (
        Callable[[OperationLeaseStore, ActiveOperationLease], ProcessCleanupResult]
        | None
    ) = None,
) -> OperationRecoveryResult:
    """Recover every persisted operation or return a typed quarantine result."""
    store = OperationLeaseStore(run_dir)
    try:
        leases = store.list()
    except OperationLeaseError as error:
        return OperationRecoveryResult(False, detail=str(error))
    recovered: list[str] = []
    quarantined: list[str] = []
    details: list[str] = []
    for lease in leases:
        try:
            if lease.backend == "trusted-local":
                result = recover_local_operation(store, lease)
            elif docker_recover is not None:
                result = docker_recover(store, lease)
            else:
                result = ProcessCleanupResult(
                    False,
                    (
                        f"Docker recovery is unavailable for operation "
                        f"{lease.operation_id}"
                    ),
                )
        except (OSError, OperationLeaseError) as error:
            result = ProcessCleanupResult(
                False,
                f"operation recovery failed closed: {error}",
            )
        details.append(f"{lease.operation_id}: {result.detail}")
        if result.confirmed:
            recovered.append(lease.operation_id)
        else:
            quarantined.append(lease.operation_id)
    return OperationRecoveryResult(
        not quarantined,
        tuple(recovered),
        tuple(quarantined),
        "; ".join(details),
    )


def _launcher(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lease-run-dir", required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--release-fd", required=True, type=int)
    namespace, command = parser.parse_known_args(argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        return 127
    store = OperationLeaseStore(namespace.lease_run_dir)
    try:
        lease = store.load(namespace.operation_id)
        if (
            lease.command_sha256 != _command_sha256(command)
            or lease.cwd_sha256 != _sha256_text(str(Path.cwd().resolve()))
        ):
            raise OperationLeaseError(
                "launcher command or working directory does not match its lease"
            )
        store.activate_local(
            namespace.operation_id,
            pid=os.getpid(),
            pgid=os.getpgrp(),
        )
        released = os.read(namespace.release_fd, 1)
    except (OSError, OperationLeaseError) as error:
        print(f"operation launcher failed: {error}", file=sys.stderr)
        return 126
    finally:
        try:
            os.close(namespace.release_fd)
        except OSError:
            pass
    if released != b"G":
        print("operation launcher was not released by its parent", file=sys.stderr)
        return 126
    try:
        os.execvpe(command[0], command, os.environ)
    except OSError as error:
        print(f"failed to execute {command[0]!r}: {error}", file=sys.stderr)
        return 127


if __name__ == "__main__":  # pragma: no cover - exercised by process tests
    raise SystemExit(_launcher(sys.argv[1:]))
