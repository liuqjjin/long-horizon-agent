"""Docker execution backend: network-less, resource-bounded containers.

The default for external/untrusted target repos. The container does not inherit
the host process environment, but image-defined variables remain. The working
directory and a restricted ``/tmp`` tmpfs are writable; the root filesystem and
declared read-only mounts are not. The network is off, CPU/memory/process counts
are capped, and a timeout force-removes the container when Docker responds.

The image must provide the tools the run needs — the default python:3.12-slim
carries none of pytest/pytest-json-report/ruff, so code-verification tasks need
a purpose-built image (see SECURITY.md, "Execution backends").
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..tools.shell import ProcResult, trusted_executable
from .base import (
    PROCESS_CLEANUP_RETURN_CODE,
    ExecutionBackend,
    ProcessCleanupResult,
    ResourceLimits,
    run_bounded_process,
    terminate_process_group,
)

_WORK = "/work"
_TMPFS = "/tmp:rw,nosuid,nodev,size=256m,mode=1777"
_REMOVE_TIMEOUT_S = 30.0
_PROVENANCE_TIMEOUT_S = 30.0
_PROVENANCE_OUTPUT_BYTES = 64 * 1024
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_OPERATION_LABEL = "lha.operation_id"
_SECURITY_RUN_ARGS = [
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--init",
]
_VERSION_PROBE = """
import importlib.metadata
import json
import platform

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "python": platform.python_version(),
    "pytest": package_version("pytest"),
    "ruff": package_version("ruff"),
}, sort_keys=True))
""".strip()
_MAX_DOCKER_EXECUTABLE_BYTES = 256 * 1024 * 1024
_EXECUTABLE_READ_BYTES = 1024 * 1024
_CONTAINER_ID_TIMEOUT_S = 5.0


class DockerContainerIdentityError(RuntimeError):
    """A daemon-side container could not be bound to its durable lease."""


class _DockerContainerIdentityPending(RuntimeError):
    """Docker created the cidfile but has not finished its bounded write."""


@dataclass(frozen=True)
class DockerExecutableIdentity:
    """Absolute Docker client bytes fixed for one runner lifetime."""

    path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str
    trusted_install: bool

    def as_provenance(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size,
            "trusted_install": self.trusted_install,
        }


def _docker_executable_metadata(path: Path, *, digest: bool) -> DockerExecutableIdentity:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_DOCKER_EXECUTABLE_BYTES
        ):
            raise RuntimeError("Docker executable is not a bounded regular file")
        hasher = hashlib.sha256()
        if digest:
            while True:
                chunk = os.read(descriptor, _EXECUTABLE_READ_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise RuntimeError("Docker executable changed while it was inspected")
    finally:
        os.close(descriptor)
    return DockerExecutableIdentity(
        path=str(path),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        sha256=hasher.hexdigest() if digest else "",
        trusted_install=False,
    )


def resolve_docker_executable(docker: str = "docker") -> DockerExecutableIdentity:
    """Resolve Docker once without trusting a model-controlled PATH or cwd."""
    configured = Path(docker)
    if configured.name == docker:
        resolved_text = trusted_executable(docker, require_unwritable=False)
        if resolved_text is None:
            raise RuntimeError("Docker executable was not found on a sanitized PATH")
        resolved = Path(resolved_text)
        trusted = trusted_executable(docker, require_unwritable=True) == str(resolved)
    else:
        if not configured.is_absolute():
            raise RuntimeError("Docker executable must be a basename or absolute path")
        try:
            resolved = configured.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RuntimeError("configured Docker executable is unavailable") from error
        trusted = trusted_executable(
            resolved.name,
            path="",
            extra_dirs=(resolved.parent,),
            require_unwritable=True,
        ) == str(resolved)
    identity = _docker_executable_metadata(resolved, digest=True)
    return DockerExecutableIdentity(
        path=identity.path,
        device=identity.device,
        inode=identity.inode,
        size=identity.size,
        mtime_ns=identity.mtime_ns,
        sha256=identity.sha256,
        trusted_install=trusted,
    )


def _docker_control(
    argv: list[str],
    *,
    output_bytes: int,
    timeout: float = _REMOVE_TIMEOUT_S,
) -> ProcResult:
    """Run one Docker control command without leaving its own process tree."""
    return run_bounded_process(
        argv,
        timeout=timeout,
        output_bytes=output_bytes,
        start_new_session=True,
        on_exit=terminate_process_group,
    )


def _container_absence_probe(
    docker: str,
    name: str,
    *,
    output_bytes: int,
) -> ProcessCleanupResult:
    """Confirm absence through a successful daemon query, not error wording."""
    argv = [
        docker,
        "container",
        "ls",
        "--all",
        "--quiet",
        "--no-trunc",
        "--filter",
        f"name=^/{name}$",
    ]
    try:
        probe = _docker_control(argv, output_bytes=output_bytes)
    except OSError as error:
        return ProcessCleanupResult(
            False,
            f"container absence probe could not start: {error}",
        )
    if probe.cleanup_unconfirmed:
        return ProcessCleanupResult(
            False,
            f"container absence probe cleanup was not confirmed: {probe.cleanup_detail}",
        )
    if probe.returncode != 0 or probe.output_truncated:
        detail = (probe.stderr or probe.stdout).strip()[-2000:]
        reason = (
            "container absence probe output was incomplete"
            if probe.output_truncated
            else f"container absence probe failed with exit code {probe.returncode}"
        )
        return ProcessCleanupResult(
            False,
            f"{reason}{f': {detail}' if detail else ''}",
        )
    if probe.stdout.strip():
        return ProcessCleanupResult(
            False,
            f"container {name} still exists after forced removal",
        )
    return ProcessCleanupResult(True, f"container {name} is absent")


def _container_id_absence_probe(
    docker: str,
    container_id: str,
    *,
    output_bytes: int,
) -> ProcessCleanupResult:
    """Confirm the leased immutable container ID is absent."""
    if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        return ProcessCleanupResult(False, "container ID is not a full digest")
    argv = [
        docker,
        "container",
        "ls",
        "--all",
        "--quiet",
        "--no-trunc",
        "--filter",
        f"id={container_id}",
    ]
    try:
        probe = _docker_control(argv, output_bytes=output_bytes)
    except OSError as error:
        return ProcessCleanupResult(
            False,
            f"container ID absence probe could not start: {error}",
        )
    if probe.cleanup_unconfirmed:
        return ProcessCleanupResult(
            False,
            (f"container ID absence probe cleanup was not confirmed: {probe.cleanup_detail}"),
        )
    if probe.returncode != 0 or probe.output_truncated:
        detail = (probe.stderr or probe.stdout).strip()[-2000:]
        reason = (
            "container ID absence probe output was incomplete"
            if probe.output_truncated
            else f"container ID absence probe failed with exit code {probe.returncode}"
        )
        return ProcessCleanupResult(
            False,
            f"{reason}{f': {detail}' if detail else ''}",
        )
    observed = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not observed:
        return ProcessCleanupResult(True, f"container {container_id} is absent")
    if observed == [container_id]:
        return ProcessCleanupResult(
            False,
            f"container {container_id} still exists after forced removal",
        )
    return ProcessCleanupResult(
        False,
        "container ID absence probe returned an unexpected identity",
    )


def _remove_container(
    docker: str,
    name: str,
    *,
    output_bytes: int,
    container_id: str | None = None,
) -> ProcessCleanupResult:
    """Force-remove a named container and independently confirm ambiguous exits."""
    reference = container_id or name
    argv = [docker, "rm", "-f", reference]
    try:
        cleanup = _docker_control(argv, output_bytes=output_bytes)
    except OSError as error:
        return ProcessCleanupResult(
            False,
            (f"container cleanup could not start; {reference} may still be running: {error}"),
        )

    if cleanup.returncode == 0 and not cleanup.output_truncated:
        return ProcessCleanupResult(True, f"container {reference} removed")
    if cleanup.cleanup_unconfirmed:
        return ProcessCleanupResult(
            False,
            f"container cleanup process was not confirmed stopped: {cleanup.cleanup_detail}",
        )

    removal_detail = (cleanup.stderr or cleanup.stdout).strip()[-2000:]
    if cleanup.returncode == 124:
        detail = (cleanup.stderr or cleanup.stdout).strip()[-2000:]
        suffix = f": {detail}" if detail else ""
        removal_detail = (
            f"container cleanup timed out after {_REMOVE_TIMEOUT_S:g}s; "
            f"{reference} may still be running{suffix}"
        )
    elif cleanup.output_truncated:
        removal_detail = (
            f"container cleanup output was incomplete; {reference} may still be running"
        )
    else:
        suffix = f": {removal_detail}" if removal_detail else ""
        removal_detail = (
            f"container cleanup failed with exit code {cleanup.returncode}; "
            f"{reference} may still be running{suffix}"
        )

    absence = (
        _container_id_absence_probe(
            docker,
            container_id,
            output_bytes=output_bytes,
        )
        if container_id is not None
        else _container_absence_probe(
            docker,
            name,
            output_bytes=output_bytes,
        )
    )
    return ProcessCleanupResult(
        absence.confirmed,
        "; ".join(detail for detail in (removal_detail, absence.detail) if detail),
    )


def _container_id_path(store, operation_id: str) -> Path:
    return store.run_dir / "active-container-ids" / f"{operation_id}.cid"


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _discard_cidfile(store, operation_id: str) -> ProcessCleanupResult:
    """Remove a Docker-created cidfile and persist the directory update."""
    path = _container_id_path(store, operation_id)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return ProcessCleanupResult(True, "Docker cidfile is absent")
    except OSError as error:
        return ProcessCleanupResult(
            False,
            f"Docker cidfile could not be inspected: {error}",
        )
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        return ProcessCleanupResult(
            False,
            f"Docker cidfile path is unsafe: {path}",
        )
    try:
        path.unlink()
        _fsync_directory(path.parent)
    except OSError as error:
        return ProcessCleanupResult(
            False,
            f"Docker cidfile could not be removed durably: {error}",
        )
    return ProcessCleanupResult(True, "Docker cidfile removed")


def _clear_docker_operation(store, lease) -> ProcessCleanupResult:
    """Clear auxiliary identity first, then the authoritative operation lease."""
    from ..operation_lease import OperationLeaseError

    cidfile = _discard_cidfile(store, lease.operation_id)
    if not cidfile.confirmed:
        return cidfile
    try:
        store.clear(lease.operation_id)
    except (OSError, OperationLeaseError) as error:
        return ProcessCleanupResult(
            False,
            f"container is absent but lease cleanup failed: {error}",
        )
    return ProcessCleanupResult(True, "Docker operation evidence cleared")


def _read_container_id(path: Path) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise DockerContainerIdentityError("Docker cidfile is not a bounded standalone file")
        if before.st_size < 64:
            raise _DockerContainerIdentityPending("Docker cidfile write is not complete")
        if before.st_size > 65:
            raise DockerContainerIdentityError("Docker cidfile is not a bounded standalone file")
        payload = os.read(descriptor, 66)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise _DockerContainerIdentityPending("Docker cidfile changed while it was read")
    finally:
        os.close(descriptor)
    if len(payload) == 65:
        if payload[-1:] != b"\n":
            raise DockerContainerIdentityError("Docker cidfile contains an invalid container ID")
        payload = payload[:64]
    try:
        value = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise DockerContainerIdentityError(
            "Docker cidfile contains an invalid container ID"
        ) from error
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DockerContainerIdentityError("Docker cidfile contains an invalid container ID")
    return value


def _cleanup_failure_result(
    result: ProcResult,
    cleanup: ProcessCleanupResult,
    *,
    output_bytes: int,
) -> ProcResult:
    """Replace an ordinary command status when the container may still write."""
    detail = f"process cleanup could not be confirmed: {cleanup.detail}"
    separator = "\n" if result.stderr and not result.stderr.endswith("\n") else ""
    suffix = f"{separator}{detail}".encode(errors="replace")
    if len(suffix) >= output_bytes:
        stderr = suffix[-output_bytes:].decode(errors="replace")
    else:
        prefix = result.stderr.encode(errors="replace")[: output_bytes - len(suffix)].decode(
            errors="replace"
        )
        stderr = f"{prefix}{suffix.decode(errors='replace')}"
    return ProcResult(
        PROCESS_CLEANUP_RETURN_CODE,
        result.stdout,
        stderr,
        result.duration_s,
        output_truncated=result.output_truncated,
        cleanup_confirmed=False,
        cleanup_detail=cleanup.detail,
    )


def _recover_docker_operation(docker: str, output_bytes: int, store, lease):
    """Validate a leased container label before force-removing it."""
    name = lease.container_name
    identity = lease.container_identity
    if not name or not identity:
        return ProcessCleanupResult(False, "Docker lease identity is incomplete")
    container_id = lease.container_id
    cidfile = _container_id_path(store, lease.operation_id)
    if container_id is None:
        try:
            container_id = _read_container_id(cidfile)
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError, DockerContainerIdentityError) as error:
            return ProcessCleanupResult(
                False,
                f"Docker cidfile identity is invalid: {error}",
            )
    if container_id is not None and re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        return ProcessCleanupResult(
            False,
            "Docker lease does not contain a full container ID",
        )
    absence = (
        _container_id_absence_probe(
            docker,
            container_id,
            output_bytes=output_bytes,
        )
        if container_id is not None
        else _container_absence_probe(
            docker,
            name,
            output_bytes=output_bytes,
        )
    )
    if absence.confirmed:
        cleared = _clear_docker_operation(store, lease)
        return ProcessCleanupResult(
            cleared.confirmed,
            "; ".join(detail for detail in (absence.detail, cleared.detail) if detail),
        )

    try:
        inspected = _docker_control(
            [
                docker,
                "container",
                "inspect",
                "--format",
                "{{json .}}",
                container_id or name,
            ],
            output_bytes=output_bytes,
        )
    except OSError as error:
        return ProcessCleanupResult(
            False,
            f"container identity probe could not start: {error}",
        )
    if inspected.cleanup_unconfirmed or inspected.output_truncated or inspected.returncode != 0:
        detail = (inspected.stderr or inspected.stdout).strip()[-2000:]
        return ProcessCleanupResult(
            False,
            (f"container identity could not be confirmed{f': {detail}' if detail else ''}"),
        )
    try:
        payload = json.loads(inspected.stdout)
        labels = payload["Config"]["Labels"] or {}
        inspected_id = payload["Id"]
        inspected_name = payload["Name"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        return ProcessCleanupResult(
            False,
            f"container identity response was invalid: {error}",
        )
    if (
        not isinstance(labels, dict)
        or labels.get(_OPERATION_LABEL) != identity
        or not isinstance(inspected_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", inspected_id) is None
        or not isinstance(inspected_name, str)
    ):
        return ProcessCleanupResult(
            False,
            f"container {name} does not match its operation lease",
        )
    if container_id is not None and container_id != inspected_id:
        return ProcessCleanupResult(
            False,
            f"container {name} ID does not match its operation lease",
        )
    if container_id is None and inspected_name != f"/{name}":
        return ProcessCleanupResult(
            False,
            f"container {name} name changed before its ID was bound",
        )
    cleanup = _remove_container(
        docker,
        name,
        output_bytes=output_bytes,
        container_id=inspected_id,
    )
    if not cleanup.confirmed:
        return cleanup
    cleared = _clear_docker_operation(store, lease)
    return ProcessCleanupResult(
        cleared.confirmed,
        "; ".join(detail for detail in (cleanup.detail, cleared.detail) if detail),
    )


def _remove_owned_container(
    docker: str,
    name: str,
    *,
    output_bytes: int,
    store=None,
    lease=None,
) -> ProcessCleanupResult:
    """Remove only a container proven to belong to the current lease."""
    if store is not None and lease is not None:
        return _recover_docker_operation(
            docker,
            output_bytes,
            store,
            lease,
        )
    return _remove_container(docker, name, output_bytes=output_bytes)


class DockerBackend(ExecutionBackend):
    name = "docker"

    def __init__(
        self,
        image: str = "python:3.12-slim",
        limits: ResourceLimits | None = None,
        ro_mounts: dict[str, str] | None = None,
        docker: str = "docker",
        operation_lease_dir: str | Path | None = None,
    ):
        self.image = image
        self.limits = limits or ResourceLimits(memory_mb=4096, pids=512)
        # host path -> container path, mounted read-only
        self.ro_mounts = dict(ro_mounts or {})
        self.docker = docker
        self._docker_identity: DockerExecutableIdentity | None = None
        self.operation_lease_dir = (
            Path(operation_lease_dir).resolve() if operation_lease_dir is not None else None
        )

    def bind_control_plane(self, *, verify_digest: bool = False) -> dict[str, object]:
        """Resolve Docker to fixed bytes and recheck that identity before use."""
        if self._docker_identity is None:
            self._docker_identity = resolve_docker_executable(self.docker)
            self.docker = self._docker_identity.path
        measured = _docker_executable_metadata(
            Path(self._docker_identity.path),
            digest=verify_digest,
        )
        expected = self._docker_identity
        if (
            measured.device,
            measured.inode,
            measured.size,
            measured.mtime_ns,
        ) != (
            expected.device,
            expected.inode,
            expected.size,
            expected.mtime_ns,
        ):
            raise RuntimeError("Docker executable changed after its identity was recorded")
        if verify_digest and measured.sha256 != expected.sha256:
            raise RuntimeError("Docker executable bytes changed after their digest was recorded")
        return expected.as_provenance()

    @classmethod
    def available(cls, docker: str = "docker") -> bool:
        try:
            identity = resolve_docker_executable(docker)
        except (OSError, RuntimeError, ValueError):
            return False
        try:
            probe = _docker_control(
                [identity.path, "info"],
                timeout=10,
                output_bytes=_PROVENANCE_OUTPUT_BYTES,
            )
        except OSError:
            return False
        return probe.returncode == 0 and not probe.output_truncated

    def python(self) -> str:
        return "python"

    def tool(self, name: str) -> str:
        return name

    def recover_active_operations(self, run_dir: str | Path):
        """Reap durable local/container operations before a run resumes."""
        from ..operation_lease import recover_active_operations

        self.bind_control_plane(verify_digest=True)
        return recover_active_operations(
            run_dir,
            docker_recover=lambda store, lease: _recover_docker_operation(
                self.docker,
                self.limits.output_bytes,
                store,
                lease,
            ),
        )

    def provenance(self) -> dict[str, object]:
        """Inspect immutable image identity and tool versions without target mounts.

        Provenance is diagnostic evidence, not a verifier gate. Every failure is
        returned as an explicit unavailable status instead of raising into the
        task being verified.
        """
        docker_executable: dict[str, object] | None = None
        docker_executable_reason: str | None = None
        try:
            docker_executable = self.bind_control_plane(verify_digest=True)
        except (OSError, RuntimeError, ValueError) as error:
            docker_executable_reason = type(error).__name__

        image_id: str | None = None
        image_id_reason: str | None = None
        try:
            if docker_executable is None:
                raise OSError("Docker executable identity is unavailable")
            inspected = _docker_control(
                [
                    self.docker,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    self.image,
                ],
                timeout=_PROVENANCE_TIMEOUT_S,
                output_bytes=_PROVENANCE_OUTPUT_BYTES,
            )
        except OSError:
            image_id_reason = "inspect_could_not_start"
        else:
            candidate = inspected.stdout.strip()
            if (
                inspected.returncode == 0
                and not inspected.output_truncated
                and _IMAGE_ID.fullmatch(candidate)
            ):
                image_id = candidate
            elif inspected.returncode == 124:
                image_id_reason = "inspect_timeout"
            elif inspected.output_truncated:
                image_id_reason = "inspect_output_truncated"
            elif inspected.returncode != 0:
                image_id_reason = "inspect_failed"
            else:
                image_id_reason = "inspect_returned_invalid_id"

        versions: dict[str, str | None] = {
            "python": None,
            "pytest": None,
            "ruff": None,
        }
        versions_reason: str | None = None
        probe_image = image_id or self.image
        name = f"lha-provenance-{uuid.uuid4().hex[:12]}"
        argv = [
            self.docker,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            *_SECURITY_RUN_ARGS,
            "--tmpfs",
            _TMPFS,
            "--env",
            "HOME=/tmp",
        ]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if self.limits.memory_mb:
            argv += ["--memory", f"{self.limits.memory_mb}m"]
        if self.limits.pids:
            argv += ["--pids-limit", str(self.limits.pids)]
        if self.limits.cpu_s:
            argv += ["--cpus", "1"]
        argv += ["--entrypoint", "python", probe_image, "-c", _VERSION_PROBE]

        def remove_probe(process) -> ProcessCleanupResult:
            client = terminate_process_group(process)
            if not client.confirmed:
                return client
            container = _remove_container(
                self.docker,
                name,
                output_bytes=_PROVENANCE_OUTPUT_BYTES,
            )
            return ProcessCleanupResult(
                container.confirmed,
                "; ".join(detail for detail in (client.detail, container.detail) if detail),
            )

        try:
            probe = run_bounded_process(
                argv,
                timeout=_PROVENANCE_TIMEOUT_S,
                output_bytes=_PROVENANCE_OUTPUT_BYTES,
                on_timeout=remove_probe,
                start_new_session=True,
                on_exit=terminate_process_group,
            )
        except OSError:
            versions_reason = "version_probe_could_not_start"
        else:
            if probe.cleanup_unconfirmed:
                versions_reason = "version_probe_cleanup_unconfirmed"
            elif probe.returncode == 0 and not probe.output_truncated:
                try:
                    payload = json.loads(probe.stdout)
                except json.JSONDecodeError:
                    versions_reason = "version_probe_returned_invalid_json"
                else:
                    if (
                        isinstance(payload, dict)
                        and set(payload) == set(versions)
                        and isinstance(payload.get("python"), str)
                        and bool(payload["python"])
                        and all(
                            value is None or (isinstance(value, str) and bool(value))
                            for key, value in payload.items()
                            if key != "python"
                        )
                    ):
                        versions = {key: payload.get(key) for key in versions}
                    else:
                        versions_reason = "version_probe_returned_invalid_schema"
            elif probe.returncode == 124:
                versions_reason = "version_probe_timeout"
            elif probe.output_truncated:
                versions_reason = "version_probe_output_truncated"
            else:
                versions_reason = "version_probe_failed"
            if not probe.cleanup_unconfirmed:
                cleanup = _remove_container(
                    self.docker,
                    name,
                    output_bytes=_PROVENANCE_OUTPUT_BYTES,
                )
                if not cleanup.confirmed:
                    versions = {key: None for key in versions}
                    versions_reason = "version_probe_cleanup_unconfirmed"

        return {
            "backend": self.name,
            "docker_executable": docker_executable,
            "docker_executable_status": (
                "available" if docker_executable is not None else "unavailable"
            ),
            "docker_executable_reason": docker_executable_reason,
            "image": self.image,
            "image_id": image_id,
            "image_id_status": "available" if image_id else "unavailable",
            "image_id_reason": image_id_reason,
            "versions": versions,
            "versions_image": probe_image,
            "versions_bound_to_image_id": image_id is not None,
            "versions_status": ("available" if versions_reason is None else "unavailable"),
            "versions_reason": versions_reason,
        }

    def build_argv(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        name: str,
        limits: ResourceLimits | None = None,
        operation_identity: str | None = None,
        cidfile: Path | None = None,
    ) -> list[str]:
        """The full ``docker run`` argv (separate for testability)."""
        limits = limits or self.limits
        argv = [
            self.docker,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            *_SECURITY_RUN_ARGS,
            "--tmpfs",
            _TMPFS,
            "--env",
            "HOME=/tmp",
            "-v",
            f"{Path(cwd).resolve()}:{_WORK}",
            "-w",
            _WORK,
        ]
        if operation_identity is not None:
            argv += ["--label", f"{_OPERATION_LABEL}={operation_identity}"]
        if cidfile is not None:
            argv += ["--cidfile", str(cidfile)]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if limits.memory_mb:
            argv += ["--memory", f"{limits.memory_mb}m"]
        if limits.pids:
            argv += ["--pids-limit", str(limits.pids)]
        if limits.cpu_s:
            # No direct CPU-seconds cap in docker run; bound parallelism instead
            # (wall-clock is bounded by the timeout either way).
            argv += ["--cpus", "1"]
        for host, container in self.ro_mounts.items():
            argv += ["-v", f"{Path(host).resolve()}:{container}:ro"]
        argv.append(self.image)
        # Host interpreter paths mean nothing inside the container.
        cmd = list(cmd)
        if cmd and cmd[0] == sys.executable:
            cmd[0] = "python"
        return argv + cmd

    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        timeout: float = 300.0,
        input: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> ProcResult:
        try:
            self.bind_control_plane(verify_digest=True)
        except (OSError, RuntimeError, ValueError) as error:
            return ProcResult(
                127,
                "",
                f"Docker control executable is unavailable: {type(error).__name__}",
                0.0,
            )
        result = self._run_bound(
            cmd,
            cwd=cwd,
            timeout=timeout,
            input=input,
            limits=limits,
        )
        try:
            self.bind_control_plane(verify_digest=True)
        except (OSError, RuntimeError, ValueError) as error:
            return _cleanup_failure_result(
                result,
                ProcessCleanupResult(
                    False,
                    f"Docker executable identity changed during execution: {type(error).__name__}",
                ),
                output_bytes=(limits or self.limits).output_bytes,
            )
        return result

    def _run_bound(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        timeout: float = 300.0,
        input: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> ProcResult:
        from ..operation_lease import (
            OperationLeaseError,
            OperationLeaseStore,
            operation_store_for_workdir,
        )

        store = (
            OperationLeaseStore(self.operation_lease_dir)
            if self.operation_lease_dir is not None
            else operation_store_for_workdir(cwd)
        )
        try:
            lease = store.activate_docker(cmd, cwd=cwd) if store is not None else None
        except (OSError, OperationLeaseError) as error:
            return ProcResult(
                127,
                "",
                f"failed to persist operation lease: {error}",
                0.0,
            )

        def fail_before_spawn(detail: str) -> ProcResult:
            if lease is not None and store is not None:
                try:
                    store.clear(lease.operation_id)
                except OperationLeaseError as error:
                    return ProcResult(
                        PROCESS_CLEANUP_RETURN_CODE,
                        "",
                        detail,
                        0.0,
                        cleanup_confirmed=False,
                        cleanup_detail=(
                            "no Docker client was spawned, but its operation "
                            f"lease could not be cleared: {error}"
                        ),
                    )
            return ProcResult(127, "", detail, 0.0)

        name = lease.container_name if lease is not None else f"lha-{uuid.uuid4().hex[:12]}"
        assert name is not None
        cidfile: Path | None = None
        if lease is not None and store is not None:
            cid_directory = store.run_dir / "active-container-ids"
            cid_directory_existed = cid_directory.exists()
            try:
                cid_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                if cid_directory.is_symlink() or not cid_directory.is_dir():
                    raise OSError("Docker cidfile directory is unsafe")
                if not cid_directory_existed:
                    _fsync_directory(store.run_dir)
            except OSError as error:
                return fail_before_spawn(f"failed to prepare Docker cidfile directory: {error}")
            cidfile = _container_id_path(store, lease.operation_id)
            if cidfile.is_symlink() or cidfile.exists():
                return fail_before_spawn("Docker cidfile path already exists")
        argv = self.build_argv(
            cmd,
            cwd=cwd,
            name=name,
            limits=limits,
            operation_identity=(lease.container_identity if lease is not None else None),
            cidfile=cidfile,
        )
        if input is not None:  # keep stdin open so piped text reaches the container
            argv.insert(argv.index("run") + 1, "-i")
        effective_limits = limits or self.limits

        timeout_cleanup: ProcessCleanupResult | None = None
        client_started = False

        def bind_container_id(process) -> None:
            nonlocal client_started, lease
            client_started = True
            if cidfile is None or store is None or lease is None:
                return
            deadline = time.monotonic() + _CONTAINER_ID_TIMEOUT_S
            while True:
                try:
                    container_id = _read_container_id(cidfile)
                except (FileNotFoundError, _DockerContainerIdentityPending):
                    container_id = None
                if container_id is not None:
                    lease = store.bind_container_id(
                        lease.operation_id,
                        container_id,
                    )
                    discarded = _discard_cidfile(
                        store,
                        lease.operation_id,
                    )
                    if not discarded.confirmed:
                        raise DockerContainerIdentityError(discarded.detail)
                    return
                if process.poll() is not None:
                    return
                if time.monotonic() >= deadline:
                    raise DockerContainerIdentityError(
                        "Docker did not publish a container ID before execution"
                    )
                time.sleep(0.01)

        def remove_container(process) -> ProcessCleanupResult:
            # The client can create the container until its process group is
            # gone. Confirm that boundary first, then inspect/remove the
            # daemon-side object by its immutable ID and lease label.
            nonlocal timeout_cleanup
            client_cleanup = terminate_process_group(process)
            if not client_cleanup.confirmed:
                timeout_cleanup = client_cleanup
                return timeout_cleanup
            container_cleanup = _remove_owned_container(
                self.docker,
                name,
                output_bytes=effective_limits.output_bytes,
                store=store,
                lease=lease,
            )
            timeout_cleanup = ProcessCleanupResult(
                container_cleanup.confirmed,
                "; ".join(
                    detail
                    for detail in (
                        client_cleanup.detail,
                        container_cleanup.detail,
                    )
                    if detail
                ),
            )
            return timeout_cleanup

        try:
            result = run_bounded_process(
                argv,
                timeout=timeout,
                input=input,
                output_bytes=effective_limits.output_bytes,
                on_timeout=remove_container,
                start_new_session=True,
                on_exit=terminate_process_group,
                on_started=bind_container_id,
            )
        except (
            OSError,
            UnicodeError,
            DockerContainerIdentityError,
            OperationLeaseError,
        ) as e:
            if lease is None or store is None:
                return ProcResult(127, "", f"failed to start docker: {e}", 0.0)
            cleanup = (
                _remove_owned_container(
                    self.docker,
                    name,
                    output_bytes=effective_limits.output_bytes,
                    store=store,
                    lease=lease,
                )
                if client_started
                else _clear_docker_operation(store, lease)
            )
            if not cleanup.confirmed:
                return ProcResult(
                    PROCESS_CLEANUP_RETURN_CODE,
                    "",
                    f"failed to start docker: {e}",
                    0.0,
                    cleanup_confirmed=False,
                    cleanup_detail=cleanup.detail,
                )
            return ProcResult(127, "", f"failed to start docker: {e}", 0.0)

        if result.cleanup_unconfirmed:
            # A live or unconfirmed client can still create a daemon-side
            # container. Preserve the lease for recovery instead of clearing
            # it after a racy absence check.
            return result
        if timeout_cleanup is not None:
            if not timeout_cleanup.confirmed and not result.cleanup_unconfirmed:
                return _cleanup_failure_result(
                    result,
                    timeout_cleanup,
                    output_bytes=effective_limits.output_bytes,
                )
            cleanup = timeout_cleanup
        else:
            # Even a successful client can leave descendants or a daemon-side
            # object. The lease identity must be checked before removal.
            cleanup = _remove_owned_container(
                self.docker,
                name,
                output_bytes=effective_limits.output_bytes,
                store=store,
                lease=lease,
            )

        if not cleanup.confirmed:
            return _cleanup_failure_result(
                result,
                cleanup,
                output_bytes=effective_limits.output_bytes,
            )
        if lease is not None and store is not None:
            try:
                store.clear(lease.operation_id)
            except (OSError, OperationLeaseError) as error:
                return _cleanup_failure_result(
                    result,
                    ProcessCleanupResult(
                        False,
                        f"container is absent but lease cleanup failed: {error}",
                    ),
                    output_bytes=effective_limits.output_bytes,
                )
        return result
