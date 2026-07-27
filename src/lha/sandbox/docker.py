"""Docker execution backend: network-less, resource-bounded containers.

The default for external/untrusted target repos. The working directory is the
only writable mount; declared read-only mounts (base repo, canonical tests,
scorer data) cannot be modified from inside; the environment starts empty; the
network is off; CPU/memory/pids are capped; a timeout force-removes the
container when Docker responds and reports any cleanup failure.

The image must provide the tools the run needs — the default python:3.12-slim
carries none of pytest/pytest-json-report/ruff, so code-verification tasks need
a purpose-built image (see SECURITY.md, "Execution backends").
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

from ..tools.shell import ProcResult
from .base import ExecutionBackend, ResourceLimits, run_bounded_process

_WORK = "/work"
_TMPFS = "/tmp:rw,nosuid,nodev,size=256m,mode=1777"
_REMOVE_TIMEOUT_S = 30.0
_PROVENANCE_TIMEOUT_S = 30.0
_PROVENANCE_OUTPUT_BYTES = 64 * 1024
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
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


def _remove_after_timeout(docker: str, name: str, *, output_bytes: int) -> str:
    """Try to kill the timed-out container and report what actually happened."""
    argv = [docker, "rm", "-f", name]
    try:
        cleanup = run_bounded_process(
            argv,
            timeout=_REMOVE_TIMEOUT_S,
            output_bytes=output_bytes,
        )
    except OSError as error:
        return f"container cleanup could not start; {name} may still be running: {error}"

    if cleanup.returncode == 124:
        detail = (cleanup.stderr or cleanup.stdout).strip()
        suffix = f": {detail}" if detail else ""
        return (
            f"container cleanup timed out after {_REMOVE_TIMEOUT_S:g}s; "
            f"{name} may still be running{suffix}"
        )
    if cleanup.returncode != 0:
        detail = (cleanup.stderr or cleanup.stdout).strip()
        suffix = f": {detail}" if detail else ""
        return (
            f"container cleanup failed with exit code {cleanup.returncode}; "
            f"{name} may still be running{suffix}"
        )
    return "container removed"


class DockerBackend(ExecutionBackend):
    name = "docker"

    def __init__(
        self,
        image: str = "python:3.12-slim",
        limits: ResourceLimits | None = None,
        ro_mounts: dict[str, str] | None = None,
        docker: str = "docker",
    ):
        self.image = image
        self.limits = limits or ResourceLimits(memory_mb=4096, pids=512)
        # host path -> container path, mounted read-only
        self.ro_mounts = dict(ro_mounts or {})
        self.docker = docker

    @classmethod
    def available(cls, docker: str = "docker") -> bool:
        exe = shutil.which(docker)
        if not exe:
            return False
        try:
            probe = run_bounded_process(
                [exe, "info"],
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

    def provenance(self) -> dict[str, object]:
        """Inspect immutable image identity and tool versions without target mounts.

        Provenance is diagnostic evidence, not a verifier gate. Every failure is
        returned as an explicit unavailable status instead of raising into the
        task being verified.
        """
        image_id: str | None = None
        image_id_reason: str | None = None
        try:
            inspected = run_bounded_process(
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

        def remove_probe(_process) -> str:
            return _remove_after_timeout(
                self.docker,
                name,
                output_bytes=_PROVENANCE_OUTPUT_BYTES,
            )

        try:
            probe = run_bounded_process(
                argv,
                timeout=_PROVENANCE_TIMEOUT_S,
                output_bytes=_PROVENANCE_OUTPUT_BYTES,
                on_timeout=remove_probe,
            )
        except OSError:
            versions_reason = "version_probe_could_not_start"
        else:
            if probe.returncode == 0 and not probe.output_truncated:
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
                            value is None
                            or (isinstance(value, str) and bool(value))
                            for key, value in payload.items()
                            if key != "python"
                        )
                    ):
                        versions = {
                            key: payload.get(key)
                            for key in versions
                        }
                    else:
                        versions_reason = "version_probe_returned_invalid_schema"
            elif probe.returncode == 124:
                versions_reason = "version_probe_timeout"
            elif probe.output_truncated:
                versions_reason = "version_probe_output_truncated"
            else:
                versions_reason = "version_probe_failed"

        return {
            "backend": self.name,
            "image": self.image,
            "image_id": image_id,
            "image_id_status": "available" if image_id else "unavailable",
            "image_id_reason": image_id_reason,
            "versions": versions,
            "versions_image": probe_image,
            "versions_bound_to_image_id": image_id is not None,
            "versions_status": (
                "available" if versions_reason is None else "unavailable"
            ),
            "versions_reason": versions_reason,
        }

    def build_argv(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        name: str,
        limits: ResourceLimits | None = None,
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
            "--tmpfs",
            _TMPFS,
            "--env",
            "HOME=/tmp",
            "-v",
            f"{Path(cwd).resolve()}:{_WORK}",
            "-w",
            _WORK,
        ]
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
        name = f"lha-{uuid.uuid4().hex[:12]}"
        argv = self.build_argv(cmd, cwd=cwd, name=name, limits=limits)
        if input is not None:  # keep stdin open so piped text reaches the container
            argv.insert(argv.index("run") + 1, "-i")
        effective_limits = limits or self.limits

        def remove_container(_process) -> str:
            # Killing the docker client is not enough: explicitly remove the
            # named container, then the shared runner closes the client.
            return _remove_after_timeout(
                self.docker,
                name,
                output_bytes=effective_limits.output_bytes,
            )

        try:
            return run_bounded_process(
                argv,
                timeout=timeout,
                input=input,
                output_bytes=effective_limits.output_bytes,
                on_timeout=remove_container,
            )
        except OSError as e:
            return ProcResult(127, "", f"failed to start docker: {e}", 0.0)
