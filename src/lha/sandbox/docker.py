"""Docker execution backend: network-less, resource-bounded containers.

The default for external/untrusted target repos. The working directory is the
only writable mount; declared read-only mounts (base repo, canonical tests,
scorer data) cannot be modified from inside; the environment starts empty; the
network is off; CPU/memory/pids are capped; a timeout removes the container,
which kills the entire process tree.

The image must provide the tools the run needs — the default python:3.12-slim
carries none of pytest/pytest-json-report/ruff, so code-verification tasks need
a purpose-built image (see SECURITY.md, "Execution backends").
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ..tools.shell import ProcResult
from .base import ExecutionBackend, ResourceLimits

_WORK = "/work"


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
            probe = subprocess.run(
                [exe, "info"], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return probe.returncode == 0

    def python(self) -> str:
        return "python"

    def tool(self, name: str) -> str:
        return name

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
            "--env",
            "HOME=/tmp",
            "-v",
            f"{Path(cwd).resolve()}:{_WORK}",
            "-w",
            _WORK,
        ]
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
        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=input,
            )
            return ProcResult(proc.returncode, proc.stdout, proc.stderr, time.monotonic() - start)
        except subprocess.TimeoutExpired as e:
            # Removing the container kills its whole process tree.
            subprocess.run(
                [self.docker, "rm", "-f", name], capture_output=True, text=True, timeout=30
            )
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ProcResult(
                124, out, f"timeout after {timeout}s (container removed)", time.monotonic() - start
            )
        except OSError as e:
            return ProcResult(127, "", f"failed to start docker: {e}", 0.0)
