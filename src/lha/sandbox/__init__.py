"""Execution backends: where target/model-influenced code actually runs.

The harness executes code it did not write — patched repos, experiment
scripts, test suites. ``ExecutionBackend`` is the seam that decides where:

- ``trusted-local`` runs on the host interpreter with a scrubbed environment,
  resource limits, and process-group kill. It is named what it is: only for
  repositories you already trust (this repo's own self-eval and tests).
- ``docker`` runs in a network-less container with CPU/memory/pids limits,
  for external or untrusted target repos.
"""

from .base import ExecutionBackend, ResourceLimits, scrub_env
from .docker import DockerBackend
from .local import TrustedLocalBackend


def make_backend(name: str, **kwargs) -> ExecutionBackend:
    if name == "trusted-local":
        return TrustedLocalBackend(**kwargs)
    if name == "docker":
        return DockerBackend(**kwargs)
    raise ValueError(f"unknown execution backend: {name!r}")


__all__ = [
    "ExecutionBackend",
    "ResourceLimits",
    "TrustedLocalBackend",
    "DockerBackend",
    "make_backend",
    "scrub_env",
]
