"""Trusted-local execution using the host interpreter.

Only for repositories you already trust (this repo's own tests and self-eval).
Still applies the cheap protections that cost nothing: a scrubbed environment
(no inherited secrets), best-effort POSIX resource limits, and process-group
cleanup after every exit so a target cannot orphan background children.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..tools.shell import ProcResult, venv_tool
from .base import (
    PROCESS_CLEANUP_RETURN_CODE,
    ExecutionBackend,
    ResourceLimits,
    process_group_cleanup_supported,
    run_bounded_process,
    scrub_env,
    terminate_process_group,
)


def _limited_command(cmd: list[str], limits: ResourceLimits) -> list[str]:
    if not limits.has_process_limits:
        return cmd
    launcher = [sys.executable, "-m", "lha.sandbox.limit_exec"]
    if limits.cpu_s is not None:
        launcher.extend(["--cpu-s", str(limits.cpu_s)])
    if limits.memory_mb is not None:
        launcher.extend(["--memory-mb", str(limits.memory_mb)])
    if limits.pids is not None:
        launcher.extend(["--pids", str(limits.pids)])
    return [*launcher, "--", *cmd]


class TrustedLocalBackend(ExecutionBackend):
    name = "trusted-local"

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        operation_lease_dir: str | Path | None = None,
    ):
        self.limits = limits or ResourceLimits()
        self.operation_lease_dir = (
            Path(operation_lease_dir).resolve()
            if operation_lease_dir is not None
            else None
        )

    def python(self) -> str:
        return sys.executable

    def tool(self, name: str) -> str:
        return venv_tool(name)

    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        timeout: float = 300.0,
        input: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> ProcResult:
        limits = limits or self.limits
        if not process_group_cleanup_supported():
            return ProcResult(
                PROCESS_CLEANUP_RETURN_CODE,
                "",
                (
                    "trusted-local requires POSIX process-group cleanup; "
                    "use Linux, macOS, WSL2, or the Docker backend"
                ),
                0.0,
                cleanup_confirmed=False,
                cleanup_detail=(
                    "POSIX process-group cleanup is unavailable"
                ),
            )

        command = _limited_command(cmd, limits)
        from ..operation_lease import (
            OperationLeaseError,
            OperationLeaseStore,
            build_local_launcher,
            operation_store_for_workdir,
            wait_until_local_active,
        )

        store = (
            OperationLeaseStore(self.operation_lease_dir)
            if self.operation_lease_dir is not None
            else operation_store_for_workdir(cwd)
        )
        if store is None:
            try:
                return run_bounded_process(
                    command,
                    cwd=str(cwd),
                    env=scrub_env(),
                    timeout=timeout,
                    input=input,
                    output_bytes=limits.output_bytes,
                    start_new_session=True,
                    # Always remove the original group. A background process can
                    # close stdio before the leader exits, so pipe drainage alone
                    # cannot prove that no descendant survived.
                    on_exit=terminate_process_group,
                )
            except OSError as e:
                return ProcResult(127, "", f"failed to start {cmd[0]!r}: {e}", 0.0)

        try:
            lease = store.prepare_local(command, cwd=cwd)
        except (OSError, OperationLeaseError) as error:
            return ProcResult(
                127,
                "",
                f"failed to persist operation lease: {error}",
                0.0,
            )
        release_read, release_write = os.pipe()
        released = False

        def release_launcher(process) -> None:
            nonlocal released
            wait_until_local_active(
                store,
                lease.operation_id,
                pid=process.pid,
            )
            if os.write(release_write, b"G") != 1:
                raise OperationLeaseError(
                    f"could not release operation {lease.operation_id}"
                )
            released = True

        def cleanup_operation(process):
            cleanup = terminate_process_group(process)
            if cleanup.confirmed:
                try:
                    store.clear(lease.operation_id)
                except OperationLeaseError as error:
                    return type(cleanup)(
                        False,
                        f"process group is absent but lease cleanup failed: {error}",
                    )
            return cleanup

        launcher = build_local_launcher(
            store,
            lease,
            command,
            release_fd=release_read,
        )
        try:
            return run_bounded_process(
                launcher,
                cwd=str(cwd),
                env=scrub_env(),
                timeout=timeout,
                input=input,
                output_bytes=limits.output_bytes,
                start_new_session=True,  # own process group -> killable as a tree
                # Always remove the original group. A background process can
                # close stdio before the leader exits, so pipe drainage alone
                # cannot prove that no descendant survived.
                on_exit=cleanup_operation,
                on_started=release_launcher,
                pass_fds=(release_read,),
            )
        except OperationLeaseError as error:
            return ProcResult(
                127,
                "",
                f"failed to activate operation lease: {error}",
                0.0,
            )
        except OSError as e:
            # Popen did not produce a process, so no target can still be live.
            if not released:
                try:
                    store.clear(lease.operation_id)
                except OperationLeaseError as clear_error:
                    return ProcResult(
                        PROCESS_CLEANUP_RETURN_CODE,
                        "",
                        (
                            f"failed to start {cmd[0]!r}: {e}; "
                            "operation lease cleanup could not be confirmed: "
                            f"{clear_error}"
                        ),
                        0.0,
                        cleanup_confirmed=False,
                        cleanup_detail=(
                            "target process was not created, but the prepared "
                            f"operation lease could not be cleared: {clear_error}"
                        ),
                    )
            return ProcResult(127, "", f"failed to start {cmd[0]!r}: {e}", 0.0)
        finally:
            for descriptor in (release_read, release_write):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
