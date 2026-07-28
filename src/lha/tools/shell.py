"""Bounded helper for fixed control-plane commands."""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Iterable

from ..process_result import ProcResult

_CONTROL_OUTPUT_BYTES = 4 * 1024 * 1024


def _is_user_writable(path: Path) -> bool:
    """Whether normal process credentials can replace entries below ``path``."""
    try:
        info = path.stat()
    except OSError:
        return True
    mode = info.st_mode
    if (
        hasattr(os, "geteuid")
        and os.geteuid() != 0
        and os.access(path, os.W_OK)
    ):
        return True
    if mode & stat.S_IWOTH:
        return True
    groups = set(os.getgroups()) if hasattr(os, "getgroups") else set()
    if hasattr(os, "getegid"):
        groups.add(os.getegid())
    if mode & stat.S_IWGRP and info.st_gid in groups:
        return True
    if (
        hasattr(os, "geteuid")
        and os.geteuid() != 0
        and info.st_uid == os.geteuid()
        and mode & stat.S_IWUSR
    ):
        return True
    return False


def _has_writable_ancestor(path: Path) -> bool:
    current = path
    while True:
        if _is_user_writable(current):
            return True
        if current.parent == current:
            return False
        current = current.parent


def sanitized_absolute_path(
    *,
    value: str | None = None,
    extra_dirs: Iterable[str | Path] = (),
    require_unwritable: bool = False,
) -> str:
    """Return a PATH containing only existing absolute directory names.

    Relative and empty entries implicitly search the subprocess working
    directory. Control-plane processes must never inherit that behavior while
    their cwd is a model-controlled repository.
    """
    candidates = [
        *(str(item) for item in extra_dirs),
        *(value if value is not None else os.environ.get("PATH", "")).split(
            os.pathsep
        ),
        *os.defpath.split(os.pathsep),
    ]
    accepted: list[str] = []
    for raw in candidates:
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not resolved.is_dir():
            continue
        if require_unwritable and _has_writable_ancestor(resolved):
            continue
        text = str(resolved)
        if text not in accepted:
            accepted.append(text)
    return os.pathsep.join(accepted)


def trusted_executable(
    name: str,
    *,
    path: str | None = None,
    extra_dirs: Iterable[str | Path] = (),
    require_unwritable: bool = True,
) -> str | None:
    """Resolve a control executable to an absolute, validated file."""
    if not name or Path(name).name != name:
        raise ValueError(f"executable name must be a basename: {name!r}")
    safe_path = sanitized_absolute_path(
        value=path,
        extra_dirs=extra_dirs,
        require_unwritable=require_unwritable,
    )
    if not safe_path:
        return None
    found = shutil.which(name, path=safe_path)
    if found is None:
        return None
    try:
        resolved = Path(found).resolve(strict=True)
        info = resolved.stat()
    except (OSError, RuntimeError):
        return None
    if (
        not resolved.is_absolute()
        or not stat.S_ISREG(info.st_mode)
        or not os.access(resolved, os.X_OK)
        or (require_unwritable and _has_writable_ancestor(resolved))
    ):
        return None
    return str(resolved)


def run(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: float = 300.0,
    env: dict[str, str] | None = None,
    input: str | None = None,
) -> ProcResult:
    """Run a fixed harness command without inheriting secrets or orphaning children."""
    from ..sandbox.base import (
        PROCESS_CLEANUP_RETURN_CODE,
        process_group_cleanup_supported,
        run_bounded_process,
        scrub_env,
        terminate_process_group,
    )

    if not process_group_cleanup_supported():
        return ProcResult(
            PROCESS_CLEANUP_RETURN_CODE,
            "",
            (
                "control command requires POSIX process-group cleanup; "
                "use Linux, macOS, or WSL2"
            ),
            0.0,
            cleanup_confirmed=False,
            cleanup_detail="POSIX process-group cleanup is unavailable",
        )
    try:
        return run_bounded_process(
            cmd,
            cwd=cwd,
            timeout=timeout,
            output_bytes=_CONTROL_OUTPUT_BYTES,
            env=env if env is not None else scrub_env(),
            input=input,
            start_new_session=True,
            on_exit=terminate_process_group,
        )
    except OSError as error:
        return ProcResult(
            127,
            "",
            f"failed to execute {cmd[0] if cmd else '<empty>'}: {error}",
            0.0,
        )


def venv_tool(name: str) -> str:
    """Resolve a console-script tool from the running interpreter's venv first."""
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    return found or name
