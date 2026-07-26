"""Canonical evidence for array-backed experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def safe_artifact_path(workdir: str | Path, value: str | Path) -> Path:
    """Resolve a task-supplied artifact path without leaving ``workdir``.

    Experiment artifacts are evidence, so following even an in-worktree
    symlink would let a run present bytes that it did not create.  Check every
    existing component before returning the resolved path.
    """
    root = Path(workdir).resolve()
    rel = Path(value)
    if not str(value) or not rel.parts or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe experiment artifact path: {value!s}")
    candidate = root
    for part in rel.parts:
        candidate /= part
        try:
            if candidate.is_symlink():
                raise ValueError(
                    f"experiment artifact path contains a symlink: {value!s}"
                )
        except OSError as e:
            raise ValueError(
                f"experiment artifact path could not be inspected: {value!s}"
            ) from e
    resolved = (root / rel).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise ValueError(f"experiment artifact path escapes workdir: {value!s}") from e
    return resolved


def array_sha256(array: Any) -> str:
    """Hash an array's dtype, shape, and contiguous bytes."""
    import numpy as np

    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode())
    digest.update(json.dumps(list(arr.shape), separators=(",", ":")).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def raw_array_sha256(array: Any) -> str:
    """Hash raw contiguous bytes for compatibility with existing experiment inputs."""
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def array_summary(array: Any) -> dict[str, Any]:
    """Return the bounded structural evidence stored beside an array."""
    import numpy as np

    arr = np.ascontiguousarray(array)
    return {
        "dtype": arr.dtype.str,
        "shape": [int(size) for size in arr.shape],
        "size": int(arr.size),
        "sha256": array_sha256(arr),
    }


def output_sha256(reference: Any, prediction: Any) -> str:
    """Bind both output arrays, including their dtype and shape."""
    digest = hashlib.sha256()
    digest.update(array_sha256(reference).encode())
    digest.update(array_sha256(prediction).encode())
    return digest.hexdigest()
