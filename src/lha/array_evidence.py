"""Canonical evidence for array-backed experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

MAX_NPY_FILE_BYTES = 64 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 16 * 1024
MAX_NPY_DIMENSIONS = 8
MAX_NPY_ELEMENTS = 8 * 1024 * 1024
MAX_NPY_ITEMSIZE = 16


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


def load_bounded_npy(
    path: str | Path,
    *,
    max_file_bytes: int = MAX_NPY_FILE_BYTES,
    max_header_bytes: int = MAX_NPY_HEADER_BYTES,
    max_dimensions: int = MAX_NPY_DIMENSIONS,
    max_elements: int = MAX_NPY_ELEMENTS,
) -> Any:
    """Read a numeric NPY only after bounding its header and allocation."""
    import numpy as np
    from numpy.lib import format as npy_format

    limits = {
        "max_file_bytes": max_file_bytes,
        "max_header_bytes": max_header_bytes,
        "max_dimensions": max_dimensions,
        "max_elements": max_elements,
    }
    if any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("NPY bounds must be positive integers")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(Path(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("NPY artifact is not a regular file")
        if before.st_size <= 0 or before.st_size > max_file_bytes:
            raise ValueError(
                f"NPY file size {before.st_size} is outside the allowed bound"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            version = npy_format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_1_0(
                    stream,
                    max_header_size=max_header_bytes,
                )
            elif version == (2, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_2_0(
                    stream,
                    max_header_size=max_header_bytes,
                )
            else:
                raise ValueError(f"unsupported NPY format version: {version!r}")

            if len(shape) > max_dimensions:
                raise ValueError(
                    f"NPY has {len(shape)} dimensions; maximum is {max_dimensions}"
                )
            elements = 1
            for dimension in shape:
                if type(dimension) is not int or dimension < 0:
                    raise ValueError("NPY shape contains an invalid dimension")
                elements *= dimension
                if elements > max_elements:
                    raise ValueError(
                        f"NPY has more than {max_elements} elements"
                    )
            if elements <= 0:
                raise ValueError("NPY array must not be empty")

            dtype = np.dtype(dtype)
            if (
                dtype.hasobject
                or dtype.fields is not None
                or dtype.subdtype is not None
                or dtype.kind not in "iufc"
                or dtype.itemsize <= 0
                or dtype.itemsize > MAX_NPY_ITEMSIZE
            ):
                raise ValueError(f"unsupported NPY dtype: {dtype!s}")

            payload_bytes = elements * dtype.itemsize
            payload_offset = stream.tell()
            if payload_offset + payload_bytes != before.st_size:
                raise ValueError(
                    "NPY payload size does not match its declared shape and dtype"
                )
            array = np.fromfile(stream, dtype=dtype, count=elements)
            if array.size != elements:
                raise ValueError("NPY payload ended before the declared array")
            after = os.fstat(descriptor)
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ):
                raise ValueError("NPY artifact changed while it was being read")
            return array.reshape(shape, order="F" if fortran_order else "C")
    finally:
        os.close(descriptor)


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
