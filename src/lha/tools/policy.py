"""Oracle-protection policy: which files a model-generated patch may touch.

The test suite, verifier configuration, and build/CI configuration are the
oracle — if the agent can edit them, "tests pass" stops meaning anything. This
module is the single definition of what is protected; the run loop enforces it
before a patch is applied, and the ablation uses the same predicate when
stripping first attempts.

A task whose legitimate goal requires editing a protected file (e.g. "fix this
broken test") must authorize the exact paths via ``TaskSpec.allowed_protected_files``.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Iterable

from ..artifacts import Patch

# Filenames that configure the oracle or the build, wherever they appear.
_PROTECTED_NAMES = frozenset(
    {
        "conftest.py",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "noxfile.py",
        "pytest.ini",
        "ruff.toml",
        ".ruff.toml",
    }
)

# Directories whose entire contents are protected.
_PROTECTED_DIRS = frozenset({"tests", "test", ".github", ".ci"})

_DIFF_PATH = re.compile(r"^(?:---|\+\+\+) (?:a/|b/)?(?P<path>[^\t\n]+)", re.MULTILINE)
_DIFF_GIT = re.compile(r"^diff --git a/(?P<a>[^ ]+) b/(?P<b>[^\n]+)", re.MULTILINE)


def is_protected(rel: str) -> bool:
    """Whether a (relative) path is part of the oracle/config surface."""
    p = PurePosixPath(str(Path(rel).as_posix()))
    name = p.name
    return (
        any(part in _PROTECTED_DIRS for part in p.parts[:-1])
        or name in _PROTECTED_NAMES
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
    )


def diff_paths(unified_diff: str) -> list[str]:
    """Every path a unified diff touches, from both git and ---/+++ headers.

    Declared ``touched_files`` can lie; the diff itself is what ``git apply``
    will act on, so policy decisions must parse it.
    """
    paths: set[str] = set()
    for m in _DIFF_GIT.finditer(unified_diff or ""):
        paths.add(m.group("a"))
        paths.add(m.group("b"))
    for m in _DIFF_PATH.finditer(unified_diff or ""):
        path = m.group("path").strip()
        if path and path != "/dev/null":
            paths.add(path)
    return sorted(paths)


def patch_paths(patch: Patch) -> list[str]:
    """The full write surface of a patch: explicit contents, declared files,
    and everything its diff would touch."""
    paths = set(patch.file_contents) | set(patch.touched_files)
    paths.update(diff_paths(patch.unified_diff))
    return sorted(paths)


def check_patch(patch: Patch, allowed_overrides: Iterable[str] = ()) -> list[str]:
    """Protected paths this patch would touch, minus explicit task authorizations.

    An empty return means the patch respects the policy.
    """
    allowed = {str(Path(a).as_posix()) for a in allowed_overrides}
    return [
        rel
        for rel in patch_paths(patch)
        if is_protected(rel) and str(Path(rel).as_posix()) not in allowed
    ]


def strip_protected(patch: Patch, allowed_overrides: Iterable[str] = ()) -> Patch:
    """A copy of the patch with protected file edits removed (whole-file patches).

    Used by the ablation, where the paired design scores whatever source edits
    the first attempt made rather than rejecting the attempt outright.
    """
    allowed = {str(Path(a).as_posix()) for a in allowed_overrides}

    def ok(rel: str) -> bool:
        return not is_protected(rel) or str(Path(rel).as_posix()) in allowed

    fc = {k: v for k, v in patch.file_contents.items() if ok(k)}
    return patch.model_copy(
        update={"file_contents": fc, "touched_files": list(fc), "unified_diff": ""}
    )
