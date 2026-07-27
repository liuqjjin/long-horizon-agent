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

import shlex
from pathlib import Path, PurePosixPath
from typing import Iterable

from ..artifacts import Patch
from .patch import ResolvedPatch

# Filenames that configure the oracle or the build, wherever they appear.
_PROTECTED_NAMES = frozenset(
    {
        "conftest.py",
        "pytest.py",
        "pytest_jsonreport.py",
        "sitecustomize.py",
        "usercustomize.py",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "noxfile.py",
        "pytest.ini",
        "ruff.toml",
        ".ruff.toml",
        "uv.lock",
        "poetry.lock",
        "pdm.lock",
        "pipfile",
        "pipfile.lock",
        "dockerfile",
        "makefile",
        "justfile",
        "taskfile.yml",
        "taskfile.yaml",
        "compose.yml",
        "compose.yaml",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".gitlab-ci.yml",
        ".gitlab-ci.yaml",
        "jenkinsfile",
        "azure-pipelines.yml",
        "bitbucket-pipelines.yml",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "cargo.toml",
        "cargo.lock",
        "go.mod",
        "go.sum",
        "gemfile",
        "gemfile.lock",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
    }
)

# Directories whose entire contents are protected.
_PROTECTED_DIRS = frozenset(
    {
        "tests",
        "test",
        "pytest",
        "_pytest",
        "pytest_jsonreport",
        ".github",
        ".gitlab",
        ".circleci",
        ".ci",
    }
)


def is_protected(rel: str) -> bool:
    """Whether a (relative) path is part of the oracle/config surface.

    Comparison is case-insensitive: on the case-insensitive filesystems of
    macOS/Windows, writing ``Conftest.py`` overwrites ``conftest.py``, so a
    case variant is the same attack surface.
    """
    p = PurePosixPath(str(Path(rel).as_posix()))
    name = p.name.casefold()
    return (
        any(part.casefold() in _PROTECTED_DIRS for part in p.parts[:-1])
        or name in _PROTECTED_NAMES
        or name.startswith("dockerfile.")
        or (name.startswith("requirements") and name.endswith(".txt"))
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
    )


def _decode_git_path(token: str) -> str:
    """Decode one path token using Git's C-style quoting rules."""
    if not (token.startswith('"') and token.endswith('"')):
        return token
    value = token[1:-1]
    decoded = bytearray()
    escapes = {
        "a": b"\a",
        "b": b"\b",
        "t": b"\t",
        "n": b"\n",
        "v": b"\v",
        "f": b"\f",
        "r": b"\r",
        '"': b'"',
        "\\": b"\\",
    }
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\":
            decoded.extend(char.encode("utf-8"))
            index += 1
            continue
        index += 1
        if index >= len(value):
            raise ValueError("trailing backslash in a quoted git path")
        escaped = value[index]
        if escaped in escapes:
            decoded.extend(escapes[escaped])
            index += 1
            continue
        if escaped in "01234567":
            end = index + 1
            while end < min(index + 3, len(value)) and value[end] in "01234567":
                end += 1
            decoded.append(int(value[index:end], 8))
            index = end
            continue
        raise ValueError(f"unsupported escape in a quoted git path: \\{escaped}")
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("quoted git path is not valid UTF-8") from e


def _strip_diff_component(path: str) -> str:
    """Mirror the ``git apply -p1`` path transform used by the executor."""
    _prefix, separator, relative = path.partition("/")
    if not separator or not relative:
        raise ValueError(f"diff path has no component for -p1 to strip: {path!r}")
    return relative


def diff_paths(unified_diff: str) -> list[str]:
    """Every path a unified diff touches, from both git and ---/+++ headers.

    Declared ``touched_files`` can lie; the diff itself is what ``git apply``
    will act on, so policy decisions must parse it.
    """
    paths: set[str] = set()
    for line in (unified_diff or "").splitlines():
        tokens: list[str] = []
        if line.startswith("diff --git "):
            try:
                tokens = shlex.split(line, posix=False)
            except ValueError as e:
                raise ValueError(f"malformed diff --git header: {line!r}") from e
            if len(tokens) != 4:
                raise ValueError(f"malformed diff --git header: {line!r}")
            candidates = tokens[2:]
        elif line.startswith(("--- ", "+++ ")):
            raw = line[4:]
            if raw.startswith('"'):
                try:
                    quoted = shlex.split(raw, posix=False)
                except ValueError as e:
                    raise ValueError(f"malformed unified-diff path: {line!r}") from e
                if not quoted:
                    raise ValueError(f"missing unified-diff path: {line!r}")
                candidates = [quoted[0]]
            else:
                candidates = [raw.split("\t", 1)[0]]
        else:
            continue
        for candidate in candidates:
            path = _decode_git_path(candidate.strip())
            if path and path != "/dev/null":
                paths.add(_strip_diff_component(path))
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


def check_resolved(
    resolved: ResolvedPatch, allowed_overrides: Iterable[str] = ()
) -> list[str]:
    """Check the actual write set rather than model-authored path metadata."""
    allowed = {str(Path(a).as_posix()) for a in allowed_overrides}
    return [
        rel
        for rel in resolved.paths
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
