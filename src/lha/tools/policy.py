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

import configparser
import fnmatch
import os
import re
import shlex
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Protocol, cast

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
        "pytest.toml",
        ".pytest.toml",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "noxfile.py",
        "pytest.ini",
        ".pytest.ini",
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

_PYTEST_CONFIG_BYTES = 1024 * 1024
_PYTEST_DEFAULT_PATTERNS = ("test_*.py", "*_test.py")
_PYTEST_SCAN_IGNORES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
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


@dataclass(frozen=True)
class PytestCollectionConfig:
    """Local collection settings that enlarge Pytest's oracle surface."""

    python_files: tuple[str, ...]
    testpaths: tuple[str, ...]
    addopts: tuple[str, ...]


class ProtectedPathSet(Protocol):
    """Structural interface implemented by trusted oracle inventories."""

    @property
    def protected_paths(self) -> Iterable[str]: ...

    @property
    def protected_roots(self) -> Iterable[str]: ...


def discover_pytest_configuration(workdir: str | Path) -> PytestCollectionConfig:
    """Read supported local Pytest configuration without importing target code.

    Every discovered configuration is unioned. Pytest normally selects one
    configuration file, but protecting a harmless superset is safer than
    guessing differently from the interpreter that will later collect tests.
    """
    root = Path(workdir)
    try:
        metadata = root.lstat()
    except OSError as error:
        raise ValueError("pytest oracle root cannot be inspected") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("pytest oracle root must be a real directory")

    configured_patterns: list[str] = []
    configured_roots: list[str] = []
    configured_addopts: list[str] = []
    for name in (
        "pytest.toml",
        ".pytest.toml",
        "pytest.ini",
        ".pytest.ini",
        "pyproject.toml",
        "tox.ini",
        "setup.cfg",
    ):
        path = root / name
        try:
            path_metadata = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"pytest configuration cannot be inspected: {name}") from error
        if (
            stat.S_ISLNK(path_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or path_metadata.st_size > _PYTEST_CONFIG_BYTES
        ):
            raise ValueError(f"pytest configuration is unsafe: {name}")
        try:
            raw = path.read_bytes()
            options = _pytest_options(name, raw)
        except (OSError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as error:
            raise ValueError(f"pytest configuration is invalid: {name}") from error
        configured_patterns.extend(_pytest_option_words(options.get("python_files")))
        configured_roots.extend(_pytest_option_words(options.get("testpaths")))
        configured_addopts.extend(_pytest_option_words(options.get("addopts")))

    patterns = tuple(dict.fromkeys(configured_patterns)) or _PYTEST_DEFAULT_PATTERNS
    if any(
        not pattern
        or "/" in pattern
        or "\\" in pattern
        or pattern in {".", ".."}
        for pattern in patterns
    ):
        raise ValueError("pytest python_files must contain basename patterns")

    roots = tuple(
        dict.fromkeys(_canonical_pytest_root(raw) for raw in configured_roots)
    )
    return PytestCollectionConfig(
        python_files=patterns,
        testpaths=roots,
        addopts=tuple(configured_addopts),
    )


def discover_pytest_oracle_paths(workdir: str | Path) -> list[str]:
    """Conservatively find filename-configured Python test modules.

    This static discovery remains useful when a trusted collection backend is
    unavailable. Formal runs should use ``build_pytest_oracle_inventory``,
    which also protects collected files and every pre-existing file below
    configured ``testpaths``.
    """
    root = Path(workdir)
    config = discover_pytest_configuration(root)
    scan_roots: list[Path] = []
    if config.testpaths:
        for rel in config.testpaths:
            candidate = root / rel
            try:
                candidate_metadata = candidate.lstat()
            except FileNotFoundError:
                # Pytest treats an absent configured test path as a collection
                # problem; there is no existing file for policy to protect.
                continue
            except OSError as error:
                raise ValueError(f"pytest test path cannot be inspected: {rel}") from error
            if (
                stat.S_ISLNK(candidate_metadata.st_mode)
                or not stat.S_ISDIR(candidate_metadata.st_mode)
            ):
                raise ValueError(f"pytest test path must be a real directory: {rel}")
            scan_roots.append(candidate)
    else:
        scan_roots.append(root)

    protected: set[str] = set()

    def walk_error(error: OSError) -> None:
        raise ValueError("pytest test tree cannot be inspected") from error

    for scan_root in scan_roots:
        for directory, directory_names, file_names in os.walk(
            scan_root,
            topdown=True,
            followlinks=False,
            onerror=walk_error,
        ):
            current = Path(directory)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                if name in _PYTEST_SCAN_IGNORES:
                    continue
                candidate = current / name
                relative = candidate.relative_to(root).as_posix()
                try:
                    candidate_metadata = candidate.lstat()
                except OSError as error:
                    raise ValueError(
                        f"pytest directory changed during inspection: {relative}"
                    ) from error
                if stat.S_ISLNK(candidate_metadata.st_mode):
                    raise ValueError(f"pytest directory is a symbolic link: {relative}")
                if not stat.S_ISDIR(candidate_metadata.st_mode):
                    raise ValueError(f"pytest directory is not a directory: {relative}")
                kept_directories.append(name)
            directory_names[:] = kept_directories

            for name in sorted(file_names):
                if not any(
                    fnmatch.fnmatchcase(name, pattern)
                    for pattern in config.python_files
                ):
                    continue
                path = current / name
                relative = path.relative_to(root).as_posix()
                try:
                    file_metadata = path.lstat()
                except OSError as error:
                    raise ValueError(
                        f"pytest file changed during inspection: {relative}"
                    ) from error
                if (
                    stat.S_ISLNK(file_metadata.st_mode)
                    or not stat.S_ISREG(file_metadata.st_mode)
                    or file_metadata.st_nlink != 1
                ):
                    raise ValueError(f"pytest file is not a unique regular file: {relative}")
                protected.add(relative)
    return sorted(protected)


def _pytest_options(name: str, raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8")
    if name in {"pytest.toml", ".pytest.toml"}:
        data = tomllib.loads(text)
        options = data.get("pytest")
        if options is None:
            return {}
        if not isinstance(options, dict):
            raise ValueError("pytest must be a table")
        return options
    if name == "pyproject.toml":
        data = tomllib.loads(text)
        tool = data.get("tool")
        pytest_table = tool.get("pytest") if isinstance(tool, dict) else None
        options = None
        if isinstance(pytest_table, dict):
            native_options = {
                key: value
                for key, value in pytest_table.items()
                if key != "ini_options"
            }
            legacy_options = pytest_table.get("ini_options")
            if native_options and legacy_options is not None:
                raise ValueError(
                    "tool.pytest and tool.pytest.ini_options cannot both configure pytest"
                )
            options = native_options or legacy_options
        if options is None:
            return {}
        if not isinstance(options, dict):
            raise ValueError("tool.pytest.ini_options must be a table")
        return options

    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(text)
    sections = ("pytest", "tool:pytest")
    for section in sections:
        if parser.has_section(section):
            return dict(parser.items(section))
    return {}


def _pytest_option_words(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError as error:
            raise ValueError("invalid quoted pytest option") from error
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        words: list[str] = []
        for item in value:
            words.extend(_pytest_option_words(item))
        return words
    raise ValueError("pytest path and filename options must be strings or string arrays")


def _canonical_pytest_root(raw: str) -> str:
    if not raw or "\\" in raw or "\x00" in raw:
        raise ValueError(f"unsafe pytest test path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe pytest test path: {raw!r}")
    return path.as_posix()


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


def _take_git_path(value: str, *, context: str) -> tuple[str, str]:
    """Take one Git path atom without interpreting its C-style escapes."""
    value = value.lstrip(" ")
    if not value:
        raise ValueError(f"missing path in {context}")
    if value[0] != '"':
        token, separator, remainder = value.partition(" ")
        if not token:
            raise ValueError(f"missing path in {context}")
        return token, remainder.lstrip(" ") if separator else ""

    escaped = False
    for index in range(1, len(value)):
        char = value[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            return value[: index + 1], value[index + 1 :].lstrip(" ")
    raise ValueError(f"unterminated quoted path in {context}")


def _parse_diff_git_paths(line: str) -> tuple[str, str]:
    raw = line.removeprefix("diff --git ")
    first, remainder = _take_git_path(raw, context="diff --git header")
    second, trailing = _take_git_path(remainder, context="diff --git header")
    if trailing:
        raise ValueError(f"malformed diff --git header: {line!r}")
    return (
        _strip_diff_component(_decode_git_path(first)),
        _strip_diff_component(_decode_git_path(second)),
    )


def _parse_unified_path(line: str) -> str | None:
    """Parse a ---/+++ path, allowing the tab-separated timestamp form."""
    raw = line[4:]
    if raw.startswith('"'):
        token, trailing = _take_git_path(raw, context="unified-diff header")
        if trailing and not trailing.startswith("\t"):
            raise ValueError(f"malformed unified-diff path: {line!r}")
    else:
        token = raw.split("\t", 1)[0]
    path = _decode_git_path(token)
    if path == "/dev/null":
        return None
    return _strip_diff_component(path)


def _parse_extended_path(line: str, prefix: str) -> str:
    """Parse a rename/copy path, which deliberately has no a/ or b/ prefix."""
    token, trailing = _take_git_path(
        line.removeprefix(prefix),
        context=prefix.rstrip(),
    )
    if trailing:
        raise ValueError(f"malformed {prefix.rstrip()} header: {line!r}")
    path = _decode_git_path(token)
    if path == "/dev/null":
        raise ValueError(f"{prefix.rstrip()} cannot name /dev/null")
    return path


_HUNK_HEADER = re.compile(
    r"^@@ -\d+(?:,(?P<old>\d+))? \+\d+(?:,(?P<new>\d+))? @@(?: .*)?$"
)


@dataclass
class _DiffSection:
    git_old: str | None = None
    git_new: str | None = None
    old_header: str | None = None
    new_header: str | None = None
    old_header_seen: bool = False
    new_header_seen: bool = False
    rename_from: str | None = None
    rename_to: str | None = None
    copy_from: str | None = None
    copy_to: str | None = None

    @property
    def is_git(self) -> bool:
        return self.git_old is not None

    def resolved_paths(self) -> set[str]:
        """Validate redundant Git headers and return one complete write set."""
        if self.old_header_seen != self.new_header_seen:
            raise ValueError("unified diff must contain both --- and +++ headers")
        if (self.rename_from is None) != (self.rename_to is None):
            raise ValueError("unified diff has an incomplete rename header pair")
        if (self.copy_from is None) != (self.copy_to is None):
            raise ValueError("unified diff has an incomplete copy header pair")
        if self.rename_from is not None and self.copy_from is not None:
            raise ValueError("unified diff cannot mix rename and copy headers")

        paths = {
            path
            for path in (self.old_header, self.new_header)
            if path is not None
        }
        if not self.is_git:
            if not self.old_header_seen:
                raise ValueError("unified diff section has no file headers")
            return paths

        assert self.git_old is not None
        assert self.git_new is not None
        paths.update((self.git_old, self.git_new))
        if self.old_header is not None and self.old_header != self.git_old:
            raise ValueError(
                "diff --git and --- paths disagree: "
                f"{self.git_old!r} != {self.old_header!r}"
            )
        if self.new_header is not None and self.new_header != self.git_new:
            raise ValueError(
                "diff --git and +++ paths disagree: "
                f"{self.git_new!r} != {self.new_header!r}"
            )

        for operation, source, destination in (
            ("rename", self.rename_from, self.rename_to),
            ("copy", self.copy_from, self.copy_to),
        ):
            if source is None or destination is None:
                continue
            if source != self.git_old or destination != self.git_new:
                raise ValueError(
                    f"diff --git and {operation} paths disagree: "
                    f"{self.git_old!r} -> {self.git_new!r} != "
                    f"{source!r} -> {destination!r}"
                )
            paths.update((source, destination))
        return paths


def diff_paths(unified_diff: str) -> list[str]:
    """Return the complete, internally consistent write set of a unified diff.

    Declared ``touched_files`` can lie; the diff itself is what ``git apply``
    will act on. Git rename/copy headers use repository-relative paths while
    ``diff --git`` and ``---``/``+++`` paths lose one component under ``-p1``.
    All redundant headers must agree after those transforms.
    """
    paths: set[str] = set()
    current: _DiffSection | None = None
    old_remaining = 0
    new_remaining = 0

    def finish() -> None:
        nonlocal current
        if current is not None:
            paths.update(current.resolved_paths())
            current = None

    for line in (unified_diff or "").splitlines():
        if old_remaining or new_remaining:
            if line.startswith("\\ No newline at end of file"):
                continue
            if line.startswith(" "):
                old_remaining -= 1
                new_remaining -= 1
            elif line.startswith("-"):
                old_remaining -= 1
            elif line.startswith("+"):
                new_remaining -= 1
            else:
                raise ValueError("malformed unified-diff hunk body")
            if old_remaining < 0 or new_remaining < 0:
                raise ValueError("unified-diff hunk has more lines than declared")
            continue

        if line.startswith("diff --git "):
            finish()
            old, new = _parse_diff_git_paths(line)
            current = _DiffSection(git_old=old, git_new=new)
            continue

        if line.startswith("--- "):
            if current is None:
                current = _DiffSection()
            elif current.old_header_seen:
                if current.is_git:
                    raise ValueError("duplicate --- header in a diff --git section")
                finish()
                current = _DiffSection()
            current.old_header = _parse_unified_path(line)
            current.old_header_seen = True
            continue
        if line.startswith("+++ "):
            if current is None or not current.old_header_seen:
                raise ValueError("+++ header does not follow a --- header")
            if current.new_header_seen:
                raise ValueError("duplicate +++ header in a unified diff section")
            current.new_header = _parse_unified_path(line)
            current.new_header_seen = True
            continue

        extended_headers = (
            ("rename from ", "rename_from"),
            ("rename to ", "rename_to"),
            ("copy from ", "copy_from"),
            ("copy to ", "copy_to"),
        )
        for prefix, attribute in extended_headers:
            if not line.startswith(prefix):
                continue
            if current is None or not current.is_git:
                raise ValueError(f"{prefix.rstrip()} appears outside diff --git")
            if getattr(current, attribute) is not None:
                raise ValueError(f"duplicate {prefix.rstrip()} header")
            setattr(current, attribute, _parse_extended_path(line, prefix))
            break
        else:
            if line.startswith("@@ "):
                if current is None or not (
                    current.old_header_seen and current.new_header_seen
                ):
                    raise ValueError("unified-diff hunk appears before file headers")
                match = _HUNK_HEADER.match(line)
                if match is None:
                    raise ValueError(f"malformed unified-diff hunk header: {line!r}")
                old_remaining = int(match.group("old") or "1")
                new_remaining = int(match.group("new") or "1")

    if old_remaining or new_remaining:
        raise ValueError("truncated unified-diff hunk")
    finish()
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
    resolved: ResolvedPatch,
    allowed_overrides: Iterable[str] = (),
    *,
    additional_protected_files: Iterable[str] | ProtectedPathSet = (),
) -> list[str]:
    """Check the actual write set rather than model-authored path metadata."""
    allowed = {str(Path(a).as_posix()) for a in allowed_overrides}
    supplied_paths = cast(
        Iterable[str],
        getattr(
            additional_protected_files,
            "protected_paths",
            additional_protected_files,
        ),
    )
    additional = {
        str(Path(path).as_posix()).casefold()
        for path in supplied_paths
    }
    additional_roots = {
        str(Path(path).as_posix()).casefold().rstrip("/")
        for path in getattr(additional_protected_files, "protected_roots", ())
    }
    violations: list[str] = []
    for rel in resolved.paths:
        normalized = str(Path(rel).as_posix())
        folded = normalized.casefold()
        protected_by_root = any(
            root == "."
            or folded == root
            or folded.startswith(f"{root}/")
            for root in additional_roots
        )
        if (
            is_protected(rel)
            or folded in additional
            or protected_by_root
        ) and normalized not in allowed:
            violations.append(rel)
    return violations


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
