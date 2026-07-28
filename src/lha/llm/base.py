"""LLM backend interface and response parsing.

The patch generator depends only on ``LLMClient``. Tests and self-evaluation use
``DeterministicStub`` without network access; configured runs can use a CLI or
API backend.
"""

from __future__ import annotations

import json
import os
import re
import stat
from abc import ABC, abstractmethod
from pathlib import Path

from ..artifacts import Patch, Plan, Step
from ..live_context.models import ContextBundle

_DIFF_FENCE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
# A '### path' header immediately followed by a fenced block holding the WHOLE file.
_FILE_BLOCK = re.compile(
    r"^###[ \t]*(?P<path>[^\n]+?)[ \t]*\n+```[^\n]*\n(?P<body>.*?)\n?```",
    re.DOTALL | re.MULTILINE,
)

_PLAN_SYSTEM = (
    "You plan work for a task runner that advances only after registered checks pass. "
    "Decompose the task into an ordered list of verifiable steps. Output ONLY a JSON "
    "object in a ```json "
    "fenced block with keys `summary` (string) and `steps` (array). Each step has: "
    "`step_id`, `kind` (code|experiment|context), `action` "
    "(gather_context|edit_code|run_experiment|answer_query), `goal`, and `verifiers` "
    "(subset of: pytest, ruff, psnr, ssim, reproducibility, freshness, citation). "
    "Mirror the structure of the example plan. Do not explain."
)


def _safe_repo_text(root: Path, path: Path, *, missing_ok: bool = False) -> str:
    """Read one standalone regular file without following worktree links.

    Opening every component relative to an already-open directory keeps a
    concurrent directory swap from redirecting the read outside the worktree.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"worktree is missing or unsafe: {root}")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source file is outside the worktree: {path}") from error
    if not relative.parts:
        raise ValueError("source path points at the worktree itself")

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_only = getattr(os, "O_DIRECTORY", 0)
    descriptors: list[int] = []
    try:
        try:
            current_fd = os.open(
                root,
                os.O_RDONLY | close_on_exec | no_follow | directory_only,
            )
        except FileNotFoundError:
            if missing_ok:
                return ""
            raise
        descriptors.append(current_fd)

        for part in relative.parts[:-1]:
            try:
                current_fd = os.open(
                    part,
                    os.O_RDONLY | close_on_exec | no_follow | directory_only,
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                if missing_ok:
                    return ""
                raise
            except OSError as error:
                raise ValueError(
                    f"source path contains a link or non-directory component: {relative}"
                ) from error
            descriptors.append(current_fd)

        try:
            descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | close_on_exec | no_follow,
                dir_fd=current_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return ""
            raise
        except OSError as error:
            raise ValueError(
                f"source file is a link or cannot be opened safely: {relative}"
            ) from error
        descriptors.append(descriptor)

        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise ValueError(f"source file is not a standalone regular file: {relative}")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            text = stream.read()
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or any(getattr(before, field) != getattr(after, field) for field in stable)
        ):
            raise ValueError(f"source file changed while it was read: {relative}")
        return text
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def extract_json(text: str) -> str:
    m = _JSON_FENCE.search(text)
    if m:
        return m.group(1)
    s = text.strip()
    return s if s.startswith("{") else ""


_IMPL_SYSTEM = (
    "You are a careful software engineer fixing a bug. You are given an issue, the "
    "relevant source files, and (on a retry) the failing-test output. Make a MINIMAL "
    "fix. Return the COMPLETE corrected contents of every file you change — for each, "
    "output a line '### <relative/path>' immediately followed by a fenced code block "
    "containing the ENTIRE file. Never abbreviate or use '...'. Do not include files "
    "you did not change, and do not include test files. Do not explain."
)


def extract_unified_diff(text: str) -> str:
    m = _DIFF_FENCE.search(text)
    if m:
        return m.group(1)
    # fall back: assume the whole response is a diff if it looks like one
    if text.lstrip().startswith(("--- ", "diff --git")):
        return text
    return ""


def extract_file_blocks(text: str) -> dict[str, str]:
    """Parse '### path' + fenced-block pairs into {relpath: full_file_text}."""
    out: dict[str, str] = {}
    for m in _FILE_BLOCK.finditer(text):
        # Headers come decorated: ``### `pkg/a.py` `` or ``### **a.py**``; strip the
        # markdown, then a literal leading "./" — but NOT lstrip("./"), which would
        # also eat the ".." of "../escape" and defeat the traversal guard downstream.
        path = m.group("path").strip().strip("`*").strip()
        if path.startswith("./"):
            path = path[2:]
        if path:
            out[path] = m.group("body")
    return out


class LLMClient(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, system: str, prompt: str) -> str: ...

    def set_trusted_oracle_paths(self, paths) -> None:
        """Withhold a trusted baseline oracle from source prompts.

        The patch policy remains the enforcement boundary. Prompt filtering is
        a separate confidentiality measure: custom-collected tests, fixtures,
        helpers, and test data should not be sent to the implementation model.
        """
        self._trusted_oracle_paths = _normalize_oracle_paths(paths)

    def plan(self, task, template: Plan) -> Plan | None:
        """Ask the LLM to (re)plan the task, or return None to use the caller's template.

        Real backends inherit this; the deterministic stub overrides it to return None
        so the default/eval path stays template-driven. Any parse/validation failure
        also yields None — the Supervisor then keeps its template plan.
        """
        prompt = (
            f"## Task\nkind: {task.kind}\ntitle: {task.title}\n"
            f"description: {task.description or task.title}\n\n"
            f"## Example plan for this task kind\n{template.model_dump_json(indent=2)}\n\n"
            "Return a JSON plan decomposing this task into verifiable steps."
        )
        response = self.complete(_PLAN_SYSTEM, prompt)
        try:
            data = json.loads(extract_json(response) or "{}")
            steps = [Step.model_validate(s) for s in data.get("steps", [])]
            if not steps:
                return None
            return Plan(
                task_id=task.title,
                summary=data.get("summary", f"Plan: {task.title}"),
                steps=steps,
                overall_success=task.success or template.overall_success,
            )
        except (AttributeError, TypeError, ValueError):
            return None

    def propose_patch(self, step, bundle: ContextBundle, workdir: str | Path) -> Patch:
        """Prompt the model for whole-file rewrites and build a Patch.

        Whole-file ``file_contents`` apply cleanly (a direct write), unlike a unified
        diff, which ``git apply`` rejects on the slightest context drift — the most
        common way an LLM's correct fix gets discarded. A display diff is still
        recorded for the human-facing artifact.
        """
        workdir = Path(workdir)
        oracle_paths = getattr(self, "_trusted_oracle_paths", ())
        visible_bundle = _without_oracle_context(
            bundle,
            workdir,
            oracle_paths,
        )
        ctx_text = "\n\n".join(
            f"# {i.provenance.locator}\n{i.text}"
            for i in visible_bundle.items[:8]
        )
        files_text = self._read_repo_python(
            workdir,
            trusted_oracle_paths=oracle_paths,
        )
        prompt = (
            f"## Issue\n{step.goal}\n\n"
            f"## Prior failures (if any)\n{chr(10).join(step.prior_failures) or 'none'}\n\n"
            f"## Retrieved context\n{ctx_text or 'none'}\n\n"
            f"## Source files\n{files_text}\n\n"
            "Return the complete corrected contents of each file you change, each "
            "preceded by a '### <path>' line, as instructed."
        )
        response = self.complete(_IMPL_SYSTEM, prompt)
        return self._patch_from_response(step, visible_bundle, workdir, response)

    def _patch_from_response(
        self, step: Step, bundle: ContextBundle, workdir: Path, response: str
    ) -> Patch:
        blocks = extract_file_blocks(response) or self._single_block_fallback(
            workdir,
            response,
            trusted_oracle_paths=getattr(self, "_trusted_oracle_paths", ()),
        )
        file_contents: dict[str, str] = {}
        for rel, content in blocks.items():
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                continue  # apply_patch guards too, but never even propose an escape
            target = workdir / rel
            try:
                original = _safe_repo_text(
                    workdir,
                    target,
                    missing_ok=True,
                )
            except (OSError, UnicodeDecodeError):
                continue
            norm = content if content.endswith("\n") else content + "\n"
            if norm.strip() and norm != original:
                file_contents[rel] = norm
        return Patch(
            step_id=step.step_id,
            file_contents=file_contents,
            touched_files=list(file_contents),
            rationale=f"Whole-file fix by {self.name}.",
            based_on_context=bundle.locators(),
        )

    @classmethod
    def _single_block_fallback(
        cls,
        workdir: Path,
        response: str,
        *,
        trusted_oracle_paths=(),
    ) -> dict[str, str]:
        """If the model returned a lone code block (no '### path'), map it to the only
        non-test source file when there is exactly one — otherwise give up cleanly."""
        m = _CODE_FENCE.search(response)
        if not m:
            return {}
        srcs = cls._repo_python_paths(
            workdir,
            trusted_oracle_paths=trusted_oracle_paths,
        )
        if len(srcs) == 1:
            return {str(srcs[0].relative_to(workdir)): m.group(1)}
        return {}

    @staticmethod
    def _is_test_file(rel: Path) -> bool:
        """Test files are the oracle. Keeping them out of the prompt forces a genuine
        fix from the issue (the failing-test summary still returns via repair
        feedback) instead of the model reverse-engineering the assertions. Omitting
        a large test suite also leaves more context space for the source files."""
        name = rel.name
        return (
            "tests" in rel.parts
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name == "conftest.py"
        )

    @classmethod
    def _repo_python_paths(
        cls,
        workdir: Path,
        *,
        trusted_oracle_paths=(),
    ) -> list[Path]:
        """List safe Python sources and reject links introduced by target code."""
        if workdir.is_symlink() or not workdir.is_dir():
            raise ValueError(f"worktree is missing or unsafe: {workdir}")
        sources: list[Path] = []
        oracle_paths = _normalize_oracle_paths(trusted_oracle_paths)
        for directory, names, files in os.walk(workdir, followlinks=False):
            parent = Path(directory)
            for name in [*names, *files]:
                path = parent / name
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError(
                        "worktree contains a symbolic link: "
                        f"{path.relative_to(workdir)}"
                    )
            for name in files:
                path = parent / name
                if path.suffix != ".py":
                    continue
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ValueError(
                        "Python source is not a standalone regular file: "
                        f"{path.relative_to(workdir)}"
                    )
                rel = path.relative_to(workdir)
                if (
                    ".cocoindex_code" not in path.parts
                    and not cls._is_test_file(rel)
                    and rel.as_posix().casefold() not in oracle_paths
                ):
                    sources.append(path)
        return sorted(sources)

    @classmethod
    def _read_repo_python(
        cls,
        workdir: Path,
        limit: int = 12,
        *,
        trusted_oracle_paths=(),
    ) -> str:
        parts = []
        for path in cls._repo_python_paths(
            workdir,
            trusted_oracle_paths=trusted_oracle_paths,
        )[:limit]:
            rel = path.relative_to(workdir)
            parts.append(f"### {rel}\n{_safe_repo_text(workdir, path)}")
        return "\n\n".join(parts)


def _normalize_oracle_paths(paths) -> frozenset[str]:
    normalized: set[str] = set()
    supplied = getattr(paths, "protected_paths", paths)
    for raw in supplied:
        value = getattr(raw, "path", raw)
        if not isinstance(value, str):
            raise ValueError("trusted oracle paths must be strings")
        path = Path(value)
        posix = path.as_posix()
        if (
            not posix
            or path.is_absolute()
            or "\\" in value
            or "\x00" in value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"unsafe trusted oracle path: {value!r}")
        normalized.add(posix.casefold())
    return frozenset(normalized)


def _without_oracle_context(
    bundle: ContextBundle,
    workdir: Path,
    trusted_oracle_paths,
) -> ContextBundle:
    oracle_paths = _normalize_oracle_paths(trusted_oracle_paths)
    if not oracle_paths:
        return bundle
    visible = [
        item
        for item in bundle.items
        if _context_relative_path(item.provenance.locator, item.provenance.source_root, workdir)
        not in oracle_paths
    ]
    return bundle.model_copy(update={"items": visible})


def _context_relative_path(
    locator: str,
    source_root: str | None,
    workdir: Path,
) -> str:
    raw = locator.split(":", 1)[0]
    candidate = Path(raw)
    roots = [Path(source_root)] if source_root else []
    roots.append(workdir)
    if candidate.is_absolute():
        for root in roots:
            try:
                return candidate.relative_to(root).as_posix().casefold()
            except ValueError:
                continue
        return ""
    posix = candidate.as_posix()
    if (
        not posix
        or "\\" in raw
        or "\x00" in raw
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        return ""
    return posix.casefold()


def _touched_from_diff(diff: str) -> list[str]:
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[len("+++ b/") :].strip())
    return files
