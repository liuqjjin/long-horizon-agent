"""LLM backend abstraction.

The Implementer depends only on ``LLMClient``. The walking skeleton uses the
``DeterministicStub`` so a real pytest verifies a real fix with no network. Real
runs can select a CLI or API backend through configuration.
"""

from __future__ import annotations

import json
import re
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
        ctx_text = "\n\n".join(f"# {i.provenance.locator}\n{i.text}" for i in bundle.items[:8])
        files_text = self._read_repo_python(workdir)
        prompt = (
            f"## Issue\n{step.goal}\n\n"
            f"## Prior failures (if any)\n{chr(10).join(step.prior_failures) or 'none'}\n\n"
            f"## Retrieved context\n{ctx_text or 'none'}\n\n"
            f"## Source files\n{files_text}\n\n"
            "Return the complete corrected contents of each file you change, each "
            "preceded by a '### <path>' line, as instructed."
        )
        response = self.complete(_IMPL_SYSTEM, prompt)
        return self._patch_from_response(step, bundle, workdir, response)

    def _patch_from_response(
        self, step: Step, bundle: ContextBundle, workdir: Path, response: str
    ) -> Patch:
        blocks = extract_file_blocks(response) or self._single_block_fallback(workdir, response)
        file_contents: dict[str, str] = {}
        for rel, content in blocks.items():
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                continue  # apply_patch guards too, but never even propose an escape
            target = workdir / rel
            try:  # a header naming a dir or a binary file must skip, not abort the step
                original = target.read_text() if target.is_file() else ""
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
    def _single_block_fallback(cls, workdir: Path, response: str) -> dict[str, str]:
        """If the model returned a lone code block (no '### path'), map it to the only
        non-test source file when there is exactly one — otherwise give up cleanly."""
        m = _CODE_FENCE.search(response)
        if not m:
            return {}
        srcs = [
            p
            for p in sorted(workdir.rglob("*.py"))
            if ".cocoindex_code" not in p.parts and not cls._is_test_file(p.relative_to(workdir))
        ]
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
    def _read_repo_python(cls, workdir: Path, limit: int = 12) -> str:
        parts = []
        for p in sorted(workdir.rglob("*.py")):
            rel = p.relative_to(workdir)
            if ".cocoindex_code" in p.parts or cls._is_test_file(rel):
                continue
            parts.append(f"### {rel}\n{p.read_text()}")
            if len(parts) >= limit:
                break
        return "\n\n".join(parts)


def _touched_from_diff(diff: str) -> list[str]:
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[len("+++ b/") :].strip())
    return files
