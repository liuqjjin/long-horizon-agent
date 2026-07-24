"""Terminal-Bench 2 adapter: run the lha harness as a Harbor installed agent.

Harbor evaluates agents it cannot see into: the agent works inside the task
container, Harbor grades the container afterwards with the task's held-out
checks. This adapter installs lha (from a wheel built with ``uv build``) plus
the claude CLI into the container, turns the task instruction into a
``TaskSpec``, runs the verification loop, and copies the workdir back onto the
task directory only when the run finishes DONE — a change that failed
verification never reaches the graded filesystem.

Constraints stated up front:
  - harbor requires Python >= 3.12 while lha supports 3.11, so the harbor
    import happens inside :func:`build_agent`, not at module import time. Run
    evaluations from a 3.12 interpreter (e.g. ``uvx --python 3.12``).
  - The container has no code-search backend, so tasks run with
    ``context_requirement: optional`` — the loop works from the instruction
    and the repo itself.
  - Pinned dataset: ``terminal-bench/terminal-bench-2`` (89 frozen tasks).
    TB 2.1 exists and is newer; 2.0 is pinned here for comparability.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

DATASET = "terminal-bench/terminal-bench-2"

_RESULT_MARK = "__LHA_RESULT__ "
_RUNS = "/tmp/lha_runs"
_WHEEL = "/tmp/lha.whl"
_TASK_YAML = "/tmp/lha_task.yaml"


def task_yaml(instruction: str) -> str:
    """The TaskSpec for one TB instruction (target = the container cwd)."""
    spec = {
        "kind": "issue_to_pr",
        "title": instruction.splitlines()[0][:80] if instruction.strip() else "terminal-bench task",
        "description": instruction,
        "target_repo": ".",
        "context_requirement": "optional",  # no search backend inside the container
        "success": ["pytest passes"],
    }
    import yaml

    return yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)


def parse_result_line(stdout: str) -> dict[str, Any] | None:
    """The machine-readable result of ``lha run --json``, if present."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARK):
            try:
                return json.loads(line[len(_RESULT_MARK) :])
            except json.JSONDecodeError:
                return None
    return None


def install_commands(wheel_target: str = _WHEEL) -> list[str]:
    """Shell commands that install lha + the claude CLI inside the container."""
    return [
        "python3 -m pip install --quiet --break-system-packages "
        f"{shlex.quote(wheel_target)} || python3 -m pip install --quiet {shlex.quote(wheel_target)}",
        # claude CLI: native installer, no npm needed
        "command -v claude >/dev/null 2>&1 || curl -fsSL https://claude.ai/install.sh | bash",
    ]


def run_command(model: str, task_yaml_path: str = _TASK_YAML) -> str:
    """The in-container harness invocation for one task."""
    env = (
        f"LHA_RUNS_DIR={_RUNS} LHA_DATA_DIR=/tmp/lha_data "
        f"LHA_LLM_BACKEND=claude_cli LHA_CLAUDE_MODEL={shlex.quote(model)} "
        "PATH=$HOME/.local/bin:$PATH"
    )
    return f"{env} lha run {shlex.quote(task_yaml_path)} --auto-approve --json"


def build_agent():  # -> type[BaseInstalledAgent]
    """Create the Harbor agent class. Requires the ``harbor`` package (py>=3.12)."""
    try:
        from harbor.agents.installed.base import (  # pyright: ignore[reportMissingImports]
            BaseInstalledAgent,
        )
    except ImportError as e:  # pragma: no cover - exercised via a stubbed harbor in tests
        raise ImportError(
            "harbor is not installed (it needs Python >= 3.12). Run e.g. "
            "uvx --python 3.12 --with harbor --with dist/lha-*.whl "
            "harbor run -d terminal-bench/terminal-bench-2 ..."
        ) from e

    from .. import __version__

    class LhaAgent(BaseInstalledAgent):
        """lha's verification loop driven by the claude CLI, graded by Harbor."""

        @staticmethod
        def name() -> str:
            return "lha"

        def version(self) -> str | None:
            return __version__

        def __init__(
            self,
            logs_dir,
            wheel_path: str | None = None,
            model: str = "claude-haiku-4-5-20251001",
            *args,
            **kwargs,
        ):
            import os

            self._wheel = wheel_path or os.environ.get("LHA_WHEEL")
            self._model = model
            super().__init__(logs_dir, *args, **kwargs)

        async def install(self, environment) -> None:
            if not self._wheel:
                raise RuntimeError(
                    "no lha wheel to install: build one with `uv build` and pass "
                    "wheel_path=... or set LHA_WHEEL"
                )
            await environment.upload_file(self._wheel, _WHEEL)
            for cmd in install_commands():
                res = await environment.exec(command=cmd)
                if res.return_code != 0:
                    raise RuntimeError(f"install step failed ({cmd}): {res.stderr}")

        async def run(self, instruction: str, environment, context) -> None:
            spec = task_yaml(instruction)
            await environment.exec(
                command=f"cat > {_TASK_YAML} <<'LHA_EOF'\n{spec}\nLHA_EOF"
            )
            res = await environment.exec(command=run_command(self._model), timeout_sec=3600)
            result = parse_result_line(res.stdout or "")
            if result is None:
                return  # the run itself failed; leave the filesystem untouched
            self._fill_usage(context, result)
            if result.get("status") == "DONE" and result.get("run_id"):
                # Only a verified result may touch the graded filesystem.
                workdir = f"{_RUNS}/{result['run_id']}/workdir"
                await environment.exec(command=f"cp -a {shlex.quote(workdir)}/. .")

        @staticmethod
        def _fill_usage(context, result: dict[str, Any]) -> None:
            """Best-effort token/cost accounting from the run's LLM trace."""
            usage = result.get("llm_usage") or {}
            for src, dst in (
                ("input_tokens", "n_input_tokens"),
                ("output_tokens", "n_output_tokens"),
                ("cost_usd", "cost_usd"),
            ):
                value = usage.get(src)
                if value is not None and hasattr(context, dst):
                    setattr(context, dst, value)

    return LhaAgent
