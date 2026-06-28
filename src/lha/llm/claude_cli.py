"""LLM backend that shells out to the already-authenticated ``claude`` CLI.

Default real backend: no API key to manage. Uses headless ``claude -p``.

``model`` pins a specific model (e.g. a weaker one to calibrate task difficulty for
the verification ablation). ``no_tools`` denies the agentic file/exec/network tools
so the call is a single-shot completion from the prompt alone — the implementer
cannot reach the test oracle on disk, which keeps the ablation leak-free.
"""

from __future__ import annotations

from ..tools.shell import run
from .base import LLMClient

# Tools that would let `claude -p` read the repo (and thus the hidden test oracle),
# run code, or hit the network. Denied when ``no_tools`` is set.
_FILE_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Bash",
    "Edit",
    "Write",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "Task",
]


class ClaudeCLIClient(LLMClient):
    name = "claude_cli"

    def __init__(
        self,
        cli_path: str = "claude",
        timeout: float = 180.0,
        model: str | None = None,
        no_tools: bool = False,
    ):
        self.cli_path = cli_path
        self.timeout = timeout
        self.model = model
        self.no_tools = no_tools

    def complete(self, system: str, prompt: str) -> str:
        # Pipe the prompt via stdin instead of argv: a large-repo prompt on argv
        # overflows ARG_MAX (E2BIG). `claude -p` reads the prompt from stdin.
        cmd = [self.cli_path, "-p", "--append-system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        if self.no_tools:
            cmd += ["--disallowed-tools", *_FILE_TOOLS]
        res = run(cmd, timeout=self.timeout, input=prompt)
        if not res.ok:
            raise RuntimeError(f"claude CLI failed ({res.returncode}): {res.stderr[:400]}")
        return res.stdout
