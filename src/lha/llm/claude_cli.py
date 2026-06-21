"""LLM backend that shells out to the already-authenticated ``claude`` CLI.

Default real backend: no API key to manage. Uses headless ``claude -p``.
"""

from __future__ import annotations

from ..tools.shell import run
from .base import LLMClient


class ClaudeCLIClient(LLMClient):
    name = "claude_cli"

    def __init__(self, cli_path: str = "claude", timeout: float = 180.0):
        self.cli_path = cli_path
        self.timeout = timeout

    def complete(self, system: str, prompt: str) -> str:
        res = run(
            [self.cli_path, "-p", prompt, "--append-system-prompt", system],
            timeout=self.timeout,
        )
        if not res.ok:
            raise RuntimeError(f"claude CLI failed ({res.returncode}): {res.stderr[:400]}")
        return res.stdout
