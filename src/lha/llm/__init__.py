"""LLM backends + a factory selecting one from config."""

from __future__ import annotations

from ..config import Config
from .base import LLMClient
from .stub import DeterministicStub


def get_llm(config: Config) -> LLMClient:
    backend = config.llm_backend
    if backend == "stub":
        return DeterministicStub()
    if backend == "claude_cli":
        from .claude_cli import ClaudeCLIClient

        return ClaudeCLIClient(
            cli_path=config.claude_cli_path, model=config.claude_cli_model or None
        )
    if backend == "codex_cli":
        from .codex_cli import CodexCLIClient

        return CodexCLIClient(
            cli_path=config.codex_cli_path,
            model=config.codex_model or None,
            reasoning_effort=config.codex_reasoning_effort,
            sandbox_mode=config.codex_sandbox,
            externally_sandboxed=config.codex_external_sandbox,
            max_retries=config.codex_max_retries,
            retry_backoff_s=config.codex_retry_backoff_s,
        )
    if backend == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model=config.anthropic_model_impl)
    raise ValueError(f"unknown LLM backend: {backend!r}")


__all__ = ["LLMClient", "DeterministicStub", "get_llm"]
