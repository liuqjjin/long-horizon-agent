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
    if backend == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model=config.anthropic_model_impl)
    raise ValueError(f"unknown LLM backend: {backend!r}")


__all__ = ["LLMClient", "DeterministicStub", "get_llm"]
