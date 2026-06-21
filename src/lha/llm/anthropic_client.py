"""LLM backend using the Anthropic Python SDK (needs ANTHROPIC_API_KEY).

Install with the ``llm`` extra: ``uv sync --extra llm``. Default model is the
latest Opus for the Implementer; orchestration roles can use a cheaper model.
"""

from __future__ import annotations

from .base import LLMClient


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _ensure(self):
        if self._client is None:
            from anthropic import Anthropic  # type: ignore  # lazy optional dependency (extra: llm)

            self._client = Anthropic()
        return self._client

    def complete(self, system: str, prompt: str) -> str:
        client = self._ensure()
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
