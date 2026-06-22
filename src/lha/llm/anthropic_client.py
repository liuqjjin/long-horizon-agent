"""LLM backend using the Anthropic Python SDK (needs ANTHROPIC_API_KEY).

Install with the ``llm`` extra: ``uv sync --extra llm``. Defaults to Claude
Opus 4.8 with adaptive thinking + high effort — the recommended settings for
coding/agentic work. Orchestration roles can use a cheaper model.
"""

from __future__ import annotations

from .base import LLMClient

# Non-streaming requests above this cap risk SDK HTTP timeouts, so we stream past it.
_STREAM_ABOVE = 16000


class AnthropicResponseError(RuntimeError):
    """The model stopped without a complete answer (max_tokens or refusal).

    Raised so a truncated or refused completion can never be mistaken for a
    finished one — the same fail-closed discipline the verifiers enforce.
    """


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(
        self, model: str = "claude-opus-4-8", max_tokens: int = 16000, effort: str = "high"
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = None

    def _ensure(self):
        if self._client is None:
            from anthropic import Anthropic  # type: ignore  # lazy optional dependency (extra: llm)

            self._client = Anthropic()
        return self._client

    def complete(self, system: str, prompt: str) -> str:
        client = self._ensure()
        # Opus 4.8: adaptive thinking + effort are the recommended coding settings;
        # budget_tokens / temperature are rejected (400) on this model family.
        params: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        if self.max_tokens > _STREAM_ABOVE:
            with client.messages.stream(**params) as stream:
                msg = stream.get_final_message()
        else:
            msg = client.messages.create(**params)

        # Never let a truncated or refused completion read as a complete one.
        if msg.stop_reason == "max_tokens":
            raise AnthropicResponseError(
                f"response hit max_tokens={self.max_tokens}; raise max_tokens or split the task"
            )
        if msg.stop_reason == "refusal":
            category = getattr(getattr(msg, "stop_details", None), "category", None)
            raise AnthropicResponseError(f"model refused the request (category={category})")
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
