"""LLM call accounting: count, time, and cost every model call.

``TracedLLM`` wraps any ``LLMClient``: it enforces a max-calls budget (a run
that would otherwise loop on a broken backend pauses instead of burning money)
and, when bound to a run directory, appends one JSONL record per call to
``llm_trace.jsonl`` — kind, duration, and token/cost usage when the backend
reports it (``last_usage``).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..clock import now
from ..harness.errors import BudgetExceeded
from .base import LLMClient


@dataclass
class LLMUsageTotals:
    calls: int = 0
    wall_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class TracedLLM(LLMClient):
    name = "traced"

    def __init__(self, inner: LLMClient, *, max_calls: int | None = None):
        self.inner = inner
        self.max_calls = max_calls
        self.totals = LLMUsageTotals()
        self._sink: Path | None = None
        self.name = f"traced:{getattr(inner, 'name', type(inner).__name__)}"

    def bind(self, run_dir: str | Path) -> "TracedLLM":
        """Direct per-call records to ``<run_dir>/llm_trace.jsonl``."""
        self._sink = Path(run_dir) / "llm_trace.jsonl"
        return self

    # --- delegation with accounting -----------------------------------------
    def complete(self, system: str, prompt: str) -> str:
        return self._call("complete", lambda: self.inner.complete(system, prompt))

    def propose_patch(self, step, bundle, workdir):
        return self._call("propose_patch", lambda: self.inner.propose_patch(step, bundle, workdir))

    def plan(self, task, template):
        return self._call("plan", lambda: self.inner.plan(task, template))

    def _call(self, kind: str, fn):
        if self.max_calls is not None and self.totals.calls >= self.max_calls:
            raise BudgetExceeded(
                f"max_llm_calls={self.max_calls} exhausted (before another {kind})"
            )
        start = time.monotonic()
        try:
            return fn()
        finally:
            duration = time.monotonic() - start
            self.totals.calls += 1
            self.totals.wall_s += duration
            usage = getattr(self.inner, "last_usage", None)
            if isinstance(usage, dict):
                self.totals.input_tokens += usage.get("input_tokens") or 0
                self.totals.output_tokens += usage.get("output_tokens") or 0
                self.totals.cost_usd += usage.get("cost_usd") or 0.0
            self._record(kind, duration, usage)

    def _record(self, kind: str, duration: float, usage: dict | None) -> None:
        if self._sink is None:
            return
        rec = {
            "at": now().isoformat(),
            "kind": kind,
            "backend": getattr(self.inner, "name", type(self.inner).__name__),
            "duration_s": round(duration, 3),
            "usage": usage,
            "totals": asdict(self.totals),
        }
        try:
            with open(self._sink, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:  # tracing must never take the run down
            pass
