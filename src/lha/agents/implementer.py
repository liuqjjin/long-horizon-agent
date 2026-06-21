"""Implementer: turn a step + context into a Patch via the LLM backend."""

from __future__ import annotations

from pathlib import Path

from ..artifacts import Patch, Step
from ..live_context.models import ContextBundle
from ..llm.base import LLMClient


class Implementer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def implement(self, step: Step, bundle: ContextBundle, workdir: str | Path) -> Patch:
        return self.llm.propose_patch(step, bundle, workdir)
