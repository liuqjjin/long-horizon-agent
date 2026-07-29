"""Failure isolation and legacy-tail recovery for the diagnostic LLM trace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lha.harness.state import LLMUsageState
from lha.llm.base import LLMClient
from lha.llm.trace import TracedLLM, load_usage_checkpoint


class _MetadataLLM(LLMClient):
    name = "metadata"

    def __init__(self, usage: dict | None = None, call: dict | None = None) -> None:
        self._usage = usage
        self._call = call

    def complete(self, system: str, prompt: str) -> str:
        self.last_usage = self._usage
        self.last_call = self._call
        return "completed"


class _FailingMetadataLLM(_MetadataLLM):
    def complete(self, system: str, prompt: str) -> str:
        raise RuntimeError("credential-like detail must not enter the trace")


def _bound(
    tmp_path: Path,
    *,
    usage: dict | None = None,
    call: dict | None = None,
) -> TracedLLM:
    traced = TracedLLM(_MetadataLLM(usage, call)).bind(tmp_path)
    traced.restore_totals(LLMUsageState())
    return traced


def test_trace_discards_only_a_torn_final_fragment_before_append(
    tmp_path: Path,
) -> None:
    complete = {"kind": "old", "usage": None}
    trace = tmp_path / "llm_trace.jsonl"
    trace.write_bytes(json.dumps(complete).encode() + b'\n{"kind":"torn"')

    traced = _bound(tmp_path)
    assert traced.complete("system", "prompt") == "completed"

    lines = trace.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == complete
    assert json.loads(lines[1])["kind"] == "complete"
    assert all("torn" not in line for line in lines)


def test_trace_preserves_a_complete_legacy_record_without_final_newline(
    tmp_path: Path,
) -> None:
    complete = {"kind": "old", "usage": None}
    trace = tmp_path / "llm_trace.jsonl"
    trace.write_text(json.dumps(complete))

    traced = _bound(tmp_path)
    assert traced.complete("system", "prompt") == "completed"

    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert records[0] == complete
    assert records[1]["kind"] == "complete"


@pytest.mark.parametrize(
    "metadata",
    [
        {"opaque": object()},
        pytest.param({}, id="circular"),
    ],
)
def test_unserializable_trace_metadata_cannot_fail_a_completed_call(
    tmp_path: Path,
    metadata: dict,
) -> None:
    if not metadata:
        metadata["self"] = metadata
    usage = {
        "input_tokens": 7,
        "output_tokens": 3,
        "cost_usd": 0.25,
        **metadata,
    }
    traced = _bound(tmp_path, usage=usage)

    assert traced.complete("system", "prompt") == "completed"
    assert traced.totals.calls == 1
    assert traced.totals.input_tokens == 7
    assert traced.totals.output_tokens == 3
    assert traced.totals.cost_usd == 0.25
    assert not (tmp_path / "llm_trace.jsonl").exists()

    checkpoint = load_usage_checkpoint(tmp_path)
    assert checkpoint == traced.totals


def test_trace_helper_failure_cannot_fail_a_completed_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traced = _bound(tmp_path)

    def fail_timestamp():
        raise TypeError("diagnostic timestamp failed")

    monkeypatch.setattr("lha.llm.trace.now", fail_timestamp)
    assert traced.complete("system", "prompt") == "completed"
    assert traced.totals.calls == 1
    assert load_usage_checkpoint(tmp_path) == traced.totals
    assert not (tmp_path / "llm_trace.jsonl").exists()


def test_failed_call_trace_records_only_the_exception_type(
    tmp_path: Path,
) -> None:
    traced = TracedLLM(_FailingMetadataLLM()).bind(tmp_path)
    traced.restore_totals(LLMUsageState())

    with pytest.raises(RuntimeError, match="credential-like"):
        traced.complete("system", "prompt")

    record = json.loads((tmp_path / "llm_trace.jsonl").read_text())
    assert record["outcome"] == "error"
    assert record["error_type"] == "RuntimeError"
    assert "credential-like" not in json.dumps(record)
