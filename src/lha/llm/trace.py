"""LLM call accounting: count, time, and cost every model call.

``TracedLLM`` wraps any ``LLMClient``: it enforces a max-calls budget (a run
that would otherwise loop on a broken backend pauses instead of burning money)
and, when bound to a run directory, appends one JSONL record per call to
``llm_trace.jsonl`` — kind, duration, and token/cost usage when the backend
reports it (``last_usage``). The call count is durably reserved before control
passes to the backend, so a process crash during a model call cannot reset the
budget on resume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from ..artifacts import Patch, Plan
from ..clock import now
from ..harness.errors import BudgetExceeded, CheckpointCorrupt
from ..harness.transaction import durable_artifact_write
from .base import LLMClient

_USAGE_FILE = "llm_usage.json"
_USAGE_SCHEMA = 1
_CALL_SCHEMA = 1
_CALL_ROOT = "llm_attempts"
_IGNORED_WORKTREE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
T = TypeVar("T")


@dataclass
class LLMUsageTotals:
    calls: int = 0
    wall_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _usage_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointCorrupt(f"invalid durable LLM usage field {label}: {value!r}")
    return value


def _usage_float(value, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise CheckpointCorrupt(f"invalid durable LLM usage field {label}: {value!r}")
    return float(value)


def _validated_totals(value) -> LLMUsageTotals:
    raw = value if isinstance(value, dict) else {
        "calls": getattr(value, "calls", 0),
        "wall_s": getattr(value, "wall_s", 0.0),
        "input_tokens": getattr(value, "input_tokens", 0),
        "output_tokens": getattr(value, "output_tokens", 0),
        "cost_usd": getattr(value, "cost_usd", 0.0),
    }
    return LLMUsageTotals(
        calls=_usage_int(raw.get("calls"), "calls"),
        wall_s=_usage_float(raw.get("wall_s"), "wall_s"),
        input_tokens=_usage_int(raw.get("input_tokens"), "input_tokens"),
        output_tokens=_usage_int(raw.get("output_tokens"), "output_tokens"),
        cost_usd=_usage_float(raw.get("cost_usd"), "cost_usd"),
    )


def load_usage_checkpoint(run_dir: str | Path) -> LLMUsageTotals | None:
    """Read the per-call write-ahead usage checkpoint."""
    path = Path(run_dir) / _USAGE_FILE
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CheckpointCorrupt(f"LLM usage checkpoint path is unsafe: {path}")
    if not path.exists():
        return None
    try:
        envelope = json.loads(path.read_text())
        if envelope.get("schema_version") != _USAGE_SCHEMA:
            raise ValueError("unsupported schema")
        payload = envelope["payload"]
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        if digest != envelope["sha256"]:
            raise ValueError("checksum mismatch")
        return _validated_totals(payload)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise CheckpointCorrupt(f"invalid LLM usage checkpoint {path}: {error}") from error


def _save_usage_checkpoint(run_dir: Path, totals: LLMUsageTotals) -> None:
    path = run_dir / _USAGE_FILE
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CheckpointCorrupt(f"LLM usage checkpoint path is unsafe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(_validated_totals(totals))
    envelope = {
        "schema_version": _USAGE_SCHEMA,
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(envelope, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:  # pragma: no cover - platform without directory fsync
            pass
    finally:
        temporary.unlink(missing_ok=True)


class TracedLLM(LLMClient):
    name = "traced"

    def __init__(self, inner: LLMClient, *, max_calls: int | None = None):
        self.inner = inner
        self.max_calls = max_calls
        self.totals = LLMUsageTotals()
        self._sink: Path | None = None
        self._next_call_context: dict[str, Any] = {}
        self.name = f"traced:{getattr(inner, 'name', type(inner).__name__)}"

    def bind(self, run_dir: str | Path) -> "TracedLLM":
        """Direct per-call records to ``<run_dir>/llm_trace.jsonl``."""
        self._sink = Path(run_dir) / "llm_trace.jsonl"
        return self

    def restore_totals(self, totals) -> None:
        """Reconcile step state with the per-call write-ahead checkpoint."""
        state_totals = _validated_totals(totals)
        durable = (
            load_usage_checkpoint(self._sink.parent)
            if self._sink is not None
            else None
        )
        if durable is None or state_totals.calls > durable.calls:
            selected = state_totals
        elif durable.calls > state_totals.calls:
            # The model call completed and was accounted for, but the process
            # died before the next RunState save.
            selected = durable
        elif durable != state_totals:
            raise CheckpointCorrupt(
                "RunState and the LLM usage checkpoint disagree at the same call count"
            )
        else:
            selected = state_totals
        self.totals = selected
        if self._sink is not None and durable != selected:
            _save_usage_checkpoint(self._sink.parent, selected)

    def set_call_context(self, **context: Any) -> None:
        """Bind the next plan/patch result to its durable run attempt.

        The context is consumed by the next journaled call.  Keeping it outside
        the public backend interface avoids weakening compatibility with custom
        LLM clients while still binding a proposal to its run and attempt.
        """
        self._next_call_context = context

    # --- delegation with accounting -----------------------------------------
    def complete(self, system: str, prompt: str) -> str:
        return self._call("complete", lambda: self.inner.complete(system, prompt))

    def propose_patch(self, step, bundle, workdir):
        payload = {
            "context": self._consume_call_context(),
            "backend": self._backend_identity(),
            "step": step.model_dump(mode="json"),
            "bundle": _semantic_bundle(bundle),
            "worktree_sha256": _worktree_sha256(Path(workdir)),
        }
        return self._journaled_call(
            "propose_patch",
            payload,
            lambda: self.inner.propose_patch(step, bundle, workdir),
            encode=lambda value: {
                "type": "Patch",
                "value": value.model_dump(mode="json"),
            },
            decode=_decode_patch,
        )

    def plan(self, task, template):
        payload = {
            "context": self._consume_call_context(),
            "backend": self._backend_identity(),
            "task": task.model_dump(mode="json"),
            "template": template.model_dump(mode="json"),
        }
        return self._journaled_call(
            "plan",
            payload,
            lambda: self.inner.plan(task, template),
            encode=lambda value: {
                "type": "Plan" if value is not None else "None",
                "value": value.model_dump(mode="json") if value is not None else None,
            },
            decode=_decode_plan,
        )

    def _consume_call_context(self) -> dict[str, Any]:
        value = self._next_call_context
        self._next_call_context = {}
        return value

    def _backend_identity(self) -> dict[str, Any]:
        """Record stable public model settings without serializing credentials."""
        identity: dict[str, Any] = {
            "name": getattr(self.inner, "name", type(self.inner).__name__),
        }
        for name in ("model", "reasoning_effort", "sandbox_mode"):
            value = getattr(self.inner, name, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                identity[name] = value
        return identity

    def _journaled_call(
        self,
        kind: str,
        input_payload: dict[str, Any],
        fn: Callable[[], T],
        *,
        encode: Callable[[T], dict[str, Any]],
        decode: Callable[[dict[str, Any]], T],
    ) -> T:
        """Write intent before a paid call and its typed result before returning.

        A completed result is reused after a crash without a second call.  An
        intent with no result is ambiguous: the backend may have acted, so the
        same logical attempt fails closed instead of being submitted again.
        """
        if self._sink is None:
            return self._call(kind, fn)
        canonical_input = _canonical(input_payload)
        input_sha256 = hashlib.sha256(canonical_input).hexdigest()
        root = self._sink.parent / _CALL_ROOT
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise CheckpointCorrupt(f"LLM attempt journal root is unsafe: {root}")
        context = input_payload.get("context")
        logical_attempt = (
            str(context.get("attempt_id"))
            if isinstance(context, dict) and context.get("attempt_id")
            else "unbound"
        )
        safe_attempt = re.sub(
            r"[^A-Za-z0-9_.-]", "_", logical_attempt
        ).strip(".") or "unbound"
        if safe_attempt != logical_attempt:
            raise CheckpointCorrupt(
                f"unsafe LLM logical attempt id: {logical_attempt!r}"
            )
        kind_root = root / kind
        if kind_root.is_symlink() or (
            kind_root.exists() and not kind_root.is_dir()
        ):
            raise CheckpointCorrupt(
                f"LLM call-kind journal path is unsafe: {kind_root}"
            )
        attempt_dir = kind_root / safe_attempt
        if attempt_dir.is_symlink() or (
            attempt_dir.exists() and not attempt_dir.is_dir()
        ):
            raise CheckpointCorrupt(
                f"LLM attempt journal path is unsafe: {attempt_dir}"
            )
        intent = {
            "schema_version": _CALL_SCHEMA,
            "kind": kind,
            "logical_attempt_id": logical_attempt,
            "input_sha256": input_sha256,
            "input": input_payload,
        }
        intent_bytes = _canonical(intent)
        intent_path = attempt_dir / "intent.json"
        result_path = attempt_dir / "result.json"

        if result_path.exists() or result_path.is_symlink():
            _require_exact_file(intent_path, intent_bytes, "LLM call intent")
            result = _load_call_result(
                result_path, kind=kind, input_sha256=input_sha256
            )
            return decode(result)
        if intent_path.exists() or intent_path.is_symlink():
            _require_exact_file(intent_path, intent_bytes, "LLM call intent")
            raise CheckpointCorrupt(
                f"LLM {kind} attempt {input_sha256[:12]} has durable intent "
                "but no committed result; refusing an ambiguous replay"
            )

        self._check_call_budget(kind)
        _write_once(intent_path, intent_bytes)
        value = self._call(kind, fn)
        encoded = encode(value)
        result_payload = {
            "schema_version": _CALL_SCHEMA,
            "kind": kind,
            "input_sha256": input_sha256,
            "result_sha256": hashlib.sha256(_canonical(encoded)).hexdigest(),
            "result": encoded,
        }
        _write_once(result_path, _canonical(result_payload))
        return value

    def _check_call_budget(self, kind: str) -> None:
        if self.max_calls is not None and self.totals.calls >= self.max_calls:
            raise BudgetExceeded(
                f"max_llm_calls={self.max_calls} exhausted (before another {kind})"
            )

    def _call(self, kind: str, fn):
        self._check_call_budget(kind)
        # Reserve the call before invoking a process or network backend. A hard
        # crash cannot run ``finally``; persisting here deliberately counts an
        # uncertain call as consumed instead of allowing it to be replayed for
        # free after resume.
        self.totals.calls += 1
        if self._sink is not None:
            _save_usage_checkpoint(self._sink.parent, self.totals)
        start = time.monotonic()
        try:
            return fn()
        finally:
            duration = time.monotonic() - start
            self.totals.wall_s += duration
            usage = getattr(self.inner, "last_usage", None)
            if isinstance(usage, dict):
                self.totals.input_tokens += _non_negative_int(
                    usage.get("input_tokens")
                )
                self.totals.output_tokens += _non_negative_int(
                    usage.get("output_tokens")
                )
                self.totals.cost_usd += _non_negative_float(
                    usage.get("cost_usd")
                )
            if self._sink is not None:
                # This is the accounting commit point. It precedes the optional
                # detail trace and the coarser RunState checkpoint.
                _save_usage_checkpoint(self._sink.parent, self.totals)
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
        call_meta = getattr(self.inner, "last_call", None)
        if isinstance(call_meta, dict):
            rec["call"] = call_meta
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self._sink, flags, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                return
            with os.fdopen(descriptor, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:  # tracing must never take the run down
            pass


def _non_negative_int(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _non_negative_float(value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        return 0.0
    return float(value)


def _write_once(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise CheckpointCorrupt(f"LLM attempt artifact is a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise CheckpointCorrupt(f"LLM attempt artifact changed: {path}")
        return
    durable_artifact_write(path, data)


def _require_exact_file(path: Path, expected: bytes, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise CheckpointCorrupt(f"{label} is missing or unsafe: {path}")
    if path.read_bytes() != expected:
        raise CheckpointCorrupt(f"{label} does not match the current input: {path}")


def _load_call_result(
    path: Path,
    *,
    kind: str,
    input_sha256: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CheckpointCorrupt(f"LLM call result is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_bytes())
        encoded = value["result"]
        if (
            value.get("schema_version") != _CALL_SCHEMA
            or value.get("kind") != kind
            or value.get("input_sha256") != input_sha256
            or not isinstance(encoded, dict)
            or value.get("result_sha256")
            != hashlib.sha256(_canonical(encoded)).hexdigest()
        ):
            raise ValueError("identity or checksum mismatch")
        return encoded
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise CheckpointCorrupt(f"invalid LLM call result {path}: {error}") from error


def _decode_patch(value: dict[str, Any]) -> Patch:
    if value.get("type") != "Patch":
        raise CheckpointCorrupt("journaled propose_patch result has the wrong type")
    try:
        return Patch.model_validate(value.get("value"))
    except Exception as error:
        raise CheckpointCorrupt(
            f"journaled propose_patch result is invalid: {error}"
        ) from error


def _decode_plan(value: dict[str, Any]) -> Plan | None:
    if value.get("type") == "None" and value.get("value") is None:
        return None
    if value.get("type") != "Plan":
        raise CheckpointCorrupt("journaled plan result has the wrong type")
    try:
        return Plan.model_validate(value.get("value"))
    except Exception as error:
        raise CheckpointCorrupt(f"journaled plan result is invalid: {error}") from error


def _semantic_bundle(bundle) -> dict[str, Any]:
    """Drop observation timestamps while retaining all context-bearing fields."""
    value = bundle.model_dump(mode="json")
    freshness = dict(value.get("freshness") or {})
    freshness.pop("indexed_at", None)
    freshness.pop("source_mtime_max", None)
    value["freshness"] = freshness
    items = []
    for item in value.get("items", []):
        item = dict(item)
        provenance = dict(item.get("provenance") or {})
        provenance.pop("indexed_at", None)
        item["provenance"] = provenance
        items.append(item)
    value["items"] = items
    return value


def _worktree_sha256(root: Path) -> str:
    """Hash the regular-file view available to a model-authored patch call."""
    if root.is_symlink() or not root.is_dir():
        raise CheckpointCorrupt(f"LLM patch worktree is missing or unsafe: {root}")
    digest = hashlib.sha256()
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name not in _IGNORED_WORKTREE_DIRS
        )
        current_path = Path(current)
        for name in sorted(filenames):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            try:
                mode = path.lstat().st_mode
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                if stat.S_ISLNK(mode):
                    digest.update(b"link\0")
                    digest.update(os.readlink(path).encode("utf-8"))
                elif stat.S_ISREG(mode):
                    digest.update(b"file\0")
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                else:
                    digest.update(f"mode:{mode}".encode())
                digest.update(b"\0")
            except OSError as error:
                raise CheckpointCorrupt(
                    f"cannot bind LLM patch input to {relative}: {error}"
                ) from error
    return digest.hexdigest()
