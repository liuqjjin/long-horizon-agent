"""Durable human-approval evidence for pause/resume.

``pending_approval.json`` and ``approval.json`` are transient aliases used by
the CLI.  The audit trail lives under the logical attempt:

``steps/<step>/attempts/<attempt>/approval_request.json``
``steps/<step>/attempts/<attempt>/approval_decision.json``

Both files are checksummed envelopes written once.  A decision names the exact
request bytes and reviewed artifact hash, so clearing the aliases after resume
does not erase what was reviewed or how the reviewer answered.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterator, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

from .transaction import attempt_artifact_dir, durable_artifact_write

_Model = TypeVar("_Model", bound=BaseModel)


class ApprovalRequest(BaseModel):
    step_id: str
    attempt_id: str
    goal: str
    summary: str
    artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class ApprovalDecision(BaseModel):
    approved: bool
    outcome: Literal["approved", "rejected"] | None = None
    note: str = ""
    step_id: str | None = None
    attempt_id: str | None = None
    request_ref: str | None = None
    request_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def _outcome_matches_boolean(self) -> "ApprovalDecision":
        expected = "approved" if self.approved else "rejected"
        if self.outcome is None:
            self.outcome = expected
        elif self.outcome != expected:
            raise ValueError("approval outcome does not match approved")
        return self

    def binds(
        self,
        *,
        step_id: str,
        attempt_id: str,
        request_ref: str,
        request_sha256: str,
        artifact_sha256: str | None,
    ) -> bool:
        """Whether this decision answers one exact immutable request."""
        return (
            self.step_id == step_id
            and self.attempt_id == attempt_id
            and self.request_ref == request_ref
            and self.request_sha256 == request_sha256
            and self.artifact_sha256 == artifact_sha256
        )


@dataclass(frozen=True)
class ApprovalEvidence(Generic[_Model]):
    reference: str
    sha256: str
    value: _Model


def approval_request_ref(attempt_id: str) -> str:
    return (Path("attempts") / attempt_id / "approval_request.json").as_posix()


def approval_decision_ref(attempt_id: str) -> str:
    return (Path("attempts") / attempt_id / "approval_decision.json").as_posix()


def approval_request_path(run_dir: Path, step_id: str, attempt_id: str) -> Path:
    return attempt_artifact_dir(run_dir, step_id, attempt_id) / "approval_request.json"


def approval_decision_path(run_dir: Path, step_id: str, attempt_id: str) -> Path:
    return attempt_artifact_dir(run_dir, step_id, attempt_id) / "approval_decision.json"


def read_approval_request(
    run_dir: str | Path,
    step_id: str,
    attempt_id: str,
) -> ApprovalEvidence[ApprovalRequest] | None:
    path = approval_request_path(Path(run_dir), step_id, attempt_id)
    return _read_evidence(
        path,
        ApprovalRequest,
        approval_request_ref(attempt_id),
        required=False,
    )


def read_approval_decision(
    run_dir: str | Path,
    step_id: str,
    attempt_id: str,
) -> ApprovalEvidence[ApprovalDecision] | None:
    path = approval_decision_path(Path(run_dir), step_id, attempt_id)
    return _read_evidence(
        path,
        ApprovalDecision,
        approval_decision_ref(attempt_id),
        required=False,
    )


class HumanApprovalGate:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.pending = self.run_dir / "pending_approval.json"
        self.decision_file = self.run_dir / "approval.json"
        self.lock_file = self.run_dir / ".approval.lock"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"unsafe approval path: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _decision_lock(self) -> Iterator[None]:
        if self.lock_file.is_symlink() or (
            self.lock_file.exists() and not self.lock_file.is_file()
        ):
            raise ValueError(f"unsafe approval lock path: {self.lock_file}")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.lock_file, flags, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError(f"approval lock is not a regular file: {self.lock_file}")
        try:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows fallback
                pass
            yield
        finally:
            os.close(descriptor)

    def request(
        self,
        step,
        attempt_id: str,
        summary: str,
        *,
        artifact_sha256: str | None = None,
    ) -> ApprovalEvidence[ApprovalRequest]:
        """Persist or validate one immutable request, then publish its alias."""
        request = ApprovalRequest(
            step_id=step.step_id,
            attempt_id=attempt_id,
            goal=step.goal,
            summary=summary,
            artifact_sha256=artifact_sha256,
        )
        if self.pending.exists() or self.pending.is_symlink():
            existing = self.pending_request()
            if existing != request:
                raise ValueError(
                    "pending approval request belongs to another attempt"
                )
        if self.decision_file.exists() or self.decision_file.is_symlink():
            if self.decision_file.is_symlink() or not self.decision_file.is_file():
                raise ValueError("approval decision alias is unsafe")
            try:
                alias = ApprovalDecision.model_validate_json(
                    self.decision_file.read_bytes()
                )
            except Exception as error:
                raise ValueError(
                    f"invalid approval decision alias: {error}"
                ) from error
            if alias.step_id != step.step_id or alias.attempt_id != attempt_id:
                raise ValueError(
                    "approval decision belongs to another attempt"
                )
        path = approval_request_path(self.run_dir, step.step_id, attempt_id)
        data = _envelope_bytes(request)
        _write_immutable(path, data)
        evidence = ApprovalEvidence(
            reference=approval_request_ref(attempt_id),
            sha256=_sha256(data),
            value=request,
        )

        # A decision may already be durable after a crash between its fsync and
        # alias cleanup. Do not re-advertise that request as still pending.
        decision_path = approval_decision_path(
            self.run_dir, step.step_id, attempt_id
        )
        if not (decision_path.exists() or decision_path.is_symlink()):
            self._atomic_write(self.pending, request.model_dump_json(indent=2))
        return evidence

    def pending_request(self) -> ApprovalRequest:
        if self.pending.is_symlink() or not self.pending.is_file():
            raise ValueError("no safe pending approval request exists")
        try:
            request = ApprovalRequest.model_validate_json(self.pending.read_bytes())
            evidence = read_approval_request(
                self.run_dir, request.step_id, request.attempt_id
            )
        except Exception as error:
            raise ValueError(f"invalid pending approval request: {error}") from error
        if evidence is None or evidence.value != request:
            raise ValueError(
                "pending approval request does not match immutable evidence"
            )
        return request

    def request_evidence(
        self,
        step_id: str,
        attempt_id: str,
        *,
        validate_alias: bool = True,
    ) -> ApprovalEvidence[ApprovalRequest] | None:
        evidence = read_approval_request(self.run_dir, step_id, attempt_id)
        if validate_alias and (
            self.pending.exists() or self.pending.is_symlink()
        ):
            pending = self.pending_request()
            if pending.step_id != step_id or pending.attempt_id != attempt_id:
                raise ValueError(
                    "pending approval request belongs to another attempt"
                )
            if evidence is None or pending != evidence.value:
                raise ValueError(
                    "pending approval request does not match immutable evidence"
                )
        return evidence

    def decision_evidence(
        self,
        step_id: str,
        attempt_id: str,
        *,
        validate_alias: bool = True,
    ) -> ApprovalEvidence[ApprovalDecision] | None:
        evidence = read_approval_decision(self.run_dir, step_id, attempt_id)
        if validate_alias and (
            self.decision_file.exists() or self.decision_file.is_symlink()
        ):
            if self.decision_file.is_symlink() or not self.decision_file.is_file():
                raise ValueError("approval decision alias is unsafe")
            try:
                alias = ApprovalDecision.model_validate_json(
                    self.decision_file.read_bytes()
                )
            except Exception as error:
                raise ValueError(f"invalid approval decision alias: {error}") from error
            if alias.step_id != step_id or alias.attempt_id != attempt_id:
                raise ValueError("approval decision belongs to another attempt")
            if evidence is None or alias != evidence.value:
                raise ValueError(
                    "approval decision alias does not match immutable evidence"
                )
        return evidence

    def decision(
        self, step_id: str, attempt_id: str
    ) -> ApprovalDecision | None:
        evidence = self.decision_evidence(step_id, attempt_id)
        return evidence.value if evidence is not None else None

    def resolve(
        self,
        approved: bool,
        note: str = "",
    ) -> ApprovalEvidence[ApprovalDecision]:
        """Write one immutable decision while holding the approval lock."""
        with self._decision_lock():
            pending = self.pending_request()
            request = read_approval_request(
                self.run_dir, pending.step_id, pending.attempt_id
            )
            if request is None:
                raise ValueError("immutable approval request is missing")
            reference = approval_decision_ref(pending.attempt_id)
            decision = ApprovalDecision(
                approved=approved,
                outcome="approved" if approved else "rejected",
                note=note,
                step_id=pending.step_id,
                attempt_id=pending.attempt_id,
                request_ref=request.reference,
                request_sha256=request.sha256,
                artifact_sha256=pending.artifact_sha256,
            )
            path = approval_decision_path(
                self.run_dir, pending.step_id, pending.attempt_id
            )
            data = _envelope_bytes(decision)
            _write_immutable(path, data)
            evidence = ApprovalEvidence(
                reference=reference,
                sha256=_sha256(data),
                value=decision,
            )
            self._atomic_write(
                self.decision_file, decision.model_dump_json(indent=2)
            )
            self.pending.unlink()
            _fsync_directory(self.run_dir)
            return evidence

    def clear_transient(self) -> None:
        """Remove aliases only; immutable request and decision remain."""
        self.pending.unlink(missing_ok=True)
        self.decision_file.unlink(missing_ok=True)
        _fsync_directory(self.run_dir)

    # Kept as a narrow compatibility name for callers that clear stale aliases.
    def clear(self) -> None:
        self.clear_transient()


def validate_decision_binding(
    *,
    request: ApprovalEvidence[ApprovalRequest],
    decision: ApprovalEvidence[ApprovalDecision],
    step_id: str,
    attempt_id: str,
    goal: str,
    artifact_sha256: str | None,
) -> None:
    """Validate request identity, then bind the decision to its exact bytes."""
    expected_request_ref = approval_request_ref(attempt_id)
    if (
        request.reference != expected_request_ref
        or request.value.step_id != step_id
        or request.value.attempt_id != attempt_id
        or request.value.goal != goal
        or request.value.artifact_sha256 != artifact_sha256
    ):
        raise ValueError("approval request does not match the current attempt")
    if not decision.value.binds(
        step_id=step_id,
        attempt_id=attempt_id,
        request_ref=expected_request_ref,
        request_sha256=request.sha256,
        artifact_sha256=artifact_sha256,
    ):
        raise ValueError("approval decision does not bind the current request")


def _envelope_bytes(model: BaseModel) -> bytes:
    payload = model.model_dump(mode="json")
    envelope = {
        "schema_version": 1,
        "sha256": _sha256(_canonical(payload)),
        "payload": payload,
    }
    return json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8")


def _read_evidence(
    path: Path,
    model_type: type[_Model],
    reference: str,
    *,
    required: bool,
) -> ApprovalEvidence[_Model] | None:
    _validate_evidence_path(path)
    if not path.exists():
        if required:
            raise ValueError(f"approval evidence is missing: {path}")
        return None
    if not path.is_file():
        raise ValueError(f"approval evidence is not a regular file: {path}")
    try:
        data = path.read_bytes()
        raw = json.loads(data)
        payload = raw["payload"]
        if (
            raw.get("schema_version") != 1
            or raw.get("sha256") != _sha256(_canonical(payload))
        ):
            raise ValueError("checksum mismatch")
        value = model_type.model_validate(payload)
    except Exception as error:
        raise ValueError(f"invalid approval evidence {path}: {error}") from error
    return ApprovalEvidence(reference=reference, sha256=_sha256(data), value=value)


def _write_immutable(path: Path, data: bytes) -> None:
    _validate_evidence_path(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"immutable approval evidence changed: {path}")
        return
    durable_artifact_write(path, data)


def _validate_evidence_path(path: Path) -> None:
    try:
        run_dir = path.parents[4]
        if (
            path.parents[1].name != "attempts"
            or path.parents[3].name != "steps"
        ):
            raise ValueError
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"approval evidence is outside an attempt directory: {path}"
        ) from error
    current = run_dir
    for part in path.relative_to(run_dir).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"approval evidence path contains a symlink: {current}"
            )
        if current != path and current.exists() and not current.is_dir():
            raise ValueError(
                f"approval evidence parent is not a directory: {current}"
            )


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        directory = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:  # pragma: no cover - platform without directory fsync
        pass
