"""Append-only registration records for formal ablation attempts.

The registry prevents a formal run from moving to a fresh output directory
without leaving a committed record.  It does not claim that private forks or
deleted Git history never existed; it constrains the official runner and the
evidence accepted by the release check.
"""

from __future__ import annotations

import fcntl
import hashlib
import ipaddress
import json
import os
import re
import stat
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Iterator, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

FORMAL_ABLATION_ATTEMPTS_PATH = PurePosixPath(
    "benchmarks/formal_ablation_attempts.json"
)
FORMAL_ABLATION_ATTEMPTS_SCHEMA = 2
FORMAL_ABLATION_PROTOCOL_SCHEMA = 2
MAX_FORMAL_ABLATION_ATTEMPTS_BYTES = 1024 * 1024

_HEX_40_PATTERN = r"^[0-9a-f]{40}$"
_HEX_64_PATTERN = r"^[0-9a-f]{64}$"
_DOCKER_IMAGE_ID_PATTERN = r"^sha256:[0-9a-f]{64}$"
_WITNESS_REMOTE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_PUBLIC_DNS_NAME_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)

FORMAL_ATTEMPT_WITNESS_REF_PREFIX = "refs/heads/formal-attempts"
FORMAL_ATTEMPT_WITNESS_SCHEMA = 1
FORMAL_ATTEMPT_WITNESS_IDENTITY = (
    "liuqjjin <156533013+liuqjjin@users.noreply.github.com>"
)
FORMAL_ATTEMPT_WITNESS_GIT_TIME = "946684800 +0000"
FORMAL_ATTEMPT_LOCK_NAME = "lha-formal-attempt.lock"


def _formal_git_directory(repository_root: Path) -> Path:
    marker = repository_root / ".git"
    metadata = marker.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise OSError("formal attempt Git metadata path is a symlink")
    if stat.S_ISDIR(metadata.st_mode):
        descriptor = os.open(
            marker,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            named = marker.lstat()
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (metadata.st_dev, metadata.st_ino)
                or (named.st_dev, named.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise OSError("formal attempt Git metadata directory changed")
            git_directory = marker.resolve(strict=True)
            resolved = git_directory.lstat()
            if (resolved.st_dev, resolved.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise OSError("formal attempt Git metadata directory changed")
        finally:
            os.close(descriptor)
    elif (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and metadata.st_uid == os.geteuid()
        and not stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        if metadata.st_size <= 0 or metadata.st_size > 4096:
            raise OSError("formal attempt Git metadata file is invalid")
        descriptor = os.open(
            marker,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened_before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_before.st_mode)
                or opened_before.st_nlink != 1
                or opened_before.st_uid != os.geteuid()
                or stat.S_IMODE(opened_before.st_mode) & 0o022
                or (opened_before.st_dev, opened_before.st_ino)
                != (metadata.st_dev, metadata.st_ino)
            ):
                raise OSError("formal attempt Git metadata file changed")
            chunks: list[bytes] = []
            remaining = 4097
            while remaining:
                chunk = os.read(descriptor, min(remaining, 4096))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            opened_after = os.fstat(descriptor)
            named_after = marker.lstat()
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if (
                len(payload) > 4096
                or any(
                    getattr(opened_before, field)
                    != getattr(opened_after, field)
                    for field in stable_fields
                )
                or (named_after.st_dev, named_after.st_ino)
                != (opened_after.st_dev, opened_after.st_ino)
            ):
                raise OSError("formal attempt Git metadata file changed")
        finally:
            os.close(descriptor)
        try:
            value = payload.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise OSError("formal attempt Git metadata file is invalid") from error
        if not value.startswith("gitdir: "):
            raise OSError("formal attempt Git metadata file is invalid")
        configured = Path(value.removeprefix("gitdir: "))
        git_directory = (
            configured
            if configured.is_absolute()
            else repository_root / configured
        )
    else:
        raise OSError("formal attempt Git metadata path is unsafe")
    git_directory = git_directory.resolve(strict=True)
    named = git_directory.lstat()
    if (
        stat.S_ISLNK(named.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or named.st_uid != os.geteuid()
        or stat.S_IMODE(named.st_mode) & 0o022
    ):
        raise OSError("formal attempt Git directory is unsafe")
    return git_directory


@contextmanager
def formal_attempt_lock(
    repository_root: str | Path,
    *,
    blocking: bool = False,
) -> Iterator[None]:
    """Serialize runner and lifecycle commands above any replaceable output."""
    repository = Path(repository_root).resolve(strict=True)
    git_directory = _formal_git_directory(repository)
    expected_directory = git_directory.lstat()
    directory_descriptor = os.open(
        git_directory,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened_directory = os.fstat(directory_descriptor)
    named_directory = git_directory.lstat()
    if (
        not stat.S_ISDIR(opened_directory.st_mode)
        or (opened_directory.st_dev, opened_directory.st_ino)
        != (expected_directory.st_dev, expected_directory.st_ino)
        or (named_directory.st_dev, named_directory.st_ino)
        != (opened_directory.st_dev, opened_directory.st_ino)
    ):
        os.close(directory_descriptor)
        raise OSError("formal attempt Git directory changed before locking")
    descriptor: int | None = None
    locked = False
    try:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(
            FORMAL_ATTEMPT_LOCK_NAME,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        opened = os.fstat(descriptor)
        named = os.stat(
            FORMAL_ATTEMPT_LOCK_NAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) & 0o077
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise OSError("formal attempt lock is unsafe")
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.fsync(directory_descriptor)
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, operation)
        except (BlockingIOError, OSError) as error:
            raise RuntimeError(
                "another formal attempt command or runner is active"
            ) from error
        locked = True
        opened_after = os.fstat(descriptor)
        named_after = os.stat(
            FORMAL_ATTEMPT_LOCK_NAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        confirmed_git_directory = _formal_git_directory(repository)
        named_directory_after = git_directory.lstat()
        if (
            (opened_after.st_dev, opened_after.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (named_after.st_dev, named_after.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (
                named_directory_after.st_dev,
                named_directory_after.st_ino,
            )
            != (opened_directory.st_dev, opened_directory.st_ino)
            or confirmed_git_directory != git_directory
        ):
            raise OSError("formal attempt lock name changed during acquisition")
        yield
    finally:
        if descriptor is not None:
            if locked:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(descriptor)
        os.close(directory_descriptor)


def _validate_timestamp(value: str) -> str:
    if not value or value.strip() != value:
        raise ValueError("attempt timestamp must be a non-empty ISO-8601 value")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("attempt timestamp must be valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("attempt timestamp must include a UTC offset")
    return value


def _validate_output_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or value == "."
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("formal ablation output_path must be a fixed repository path")
    return value


def _validate_plain_text(value: str) -> str:
    if not value or value.strip() != value or "\x00" in value:
        raise ValueError("formal ablation text field is invalid")
    return value


def _validate_witness_remote_url(value: str) -> str:
    """Accept only canonical public HTTPS URLs suitable for anonymous reads."""
    if (
        not value
        or value.strip() != value
        or len(value.encode("utf-8")) > 2048
        or value.startswith("-")
        or "\\" in value
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ValueError("formal ablation witness remote URL is invalid")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("formal ablation witness remote URL is invalid") from error
    host = parsed.hostname
    if (
        parsed.scheme != "https"
        or not host
        or host != host.lower()
        or parsed.netloc != host
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path
        or parsed.path == "/"
        or not parsed.path.startswith("/")
        or PurePosixPath(parsed.path).as_posix() != parsed.path
        or "%" in parsed.path
        or any(
            part in {"", ".", ".."}
            for part in PurePosixPath(parsed.path).parts[1:]
        )
    ):
        raise ValueError("formal ablation witness remote URL is invalid")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if (
            _PUBLIC_DNS_NAME_PATTERN.fullmatch(host) is None
            or host.endswith(
                (".local", ".localhost", ".internal", ".invalid", ".test")
            )
        ):
            raise ValueError(
                "formal ablation witness remote URL must use a public host"
            )
    else:
        if not address.is_global:
            raise ValueError(
                "formal ablation witness remote URL must use a public host"
            )
    return value


def validate_formal_witness_remote_url(value: str) -> str:
    """Validate a witness URL at runtime as well as at model boundaries."""
    return _validate_witness_remote_url(value)


def formal_ablation_witness_ref(attempt_id: str) -> str:
    """Derive the only remote ref that may consume an attempt."""
    if re.fullmatch(_HEX_64_PATTERN, attempt_id) is None:
        raise ValueError("formal ablation witness attempt_id is invalid")
    return f"{FORMAL_ATTEMPT_WITNESS_REF_PREFIX}/{attempt_id}"


def formal_ablation_witness_message(
    *,
    attempt_id: str,
    registration_registry_sha256: str,
    protocol_sha256: str,
    outcome_key: str,
    run_header_sha256: str,
) -> bytes:
    """Return the canonical, secret-free commit message for one start witness."""
    fields = {
        "schema_version": FORMAL_ATTEMPT_WITNESS_SCHEMA,
        "formal_attempt_id": attempt_id,
        "registration_registry_sha256": registration_registry_sha256,
        "protocol_sha256": protocol_sha256,
        "outcome_key": outcome_key,
        "run_header_sha256": run_header_sha256,
    }
    for name, value in fields.items():
        if name == "schema_version":
            continue
        if not isinstance(value, str) or re.fullmatch(_HEX_64_PATTERN, value) is None:
            raise ValueError(f"formal ablation witness {name} is invalid")
    payload = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"LHA formal ablation attempt witness\n\n{payload}\n".encode("ascii")


def formal_ablation_witness_commit_bytes(
    *,
    tree: str,
    parent: str,
    message: bytes,
) -> bytes:
    """Build the exact deterministic Git commit object content for a witness."""
    if re.fullmatch(_HEX_40_PATTERN, tree) is None:
        raise ValueError("formal ablation witness tree is invalid")
    if re.fullmatch(_HEX_40_PATTERN, parent) is None:
        raise ValueError("formal ablation witness parent is invalid")
    if (
        not message
        or b"\x00" in message
        or not message.endswith(b"\n")
        or message != message.decode("ascii").encode("ascii")
    ):
        raise ValueError("formal ablation witness message is invalid")
    identity = FORMAL_ATTEMPT_WITNESS_IDENTITY
    timestamp = FORMAL_ATTEMPT_WITNESS_GIT_TIME
    header = (
        f"tree {tree}\n"
        f"parent {parent}\n"
        f"author {identity} {timestamp}\n"
        f"committer {identity} {timestamp}\n"
        "\n"
    ).encode("ascii")
    return header + message


def formal_ablation_witness_commit_oid(payload: bytes) -> str:
    """Compute the SHA-1 object ID used by the repository's 40-hex protocol."""
    object_bytes = f"commit {len(payload)}\0".encode("ascii") + payload
    return hashlib.sha1(object_bytes, usedforsecurity=False).hexdigest()


class FormalCodexClientConfig(BaseModel):
    """Outcome-affecting Codex client settings fixed before registration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    no_tools: Literal[True] = True
    sandbox_mode: Literal["read-only"] = "read-only"
    permission_model: Literal["profile"] = "profile"
    permission_profile: Literal["lha-read"] = "lha-read"
    credential_barrier: Literal["verified"] = "verified"
    externally_sandboxed: Literal[False] = False
    max_retries: int = Field(ge=0, strict=True)
    timeout_s: float = Field(gt=0, allow_inf_nan=False, strict=True)
    retry_backoff_s: float = Field(
        ge=0,
        allow_inf_nan=False,
        strict=True,
    )


class FormalGitCredentialHelper(BaseModel):
    """Pinned GitHub CLI bytes used only for the authenticated witness push."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str
    executable_path: str
    executable_sha256: str = Field(pattern=_HEX_64_PATTERN)
    version: str
    command: str

    _version = field_validator("version")(_validate_plain_text)

    @model_validator(mode="after")
    def _binding_is_canonical(self) -> "FormalGitCredentialHelper":
        path = Path(self.executable_path)
        if (
            not path.is_absolute()
            or str(path) != self.executable_path
            or self.executable_path.strip() != self.executable_path
            or "\x00" in self.executable_path
            or any(character.isspace() for character in self.executable_path)
        ):
            raise ValueError(
                "formal Git credential helper executable path is invalid"
            )
        if (
            self.host != self.host.lower()
            or _PUBLIC_DNS_NAME_PATTERN.fullmatch(self.host) is None
            or self.host.endswith(
                (".local", ".localhost", ".internal", ".invalid", ".test")
            )
        ):
            raise ValueError("formal Git credential helper host is invalid")
        if self.command != f"!{self.executable_path} auth git-credential":
            raise ValueError("formal Git credential helper command is invalid")
        return self


def make_formal_codex_client(
    *,
    cli_path: str,
    model: str,
    reasoning_effort: str,
) -> Any:
    """Construct the only Codex client accepted by a formal ablation."""
    from .llm.codex_cli import CodexCLIClient

    return CodexCLIClient(
        cli_path=cli_path,
        timeout=300.0,
        model=model,
        reasoning_effort=reasoning_effort,
        no_tools=True,
        sandbox_mode="read-only",
        externally_sandboxed=False,
        max_retries=2,
        retry_backoff_s=1.0,
    )


def formal_codex_client_config_from_runtime(
    client: Any,
) -> FormalCodexClientConfig:
    """Resolve the exact client fields consumed by the formal runner."""
    try:
        return FormalCodexClientConfig(
            no_tools=client.no_tools,
            sandbox_mode=client.sandbox_mode,
            permission_model=client.permission_model,
            permission_profile=client.permission_profile,
            credential_barrier=client.credential_barrier,
            externally_sandboxed=client.externally_sandboxed,
            max_retries=client.max_retries,
            timeout_s=float(client.timeout),
            retry_backoff_s=float(client.retry_backoff_s),
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise ValueError(
            "formal Codex runtime does not match the fixed client protocol"
        ) from error


def formal_codex_client_sha256(config: FormalCodexClientConfig) -> str:
    """Hash the canonical outcome-affecting Codex client configuration."""
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class FormalAblationProtocol(BaseModel):
    """Fields that select the code, corpus, model, and execution image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Omitted version means the historical schema so existing schema-2
    # registry events retain their original canonical protocol bytes.
    schema_version: Literal[1, 2] = 1
    source_commit: str = Field(pattern=_HEX_40_PATTERN)
    source_tree_sha256: str = Field(pattern=_HEX_64_PATTERN)
    manifest_sha256: str = Field(pattern=_HEX_64_PATTERN)
    model: str
    reasoning_effort: str
    docker_image_id: str = Field(pattern=_DOCKER_IMAGE_ID_PATTERN)
    scorer_runtime_sha256: str | None = Field(
        default=None,
        pattern=_HEX_64_PATTERN,
    )
    codex_cli_version: str
    codex_cli_executable_sha256: str = Field(pattern=_HEX_64_PATTERN)
    codex_client: FormalCodexClientConfig
    codex_client_sha256: str = Field(pattern=_HEX_64_PATTERN)
    witness_credential_helper: FormalGitCredentialHelper | None = None

    _model = field_validator("model")(_validate_plain_text)
    _effort = field_validator("reasoning_effort")(_validate_plain_text)
    _cli_version = field_validator("codex_cli_version")(_validate_plain_text)

    @model_validator(mode="after")
    def _client_digest_matches(self) -> "FormalAblationProtocol":
        if (self.schema_version == 1) != (
            self.scorer_runtime_sha256 is None
        ):
            raise ValueError(
                "formal ablation protocol schema 1 omits scorer_runtime_sha256 "
                "and schema 2 requires it"
            )
        if self.codex_client_sha256 != formal_codex_client_sha256(
            self.codex_client
        ):
            raise ValueError(
                "formal ablation codex_client_sha256 does not match codex_client"
            )
        return self


def formal_ablation_protocol_sha256(protocol: FormalAblationProtocol) -> str:
    """Hash the canonical execution choices, excluding attempt ID and timestamp."""
    payload = json.dumps(
        protocol.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def formal_ablation_selection_sha256(protocol: FormalAblationProtocol) -> str:
    """Hash outcome-affecting inputs while ignoring commit-only history changes.

    The protocol still records the exact source commit.  This second identity
    prevents an empty commit or a registry-only commit after ABANDONED from
    making the same code/model/corpus selection eligible for another attempt.
    """
    payload = protocol.model_dump(mode="json", exclude_none=True)
    payload.pop("source_commit")
    # Schema 1 included the image ID in the experiment selection. Preserve that
    # identity exactly for historical registry events. Schema 2 records the
    # image ID as provenance but binds the selection to the committed inputs
    # that build the scorer runtime, so rebuilding identical inputs cannot
    # reopen a consumed attempt.
    if protocol.schema_version >= 2:
        payload.pop("docker_image_id")
    # Authentication plumbing is provenance-critical but cannot change model
    # output, scorer truth, or the selected corpus.
    payload.pop("witness_credential_helper", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


class RegisteredAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["REGISTERED"] = "REGISTERED"
    attempt_id: str = Field(pattern=_HEX_64_PATTERN)
    protocol_sha256: str = Field(pattern=_HEX_64_PATTERN)
    source_commit: str = Field(pattern=_HEX_40_PATTERN)
    source_tree_sha256: str = Field(pattern=_HEX_64_PATTERN)
    manifest_sha256: str = Field(pattern=_HEX_64_PATTERN)
    output_path: str
    model: str
    reasoning_effort: str
    docker_image_id: str = Field(pattern=_DOCKER_IMAGE_ID_PATTERN)
    scorer_runtime_sha256: str | None = Field(
        default=None,
        pattern=_HEX_64_PATTERN,
    )
    codex_cli_version: str
    codex_cli_executable_sha256: str = Field(pattern=_HEX_64_PATTERN)
    codex_client: FormalCodexClientConfig
    codex_client_sha256: str = Field(pattern=_HEX_64_PATTERN)
    witness_credential_helper: FormalGitCredentialHelper | None = None
    witness_remote_name: str = Field(pattern=_WITNESS_REMOTE_NAME_PATTERN)
    witness_remote_url: str
    registered_at: str

    _output_path = field_validator("output_path")(_validate_output_path)
    _model = field_validator("model")(_validate_plain_text)
    _effort = field_validator("reasoning_effort")(_validate_plain_text)
    _cli_version = field_validator("codex_cli_version")(_validate_plain_text)
    _witness_remote_url = field_validator("witness_remote_url")(
        _validate_witness_remote_url
    )
    _registered_at = field_validator("registered_at")(_validate_timestamp)

    @property
    def witness_ref(self) -> str:
        return formal_ablation_witness_ref(self.attempt_id)

    def protocol(self) -> FormalAblationProtocol:
        return FormalAblationProtocol(
            schema_version=(2 if self.scorer_runtime_sha256 is not None else 1),
            source_commit=self.source_commit,
            source_tree_sha256=self.source_tree_sha256,
            manifest_sha256=self.manifest_sha256,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            docker_image_id=self.docker_image_id,
            scorer_runtime_sha256=self.scorer_runtime_sha256,
            codex_cli_version=self.codex_cli_version,
            codex_cli_executable_sha256=self.codex_cli_executable_sha256,
            codex_client=self.codex_client,
            codex_client_sha256=self.codex_client_sha256,
            witness_credential_helper=self.witness_credential_helper,
        )

    @model_validator(mode="after")
    def _protocol_digest_matches(self) -> "RegisteredAttempt":
        if self.witness_credential_helper is None:
            raise ValueError(
                "REGISTERED requires a pinned witness credential helper"
            )
        witness_host = urlsplit(self.witness_remote_url).hostname
        if self.witness_credential_helper.host != witness_host:
            raise ValueError(
                "REGISTERED witness credential helper host does not match its URL"
            )
        expected_output = f"runs/formal_ablation/{self.attempt_id}"
        if self.output_path != expected_output:
            raise ValueError(
                "REGISTERED output_path must equal runs/formal_ablation/<attempt_id>"
            )
        if self.protocol_sha256 != formal_ablation_protocol_sha256(self.protocol()):
            raise ValueError("REGISTERED protocol_sha256 does not match its fields")
        return self


class AbandonedAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["ABANDONED"] = "ABANDONED"
    attempt_id: str = Field(pattern=_HEX_64_PATTERN)
    recorded_at: str
    progress_status: Literal["known", "evidence_missing"] = "known"
    started_cells: int | None = Field(default=None, ge=0, le=204, strict=True)
    terminal_cells: int | None = Field(default=None, ge=0, le=204, strict=True)
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    reason: str
    report_sha256: str | None = Field(default=None, pattern=_HEX_64_PATTERN)
    report_fingerprint: str | None = Field(default=None, pattern=_HEX_64_PATTERN)

    _recorded_at = field_validator("recorded_at")(_validate_timestamp)
    _reason = field_validator("reason")(_validate_plain_text)

    @model_validator(mode="after")
    def _progress_and_report_are_consistent(self) -> "AbandonedAttempt":
        if self.progress_status == "known":
            if self.started_cells is None or self.terminal_cells is None:
                raise ValueError(
                    "ABANDONED known progress requires both cell counts"
                )
            if self.terminal_cells > self.started_cells:
                raise ValueError(
                    "ABANDONED terminal_cells cannot exceed started_cells"
                )
        elif self.started_cells is not None or self.terminal_cells is not None:
            raise ValueError(
                "ABANDONED missing evidence cannot claim cell counts"
            )
        if (self.report_sha256 is None) != (self.report_fingerprint is None):
            raise ValueError(
                "ABANDONED report_sha256 and report_fingerprint must appear together"
            )
        return self


class CompletedAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["COMPLETED"] = "COMPLETED"
    attempt_id: str = Field(pattern=_HEX_64_PATTERN)
    protocol_sha256: str = Field(pattern=_HEX_64_PATTERN)
    registration_registry_sha256: str = Field(pattern=_HEX_64_PATTERN)
    recorded_at: str
    report_sha256: str = Field(pattern=_HEX_64_PATTERN)
    report_fingerprint: str = Field(pattern=_HEX_64_PATTERN)

    _recorded_at = field_validator("recorded_at")(_validate_timestamp)


class UnregisteredRunRecorded(BaseModel):
    """A historical disclosure that never opens or authorizes an attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["UNREGISTERED_RUN_RECORDED"] = "UNREGISTERED_RUN_RECORDED"
    attempt_id: str = Field(pattern=_HEX_64_PATTERN)
    protocol_sha256: str = Field(pattern=_HEX_64_PATTERN)
    source_commit: str = Field(pattern=_HEX_40_PATTERN)
    source_tree_sha256: str = Field(pattern=_HEX_64_PATTERN)
    manifest_sha256: str = Field(pattern=_HEX_64_PATTERN)
    output_path: str
    model: str
    reasoning_effort: str
    docker_image_id: str = Field(pattern=_DOCKER_IMAGE_ID_PATTERN)
    codex_cli_version: str
    codex_cli_executable_sha256: str = Field(pattern=_HEX_64_PATTERN)
    codex_client: FormalCodexClientConfig
    codex_client_sha256: str = Field(pattern=_HEX_64_PATTERN)
    recorded_at: str
    reason: str
    published_report_path: str
    report_sha256: str = Field(pattern=_HEX_64_PATTERN)
    report_fingerprint: str = Field(pattern=_HEX_64_PATTERN)
    scheduled_cells: int = Field(ge=1, strict=True)
    usable_cells: int = Field(ge=0, strict=True)
    error_cells: int = Field(ge=0, strict=True)
    trust_delivered_correct: int = Field(ge=0, strict=True)
    trust_delivered_wrong: int = Field(ge=0, strict=True)
    gate_delivered_correct: int = Field(ge=0, strict=True)
    gate_delivered_wrong: int = Field(ge=0, strict=True)
    gate_intercepted_wrong: int = Field(ge=0, strict=True)
    gate_rejected_correct: int = Field(ge=0, strict=True)
    verify_delivered_correct: int = Field(ge=0, strict=True)
    verify_delivered_wrong: int = Field(ge=0, strict=True)
    verify_not_delivered: int = Field(ge=0, strict=True)

    _output_path = field_validator("output_path")(_validate_output_path)
    _model = field_validator("model")(_validate_plain_text)
    _effort = field_validator("reasoning_effort")(_validate_plain_text)
    _cli_version = field_validator("codex_cli_version")(_validate_plain_text)
    _recorded_at = field_validator("recorded_at")(_validate_timestamp)
    _reason = field_validator("reason")(_validate_plain_text)
    _published_report_path = field_validator("published_report_path")(
        _validate_output_path
    )

    def protocol(self) -> FormalAblationProtocol:
        return FormalAblationProtocol(
            schema_version=1,
            source_commit=self.source_commit,
            source_tree_sha256=self.source_tree_sha256,
            manifest_sha256=self.manifest_sha256,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            docker_image_id=self.docker_image_id,
            codex_cli_version=self.codex_cli_version,
            codex_cli_executable_sha256=self.codex_cli_executable_sha256,
            codex_client=self.codex_client,
            codex_client_sha256=self.codex_client_sha256,
        )

    @model_validator(mode="after")
    def _protocol_digest_matches(self) -> "UnregisteredRunRecorded":
        expected_report_path = (
            "benchmarks/formal_ablation_history/"
            f"{self.source_commit}/ablation_report.json"
        )
        if self.published_report_path != expected_report_path:
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED published_report_path must be derived "
                "from source_commit"
            )
        if self.protocol_sha256 != formal_ablation_protocol_sha256(self.protocol()):
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED protocol_sha256 does not match its fields"
            )
        if self.usable_cells + self.error_cells != self.scheduled_cells:
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED cell totals are inconsistent"
            )
        if (
            self.trust_delivered_correct + self.trust_delivered_wrong
            != self.usable_cells
        ):
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED trust counts are inconsistent"
            )
        gate_total = (
            self.gate_delivered_correct
            + self.gate_delivered_wrong
            + self.gate_intercepted_wrong
            + self.gate_rejected_correct
        )
        verify_total = (
            self.verify_delivered_correct
            + self.verify_delivered_wrong
            + self.verify_not_delivered
        )
        if gate_total != self.usable_cells or verify_total != self.usable_cells:
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED gate or verify counts are inconsistent"
            )
        return self


FormalAttemptEvent = Annotated[
    RegisteredAttempt
    | AbandonedAttempt
    | CompletedAttempt
    | UnregisteredRunRecorded,
    Field(discriminator="event"),
]


class FormalAblationAttemptRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = FORMAL_ABLATION_ATTEMPTS_SCHEMA
    events: tuple[FormalAttemptEvent, ...]

    @model_validator(mode="after")
    def _events_form_an_append_only_state_machine(
        self,
    ) -> "FormalAblationAttemptRegistry":
        registered: dict[str, RegisteredAttempt] = {}
        disclosed_ids: set[str] = set()
        disclosed_reports: set[str] = set()
        terminal_ids: set[str] = set()
        consumed_protocols: set[str] = set()
        consumed_selections: set[str] = set()
        open_attempt: str | None = None

        for event in self.events:
            if isinstance(event, RegisteredAttempt):
                if open_attempt is not None:
                    raise ValueError(
                        "an open REGISTERED attempt must end before a new attempt"
                    )
                if event.attempt_id in registered or event.attempt_id in disclosed_ids:
                    raise ValueError("formal ablation attempt_id must be unique")
                selection_sha256 = formal_ablation_selection_sha256(event.protocol())
                if (
                    event.protocol_sha256 in consumed_protocols
                    or selection_sha256 in consumed_selections
                ):
                    raise ValueError(
                        "a consumed formal ablation selection cannot be registered again"
                    )
                registered[event.attempt_id] = event
                consumed_protocols.add(event.protocol_sha256)
                consumed_selections.add(selection_sha256)
                open_attempt = event.attempt_id
                continue

            if isinstance(event, UnregisteredRunRecorded):
                if open_attempt is not None:
                    raise ValueError(
                        "an open REGISTERED attempt must end before another attempt record"
                    )
                if event.attempt_id in registered or event.attempt_id in disclosed_ids:
                    raise ValueError("formal ablation attempt_id must be unique")
                if event.published_report_path in disclosed_reports:
                    raise ValueError(
                        "formal ablation historical report path must be unique"
                    )
                selection_sha256 = formal_ablation_selection_sha256(event.protocol())
                if (
                    event.protocol_sha256 in consumed_protocols
                    or selection_sha256 in consumed_selections
                ):
                    raise ValueError(
                        "a consumed formal ablation selection cannot be recorded again"
                    )
                disclosed_ids.add(event.attempt_id)
                disclosed_reports.add(event.published_report_path)
                consumed_protocols.add(event.protocol_sha256)
                consumed_selections.add(selection_sha256)
                continue

            if event.attempt_id not in registered:
                raise ValueError(f"{event.event} must follow a REGISTERED attempt")
            if event.attempt_id in terminal_ids:
                raise ValueError("a terminal formal ablation attempt cannot change state")
            if open_attempt != event.attempt_id:
                raise ValueError(f"{event.event} does not terminate the open attempt")
            registration = registered[event.attempt_id]
            if isinstance(event, CompletedAttempt):
                if event.protocol_sha256 != registration.protocol_sha256:
                    raise ValueError(
                        "COMPLETED protocol_sha256 differs from its registration"
                    )
            terminal_ids.add(event.attempt_id)
            open_attempt = None
        return self

    def open_registration(self) -> RegisteredAttempt | None:
        open_attempt: RegisteredAttempt | None = None
        for event in self.events:
            if isinstance(event, RegisteredAttempt):
                open_attempt = event
            elif isinstance(event, (AbandonedAttempt, CompletedAttempt)):
                if open_attempt is not None and open_attempt.attempt_id == event.attempt_id:
                    open_attempt = None
        return open_attempt

    def registration(self, attempt_id: str) -> RegisteredAttempt | None:
        return next(
            (
                event
                for event in self.events
                if isinstance(event, RegisteredAttempt)
                and event.attempt_id == attempt_id
            ),
            None,
        )

    def completions(self) -> tuple[CompletedAttempt, ...]:
        return tuple(
            event for event in self.events if isinstance(event, CompletedAttempt)
        )

    def disclosures(self) -> tuple[UnregisteredRunRecorded, ...]:
        return tuple(
            event
            for event in self.events
            if isinstance(event, UnregisteredRunRecorded)
        )


def parse_formal_ablation_attempt_registry(
    payload: bytes,
) -> FormalAblationAttemptRegistry:
    if len(payload) > MAX_FORMAL_ABLATION_ATTEMPTS_BYTES:
        raise ValueError("formal ablation attempt registry is too large")
    try:
        raw: Any = json.loads(payload)
        return FormalAblationAttemptRegistry.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError("formal ablation attempt registry is invalid") from error


def formal_ablation_attempt_registry_bytes(
    registry: FormalAblationAttemptRegistry,
) -> bytes:
    """Return one stable representation suitable for a tracked registry file."""
    payload = registry.model_dump(mode="json")
    for event in payload["events"]:
        if event.get("scorer_runtime_sha256") is None:
            event.pop("scorer_runtime_sha256", None)
    return (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def registry_has_prefix(
    current: FormalAblationAttemptRegistry,
    historical: FormalAblationAttemptRegistry,
) -> bool:
    """Whether current retains every historical event in the original order."""
    if len(current.events) < len(historical.events):
        return False
    return current.events[: len(historical.events)] == historical.events
