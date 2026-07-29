"""Operator commands for the one-shot formal ablation lifecycle.

These commands only register or close an attempt. They never commit, push, or
start the benchmark, so each external state change remains an explicit
operator action.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlsplit

from .ablation import (
    _CELL_ATTEMPT_SCHEMA,
    _FORMAL_CORPUS_MANIFEST_PATH,
    _FORMAL_OUTPUT_LOCK_NAME,
    _FORMAL_REPETITIONS,
    _FORMAL_RUN_HEADER_NAME,
    _FORMAL_RUN_HEADER_SCHEMA,
    _FORMAL_TASK_COUNT,
    _canonical_json_object_bytes,
    _formal_ablation_lock,
    _formal_git_credential_helper,
    _formal_input_snapshot_is_valid,
    _formal_witness_remote_url,
    _git_control_env,
    _load_cached_cell,
    _load_formal_corpus_manifest,
    _preflight_formal_git_credential_helper,
    _probe_docker_image,
    _read_bounded_bytes,
    _recover_docker_operations,
    _resolve_docker_image_id,
    _scorer_runtime_digest,
    _source_tree_digest,
    _trusted_control_executable,
    _validate_formal_head_checkout,
)
from .ablation_attempts import (
    FORMAL_ABLATION_ATTEMPTS_PATH,
    MAX_FORMAL_ABLATION_ATTEMPTS_BYTES,
    AbandonedAttempt,
    CompletedAttempt,
    FormalAblationAttemptRegistry,
    FormalCodexClientConfig,
    FormalGitCredentialHelper,
    RegisteredAttempt,
    formal_ablation_attempt_registry_bytes,
    formal_ablation_protocol_sha256,
    formal_ablation_witness_commit_bytes,
    formal_ablation_witness_commit_oid,
    formal_ablation_witness_message,
    formal_attempt_lock,
    formal_codex_client_config_from_runtime,
    formal_codex_client_sha256,
    make_formal_codex_client,
    parse_formal_ablation_attempt_registry,
    validate_formal_witness_remote_url,
)
from .clock import now
from .config import Config
from .durable_io import (
    anchored_read_bytes,
    anchored_update_bytes,
)
from .sandbox import DockerBackend
from .tools.shell import run

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_WITNESS_REMOTE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MAX_REPORT_BYTES = 32 * 1024 * 1024
_MAX_RESULT_BYTES = 8 * 1024 * 1024


class FormalAttemptCommandError(RuntimeError):
    """An operator request failed without changing the attempt registry."""


def _publication_inspection(repository: Path):
    from .formal_publish import inspect_formal_publication

    return inspect_formal_publication(repository=repository)


def _require_no_publication_transaction(
    repository: Path,
    *,
    action: str,
) -> None:
    inspection = _publication_inspection(repository)
    if inspection.status != "CLEAN":
        detail = f": {inspection.reason}" if inspection.reason else ""
        raise FormalAttemptCommandError(
            f"cannot {action} while formal publication is "
            f"{inspection.status.lower()}{detail}"
        )


def _repository_root(root: str | Path) -> Path:
    candidate = Path(root).resolve()
    if (
        not (candidate / ".git").exists()
        or not (candidate / "pyproject.toml").is_file()
        or not (candidate / "src" / "lha").is_dir()
    ):
        raise FormalAttemptCommandError(
            "ablation-attempt must run from the LHA Git repository root"
        )
    return candidate


def _git_output(
    repository: Path,
    arguments: list[str],
    *,
    label: str,
) -> str:
    git = str(_trusted_control_executable("git")["path"])
    result = run(
        [git, *arguments],
        cwd=repository,
        timeout=30,
        env=_git_control_env(),
    )
    if (
        result.returncode != 0
        or result.output_truncated
        or result.cleanup_unconfirmed
    ):
        raise FormalAttemptCommandError(f"cannot {label}")
    return result.stdout


def _anonymous_git_output(arguments: list[str], *, label: str) -> str:
    """Run a public-remote probe without repository or user Git configuration."""
    git = str(_trusted_control_executable("git")["path"])
    with tempfile.TemporaryDirectory(prefix="lha_formal_remote_") as temporary:
        result = run(
            [git, *arguments],
            cwd=Path(temporary),
            timeout=30,
            env=_git_control_env(),
        )
    if (
        result.returncode != 0
        or result.output_truncated
        or result.cleanup_unconfirmed
    ):
        raise FormalAttemptCommandError(
            f"cannot {label}; the witness URL must be anonymously readable"
        )
    return result.stdout


def _clean_head(repository: Path) -> tuple[str, str]:
    status = _git_output(
        repository,
        ["status", "--porcelain=v1", "--untracked-files=normal"],
        label="inspect the worktree",
    )
    if status:
        raise FormalAttemptCommandError(
            "formal attempt registry updates require a clean worktree"
        )
    head = _git_output(
        repository,
        ["rev-parse", "--verify", "HEAD"],
        label="resolve HEAD",
    ).strip()
    branch = _git_output(
        repository,
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        label="resolve the current branch",
    ).strip()
    if _HEX_40.fullmatch(head) is None or not branch:
        raise FormalAttemptCommandError(
            "formal attempt registration requires a branch with a valid HEAD"
        )
    _git_output(
        repository,
        ["check-ref-format", f"refs/heads/{branch}"],
        label="validate the current branch",
    )
    return head, branch


def _registry_bytes(repository: Path) -> bytes:
    path = repository / FORMAL_ABLATION_ATTEMPTS_PATH
    try:
        payload = anchored_read_bytes(path, anchor=repository)
    except OSError as error:
        raise FormalAttemptCommandError(
            "formal attempt registry cannot be read safely"
        ) from error
    if payload is None or len(payload) > MAX_FORMAL_ABLATION_ATTEMPTS_BYTES:
        raise FormalAttemptCommandError("formal attempt registry is unavailable")
    return payload


def _committed_registry(
    repository: Path,
    *,
    head: str,
) -> tuple[bytes, FormalAblationAttemptRegistry]:
    committed, registry = _registry_at_head(repository, head=head)
    if committed != _registry_bytes(repository):
        raise FormalAttemptCommandError(
            "formal attempt registry differs from the committed HEAD"
        )
    return committed, registry


def _registry_at_head(
    repository: Path,
    *,
    head: str,
) -> tuple[bytes, FormalAblationAttemptRegistry]:
    """Read the committed registry even while a publication CAS is uncommitted."""
    relative = FORMAL_ABLATION_ATTEMPTS_PATH.as_posix()
    size_text = _git_output(
        repository,
        ["cat-file", "-s", f"{head}:{relative}"],
        label="inspect the committed attempt registry",
    ).strip()
    try:
        committed_size = int(size_text)
    except ValueError as error:
        raise FormalAttemptCommandError(
            "committed formal attempt registry has an invalid size"
        ) from error
    if not 0 <= committed_size <= MAX_FORMAL_ABLATION_ATTEMPTS_BYTES:
        raise FormalAttemptCommandError("committed formal attempt registry is too large")
    committed = _git_output(
        repository,
        ["show", f"{head}:{relative}"],
        label="read the committed attempt registry",
    ).encode("utf-8")
    if len(committed) != committed_size:
        raise FormalAttemptCommandError(
            "committed formal attempt registry changed while reading"
        )
    try:
        registry = parse_formal_ablation_attempt_registry(committed)
    except ValueError as error:
        raise FormalAttemptCommandError(
            "formal attempt registry is invalid"
        ) from error
    return committed, registry


def _https_witness_remote(
    repository: Path,
    *,
    remote_name: str,
    branch: str,
    source_commit: str,
) -> str:
    git = str(_trusted_control_executable("git")["path"])
    try:
        url = _formal_witness_remote_url(
            git,
            repository_root=repository,
            remote_name=remote_name,
        )
    except RuntimeError as error:
        raise FormalAttemptCommandError(
            "witness remote must have exactly one public HTTPS URL"
        ) from error
    remote_line = _anonymous_git_output(
        ["ls-remote", "--heads", url, f"refs/heads/{branch}"],
        label="verify the pushed source commit",
    ).strip()
    fields = remote_line.split()
    if fields != [source_commit, f"refs/heads/{branch}"]:
        raise FormalAttemptCommandError(
            "source HEAD is not published at the matching witness remote branch"
        )
    return url


def _witness_credential_helper(
    url: str,
    *,
    expected: FormalGitCredentialHelper | None = None,
) -> FormalGitCredentialHelper:
    try:
        validated = validate_formal_witness_remote_url(url)
        host = urlsplit(validated).hostname
        if host is None:
            raise ValueError("witness URL has no host")
        return _formal_git_credential_helper(host, expected=expected)
    except (RuntimeError, ValueError) as error:
        raise FormalAttemptCommandError(
            "GitHub credential helper could not be bound to stable bytes"
        ) from error


def _codex_protocol(
    *,
    config: Config,
    model: str,
    reasoning_effort: str,
) -> tuple[str, str, FormalCodexClientConfig]:
    client = make_formal_codex_client(
        cli_path=config.codex_cli_path,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    client.preflight()
    identity = client._cli_identity
    version = client._version
    if (
        not isinstance(identity, tuple)
        or len(identity) != 7
        or not isinstance(identity[5], str)
        or _HEX_64.fullmatch(identity[5]) is None
        or not isinstance(version, str)
        or not version
        or version == "unknown"
        or client.permission_model != "profile"
        or client.permission_profile != "lha-read"
        or client.credential_barrier != "verified"
    ):
        raise FormalAttemptCommandError(
            "Codex CLI preflight did not produce the fixed formal protocol"
        )
    try:
        fixed = formal_codex_client_config_from_runtime(client)
    except ValueError as error:
        raise FormalAttemptCommandError(
            "Codex CLI preflight did not produce the fixed formal protocol"
        ) from error
    return version, identity[5], fixed


def _formal_output_path_is_absent(repository: Path, attempt_id: str) -> bool:
    """Reject link/non-directory ancestors without creating the reserved path."""
    parents = (
        repository / "runs",
        repository / "runs" / "formal_ablation",
    )
    for parent in parents:
        try:
            metadata = parent.lstat()
        except FileNotFoundError:
            break
        except OSError:
            return False
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            return False
    output = parents[-1] / attempt_id
    try:
        output.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _append_event(
    repository: Path,
    *,
    expected: bytes,
    event: RegisteredAttempt | CompletedAttempt | AbandonedAttempt,
) -> tuple[FormalAblationAttemptRegistry, bytes]:
    path = repository / FORMAL_ABLATION_ATTEMPTS_PATH
    try:
        observed = anchored_read_bytes(path, anchor=repository)
    except OSError as error:
        raise FormalAttemptCommandError(
            "formal attempt registry cannot be read before the update"
        ) from error
    if observed != expected:
        raise FormalAttemptCommandError(
            "formal attempt registry changed during the update"
        )
    try:
        registry_before = parse_formal_ablation_attempt_registry(expected)
        registry_after = FormalAblationAttemptRegistry(
            events=(*registry_before.events, event)
        )
    except (TypeError, ValueError) as error:
        raise FormalAttemptCommandError(
            f"formal attempt transition to {event.event} is invalid"
        ) from error
    updated = formal_ablation_attempt_registry_bytes(registry_after)
    if len(updated) > MAX_FORMAL_ABLATION_ATTEMPTS_BYTES:
        raise FormalAttemptCommandError("formal attempt registry is too large")

    def update(current: bytes | None) -> bytes:
        if current != expected:
            raise FormalAttemptCommandError(
                "formal attempt registry changed during the update"
            )
        if current is None:
            raise FormalAttemptCommandError(
                "formal attempt registry disappeared during the update"
            )
        return updated

    try:
        anchored_update_bytes(path, update, anchor=repository)
    except OSError as error:
        # A durability barrier can fail after rename installed the exact bytes.
        # Resolve that ambiguity by reading the authoritative file before
        # reporting failure; retrying an already-installed terminal event would
        # otherwise create a misleading transition error.
        try:
            current = anchored_read_bytes(path, anchor=repository)
        except OSError:
            current = None
        if current == updated:
            return registry_after, updated
        raise FormalAttemptCommandError(
            "formal attempt registry could not be replaced atomically"
        ) from error
    try:
        current = anchored_read_bytes(path, anchor=repository)
    except OSError as error:
        raise FormalAttemptCommandError(
            "formal attempt registry update could not be confirmed"
        ) from error
    if current != updated:
        raise FormalAttemptCommandError(
            "formal attempt registry changed after the update"
        )
    return registry_after, updated


def _register_formal_attempt_locked(
    *,
    repo_root: str | Path,
    config: Config,
    model: str,
    reasoning_effort: str,
    docker_image_id: str,
    witness_remote_name: str,
    attempt_id_factory: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Append one fully resolved ``REGISTERED`` event."""
    repository = _repository_root(repo_root)
    _require_no_publication_transaction(
        repository,
        action="register a formal attempt",
    )
    if not model or model.strip() != model:
        raise FormalAttemptCommandError("formal model must be explicit")
    if not reasoning_effort or reasoning_effort.strip() != reasoning_effort:
        raise FormalAttemptCommandError("formal reasoning effort must be explicit")
    if _WITNESS_REMOTE_NAME.fullmatch(witness_remote_name) is None:
        raise FormalAttemptCommandError("formal witness remote name is invalid")
    if _DOCKER_IMAGE_ID.fullmatch(docker_image_id) is None:
        raise FormalAttemptCommandError(
            "--docker-image-id must be an immutable sha256 image ID, not a tag"
        )

    source_commit, branch = _clean_head(repository)
    registry_bytes, registry = _committed_registry(
        repository,
        head=source_commit,
    )
    if registry.open_registration() is not None:
        raise FormalAttemptCommandError(
            "an open formal attempt must be completed or abandoned first"
        )
    try:
        manifest, manifest_sha256 = _load_formal_corpus_manifest(
            repository / _FORMAL_CORPUS_MANIFEST_PATH,
            repository,
        )
        git_path = str(_trusted_control_executable("git")["path"])
        source_files = _validate_formal_head_checkout(
            repository,
            git_path=git_path,
            head=source_commit,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )
    except (OSError, TypeError, ValueError) as error:
        raise FormalAttemptCommandError(
            "formal corpus manifest or fixed inputs are invalid"
        ) from error
    except RuntimeError as error:
        raise FormalAttemptCommandError(
            "working source or formal corpus differs from the trusted HEAD"
        ) from error
    source_tree_sha256 = _source_tree_digest(source_files)
    try:
        scorer_runtime_sha256 = _scorer_runtime_digest(
            repository,
            git_path=git_path,
            commit=source_commit,
        )
    except RuntimeError as error:
        raise FormalAttemptCommandError(
            "formal scorer runtime inputs differ from the trusted HEAD"
        ) from error
    witness_url = _https_witness_remote(
        repository,
        remote_name=witness_remote_name,
        branch=branch,
        source_commit=source_commit,
    )
    witness_credential_helper = _witness_credential_helper(witness_url)
    _preflight_formal_git_credential_helper(
        git_path,
        witness_credential_helper,
    )
    measured_image_id = _resolve_docker_image_id(docker_image_id)
    if measured_image_id != docker_image_id:
        raise FormalAttemptCommandError("Docker image ID changed during inspection")
    try:
        with tempfile.TemporaryDirectory(
            prefix="lha_formal_image_probe_"
        ) as probe_directory:
            _probe_docker_image(
                DockerBackend(image=docker_image_id),
                image_id=docker_image_id,
                workdir=Path(probe_directory),
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise FormalAttemptCommandError(
            "Docker image lacks the formal scorer runtime"
        ) from error
    cli_version, cli_digest, client_config = _codex_protocol(
        config=config,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    protocol_fields = {
        "source_commit": source_commit,
        "source_tree_sha256": source_tree_sha256,
        "manifest_sha256": manifest_sha256,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "docker_image_id": docker_image_id,
        "scorer_runtime_sha256": scorer_runtime_sha256,
        "codex_cli_version": cli_version,
        "codex_cli_executable_sha256": cli_digest,
        "codex_client": client_config,
        "codex_client_sha256": formal_codex_client_sha256(client_config),
        "witness_credential_helper": witness_credential_helper,
    }
    from .ablation_attempts import FormalAblationProtocol

    protocol = FormalAblationProtocol(
        schema_version=2,
        **protocol_fields,
    )
    attempt_id = (attempt_id_factory or (lambda: secrets.token_hex(32)))()
    if _HEX_64.fullmatch(attempt_id) is None:
        raise FormalAttemptCommandError("generated formal attempt ID is invalid")
    if any(
        getattr(event, "attempt_id", None) == attempt_id
        for event in registry.events
    ):
        raise FormalAttemptCommandError("generated formal attempt ID already exists")
    if not _formal_output_path_is_absent(repository, attempt_id):
        raise FormalAttemptCommandError(
            "generated formal attempt output path exists or has unsafe ancestors"
        )
    witness_ref = f"refs/heads/formal-attempts/{attempt_id}"
    if _anonymous_git_output(
        ["ls-remote", "--refs", witness_url, witness_ref],
        label="check the new formal witness ref",
    ).strip():
        raise FormalAttemptCommandError(
            "generated formal attempt witness ref already exists"
        )
    registration = RegisteredAttempt(
        attempt_id=attempt_id,
        protocol_sha256=formal_ablation_protocol_sha256(protocol),
        output_path=f"runs/formal_ablation/{attempt_id}",
        witness_remote_name=witness_remote_name,
        witness_remote_url=witness_url,
        registered_at=now().isoformat(),
        **protocol_fields,
    )
    final_head, final_branch = _clean_head(repository)
    final_registry_bytes, final_registry = _committed_registry(
        repository,
        head=final_head,
    )
    try:
        final_manifest, final_manifest_sha256 = _load_formal_corpus_manifest(
            repository / _FORMAL_CORPUS_MANIFEST_PATH,
            repository,
        )
        final_source_files = _validate_formal_head_checkout(
            repository,
            git_path=git_path,
            head=final_head,
            manifest=final_manifest,
            manifest_sha256=final_manifest_sha256,
        )
    except (OSError, TypeError, ValueError) as error:
        raise FormalAttemptCommandError(
            "formal corpus changed during registration"
        ) from error
    except RuntimeError as error:
        raise FormalAttemptCommandError(
            "working source or formal corpus changed during registration"
        ) from error
    final_source_tree_sha256 = _source_tree_digest(final_source_files)
    try:
        final_scorer_runtime_sha256 = _scorer_runtime_digest(
            repository,
            git_path=git_path,
            commit=final_head,
        )
    except RuntimeError as error:
        raise FormalAttemptCommandError(
            "formal scorer runtime inputs changed during registration"
        ) from error
    final_witness_url = _https_witness_remote(
        repository,
        remote_name=witness_remote_name,
        branch=final_branch,
        source_commit=final_head,
    )
    final_witness_credential_helper = _witness_credential_helper(
        final_witness_url,
        expected=witness_credential_helper,
    )
    final_output_unavailable = not _formal_output_path_is_absent(
        repository,
        attempt_id,
    )
    final_witness_exists = bool(
        _anonymous_git_output(
            ["ls-remote", "--refs", witness_url, witness_ref],
            label="recheck the new formal witness ref",
        ).strip()
    )
    if (
        final_head != source_commit
        or final_branch != branch
        or final_registry_bytes != registry_bytes
        or any(
            getattr(event, "attempt_id", None) == attempt_id
            for event in final_registry.events
        )
        or final_manifest_sha256 != manifest_sha256
        or final_source_tree_sha256 != source_tree_sha256
        or final_scorer_runtime_sha256 != scorer_runtime_sha256
        or final_witness_url != witness_url
        or final_witness_credential_helper != witness_credential_helper
        or final_output_unavailable
        or final_witness_exists
        or _resolve_docker_image_id(docker_image_id) != docker_image_id
    ):
        raise FormalAttemptCommandError(
            "a formal registration input changed during preflight"
        )
    _registry_after, updated = _append_event(
        repository,
        expected=registry_bytes,
        event=registration,
    )
    return {
        "state": "REGISTERED",
        "attempt_id": attempt_id,
        "source_commit": source_commit,
        "protocol_sha256": registration.protocol_sha256,
        "registration_registry_sha256": hashlib.sha256(updated).hexdigest(),
        "output_path": registration.output_path,
        "witness_remote_name": witness_remote_name,
        "witness_remote_url": witness_url,
        "docker_image_id": docker_image_id,
        "scorer_runtime_sha256": scorer_runtime_sha256,
    }


def register_formal_attempt(
    *,
    repo_root: str | Path,
    config: Config,
    model: str,
    reasoning_effort: str,
    docker_image_id: str,
    witness_remote_name: str,
    attempt_id_factory: Callable[[], str] | None = None,
) -> dict[str, Any]:
    repository = _repository_root(repo_root)
    with formal_attempt_lock(repository):
        return _register_formal_attempt_locked(
            repo_root=repository,
            config=config,
            model=model,
            reasoning_effort=reasoning_effort,
            docker_image_id=docker_image_id,
            witness_remote_name=witness_remote_name,
            attempt_id_factory=attempt_id_factory,
        )


def _open_registration_at_clean_head(
    repository: Path,
) -> tuple[str, bytes, RegisteredAttempt]:
    head, _branch = _clean_head(repository)
    registry_bytes, registry = _committed_registry(repository, head=head)
    registration = _registration_from_committed_head(
        repository,
        head=head,
        registry=registry,
    )
    return head, registry_bytes, registration


def _registration_from_committed_head(
    repository: Path,
    *,
    head: str,
    registry: FormalAblationAttemptRegistry,
) -> RegisteredAttempt:
    registration = registry.open_registration()
    if not isinstance(registration, RegisteredAttempt):
        raise FormalAttemptCommandError(
            "formal attempt registry has no open REGISTERED attempt"
        )
    parents = _git_output(
        repository,
        ["rev-list", "--parents", "-n", "1", head],
        label="inspect the registration commit",
    ).strip().split()
    if parents != [head, registration.source_commit]:
        raise FormalAttemptCommandError(
            "registration commit must directly follow its source commit"
        )
    changed = _git_output(
        repository,
        ["diff-tree", "--no-commit-id", "--name-only", "-r", head],
        label="inspect registration commit paths",
    ).splitlines()
    if changed != [FORMAL_ABLATION_ATTEMPTS_PATH.as_posix()]:
        raise FormalAttemptCommandError(
            "registration commit may only change the formal attempt registry"
        )
    return registration


def _validate_registration_checkout(
    repository: Path,
    *,
    registration_head: str,
    registration: RegisteredAttempt,
) -> None:
    """Rebind completion to the exact source/control/corpus bytes in HEAD."""
    current_head = _git_output(
        repository,
        ["rev-parse", "--verify", "HEAD"],
        label="revalidate completion HEAD",
    ).strip()
    if current_head != registration_head:
        raise FormalAttemptCommandError(
            "formal registration HEAD changed during completion"
        )
    try:
        manifest, manifest_sha256 = _load_formal_corpus_manifest(
            repository / _FORMAL_CORPUS_MANIFEST_PATH,
            repository,
        )
        if manifest_sha256 != registration.manifest_sha256:
            raise ValueError("formal manifest differs from the registration")
        git_path = str(_trusted_control_executable("git")["path"])
        source_files = _validate_formal_head_checkout(
            repository,
            git_path=git_path,
            head=registration_head,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )
        if _source_tree_digest(source_files) != registration.source_tree_sha256:
            raise ValueError("formal source digest differs from the registration")
        if (
            registration.scorer_runtime_sha256 is not None
            and _scorer_runtime_digest(
                repository,
                git_path=git_path,
                commit=registration_head,
            )
            != registration.scorer_runtime_sha256
        ):
            raise ValueError(
                "formal scorer runtime digest differs from the registration"
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, FormalAttemptCommandError):
            raise
        raise FormalAttemptCommandError(
            "formal source, controls, manifest, or corpus changed before completion"
        ) from error


def _open_registration_at_head_during_publication(
    repository: Path,
) -> tuple[str, bytes, RegisteredAttempt]:
    head = _git_output(
        repository,
        ["rev-parse", "--verify", "HEAD"],
        label="resolve the publication registration commit",
    ).strip()
    if _HEX_40.fullmatch(head) is None:
        raise FormalAttemptCommandError(
            "publication recovery has no valid registration commit"
        )
    registry_bytes, registry = _registry_at_head(repository, head=head)
    registration = _registration_from_committed_head(
        repository,
        head=head,
        registry=registry,
    )
    return head, registry_bytes, registration


def _recover_formal_operations(
    registration: RegisteredAttempt,
    output: Path,
    *,
    allow_recovered: bool,
) -> int:
    backend = DockerBackend(
        image=registration.docker_image_id,
        operation_lease_dir=output,
    )
    try:
        return _recover_docker_operations(
            backend,
            output,
            allow_recovered=allow_recovered,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise FormalAttemptCommandError(
            "formal operation cleanup could not be confirmed"
        ) from error


def _complete_formal_attempt_locked(
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate all 204 cells and append their report-bound completion."""
    from .release_claims import validate_formal_ablation_output

    repository = _repository_root(repo_root)
    publication = _publication_inspection(repository)
    if publication.status == "QUARANTINED":
        raise FormalAttemptCommandError(
            "formal publication is quarantined"
            + (f": {publication.reason}" if publication.reason else "")
        )
    recovering = publication.status == "RECOVERY_REQUIRED"
    if recovering:
        registration_head, registry_bytes, registration = (
            _open_registration_at_head_during_publication(repository)
        )
        if publication.attempt_id != registration.attempt_id:
            raise FormalAttemptCommandError(
                "publication journal does not match the registration at HEAD"
            )
        registry_current_bytes = _registry_bytes(repository)
        _validate_registration_checkout(
            repository,
            registration_head=registration_head,
            registration=registration,
        )
        if (
            publication.transaction_state == "INSTALLED"
            and registry_current_bytes != registry_bytes
        ):
            return _cleanup_appended_formal_publication(
                repository=repository,
                registration=registration,
                registration_head=registration_head,
                registry_current_bytes=registry_current_bytes,
            )
        if publication.transaction_state == "INSTALLED":
            raw = _load_formal_report_for_recovery(
                repository / "benchmarks"
            )
            return _complete_validated_formal_attempt(
                repository=repository,
                output=repository / registration.output_path,
                registration=registration,
                registration_head=registration_head,
                registry_bytes=registry_bytes,
                raw=raw,
                recovered_operations=0,
                recovering=True,
            )
    else:
        (
            registration_head,
            registry_bytes,
            registration,
        ) = _open_registration_at_clean_head(repository)
        registry_current_bytes = registry_bytes
        _validate_registration_checkout(
            repository,
            registration_head=registration_head,
            registration=registration,
        )
    output = repository / registration.output_path
    try:
        output_lock = _formal_ablation_lock(output, require_existing=True)
        with output_lock:
            if _registry_bytes(repository) != registry_current_bytes:
                raise FormalAttemptCommandError(
                    "formal attempt registry changed before completion"
                )
            _validate_registration_checkout(
                repository,
                registration_head=registration_head,
                registration=registration,
            )
            recovered_operations = _recover_formal_operations(
                registration,
                output,
                allow_recovered=False,
            )
            if recovering:
                raw = _load_formal_report_for_recovery(output)
            else:
                try:
                    raw = validate_formal_ablation_output(
                        output,
                        repo_root=repository,
                    )
                except ValueError as error:
                    raise FormalAttemptCommandError(
                        f"formal output is incomplete or invalid: {error}"
                    ) from error
            result = _complete_validated_formal_attempt(
                repository=repository,
                output=output,
                registration=registration,
                registration_head=registration_head,
                registry_bytes=registry_bytes,
                raw=raw,
                recovered_operations=recovered_operations,
                recovering=recovering,
            )
    except RuntimeError as error:
        if isinstance(error, FormalAttemptCommandError):
            raise
        raise FormalAttemptCommandError(
            "formal output lock could not be held"
        ) from error
    return result


def _cleanup_appended_formal_publication(
    *,
    repository: Path,
    registration: RegisteredAttempt,
    registration_head: str,
    registry_current_bytes: bytes,
) -> dict[str, Any]:
    """Finish private cleanup after the exact public COMPLETED CAS succeeded."""
    from .formal_publish import FormalPublishError, cleanup_formal_publication

    try:
        current = parse_formal_ablation_attempt_registry(
            registry_current_bytes
        )
    except ValueError as error:
        raise FormalAttemptCommandError(
            "completed formal registry cannot be parsed during recovery"
        ) from error
    completions = [
        event
        for event in current.completions()
        if event.attempt_id == registration.attempt_id
    ]
    if len(completions) != 1:
        raise FormalAttemptCommandError(
            "publication recovery has no unique COMPLETED event"
        )
    completion = completions[0]
    _validate_registration_checkout(
        repository,
        registration_head=registration_head,
        registration=registration,
    )
    try:
        finalized = cleanup_formal_publication(
            repository=repository,
            attempt_id=registration.attempt_id,
        )
    except FormalPublishError as error:
        raise FormalAttemptCommandError(
            f"completed formal publication cleanup failed: {error}"
        ) from error
    if finalized.action != "COMMITTED_AND_CLEANED":
        raise FormalAttemptCommandError(
            "completed formal publication was not cleaned"
        )
    return {
        "state": "COMPLETED",
        "attempt_id": registration.attempt_id,
        "protocol_sha256": registration.protocol_sha256,
        "current_registry_sha256": hashlib.sha256(
            registry_current_bytes
        ).hexdigest(),
        "report_sha256": completion.report_sha256,
        "report_fingerprint": completion.report_fingerprint,
        "scheduled_cells": _FORMAL_TASK_COUNT * _FORMAL_REPETITIONS,
        "recovered_operations": 0,
        "publication_recovery": finalized.action,
    }


def _load_formal_report_for_recovery(report_directory: Path) -> dict[str, Any]:
    try:
        payload = _read_bounded_bytes(
            report_directory / "ablation_report.json",
            max_bytes=_MAX_REPORT_BYTES,
        )
        raw = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise FormalAttemptCommandError(
            "formal publication source report cannot be recovered"
        ) from error
    if not isinstance(raw, dict):
        raise FormalAttemptCommandError(
            "formal publication source report is not a JSON object"
        )
    return raw


def _complete_validated_formal_attempt(
    *,
    repository: Path,
    output: Path,
    registration: RegisteredAttempt,
    registration_head: str,
    registry_bytes: bytes,
    raw: dict[str, Any],
    recovered_operations: int,
    recovering: bool = False,
) -> dict[str, Any]:
    from .formal_publish import (
        FormalPublishError,
        finalize_formal_publication,
        install_formal_publication,
        recover_formal_publication,
        verify_installed_publication,
    )

    _validate_registration_checkout(
        repository,
        registration_head=registration_head,
        registration=registration,
    )
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise FormalAttemptCommandError("formal report provenance is missing")
    registration_sha256 = hashlib.sha256(registry_bytes).hexdigest()
    if (
        provenance.get("formal_attempt_id") != registration.attempt_id
        or provenance.get("formal_attempt_protocol_sha256")
        != registration.protocol_sha256
        or provenance.get("formal_attempt_registry_sha256")
        != registration_sha256
    ):
        raise FormalAttemptCommandError(
            "formal report differs from its open registration"
        )
    try:
        if recovering:
            publication = recover_formal_publication(
                repository=repository,
                output=output,
                registration=registration,
                registration_registry_bytes=registry_bytes,
                raw_report=raw,
            )
        else:
            publication = install_formal_publication(
                repository=repository,
                output=output,
                registration=registration,
                registration_registry_bytes=registry_bytes,
                raw_report=raw,
            )
        publication = verify_installed_publication(
            repository=repository,
            registration=registration,
            registration_registry_bytes=registry_bytes,
            raw_report=raw,
        )
    except (FormalPublishError, OSError, ValueError) as error:
        raise FormalAttemptCommandError(
            f"formal publication could not be installed safely: {error}"
        ) from error

    current = _registry_bytes(repository)
    if current == publication.registry_before_bytes:
        _validate_registration_checkout(
            repository,
            registration_head=registration_head,
            registration=registration,
        )
        _registry_after, updated = _append_event(
            repository,
            expected=publication.registry_before_bytes,
            event=publication.completion,
        )
        if updated != publication.registry_after_bytes:
            raise FormalAttemptCommandError(
                "formal COMPLETED registry bytes differ from the publication journal"
            )
    elif current == publication.registry_after_bytes:
        updated = current
    else:
        raise FormalAttemptCommandError(
            "formal registry differs from both publication transaction states"
        )
    try:
        finalized = finalize_formal_publication(
            repository=repository,
            attempt_id=registration.attempt_id,
            observed_registry_bytes=updated,
        )
    except FormalPublishError as error:
        raise FormalAttemptCommandError(
            f"formal publication completion could not be finalized: {error}"
        ) from error
    if finalized.action != "COMMITTED_AND_CLEANED":
        raise FormalAttemptCommandError(
            "formal publication was rolled back instead of completed"
        )
    return {
        "state": "COMPLETED",
        "attempt_id": registration.attempt_id,
        "protocol_sha256": registration.protocol_sha256,
        "registration_registry_sha256": registration_sha256,
        "current_registry_sha256": hashlib.sha256(updated).hexdigest(),
        "report_sha256": publication.report_sha256,
        "report_fingerprint": publication.report_fingerprint,
        "scheduled_cells": _FORMAL_TASK_COUNT * _FORMAL_REPETITIONS,
        "recovered_operations": recovered_operations,
        "published_files": publication.evidence_files,
        "published_bytes": publication.evidence_bytes,
        "published_horizon_files": publication.horizon_files,
        "published_horizon_bytes": publication.horizon_bytes,
    }


def complete_formal_attempt(
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    repository = _repository_root(repo_root)
    with formal_attempt_lock(repository):
        return _complete_formal_attempt_locked(repo_root=repository)


def _pre_witness_output_is_clean(output: Path) -> bool:
    """Whether local bytes prove that no model cell could have started."""
    if not output.exists() and not output.is_symlink():
        return True
    if output.is_symlink() or not output.is_dir():
        return False
    try:
        entries = list(output.iterdir())
    except OSError:
        return False
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError:
            return False
        if entry.name in {
            _FORMAL_OUTPUT_LOCK_NAME,
            _FORMAL_RUN_HEADER_NAME,
        }:
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                return False
            continue
        if entry.name in {"active-operations", "active-container-ids"}:
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
            ):
                return False
            try:
                if next(entry.iterdir(), None) is None:
                    continue
            except OSError:
                return False
        return False
    return True


def _partial_cell_counts(
    repository: Path,
    registration: RegisteredAttempt,
) -> tuple[Literal["known", "evidence_missing"], int | None, int | None]:
    try:
        validate_formal_witness_remote_url(registration.witness_remote_url)
    except ValueError as error:
        raise FormalAttemptCommandError(
            "open registration does not have a public HTTPS witness URL"
        ) from error
    output = repository / registration.output_path
    results = output / "results"
    header = output / _FORMAL_RUN_HEADER_NAME
    remote_line = _anonymous_git_output(
        [
            "ls-remote",
            "--refs",
            registration.witness_remote_url,
            registration.witness_ref,
        ],
        label="inspect the formal attempt witness",
    ).strip()
    if not header.exists() and not header.is_symlink():
        if remote_line:
            return "evidence_missing", None, None
        return (
            ("known", 0, 0)
            if _pre_witness_output_is_clean(output)
            else ("evidence_missing", None, None)
        )
    try:
        header_bytes = _read_bounded_bytes(header, max_bytes=16 * 1024)
        header_raw = json.loads(header_bytes)
        registry_sha256 = hashlib.sha256(_registry_bytes(repository)).hexdigest()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        if remote_line or not _pre_witness_output_is_clean(output):
            return "evidence_missing", None, None
        return "known", 0, 0
    if (
        not isinstance(header_raw, dict)
        or set(header_raw)
        != {
            "schema_version",
            "formal_attempt_id",
            "registration_registry_sha256",
            "protocol_sha256",
            "outcome_key",
        }
        or header_raw.get("schema_version") != _FORMAL_RUN_HEADER_SCHEMA
        or header_raw.get("formal_attempt_id") != registration.attempt_id
        or header_raw.get("registration_registry_sha256") != registry_sha256
        or header_raw.get("protocol_sha256") != registration.protocol_sha256
        or not isinstance(header_raw.get("outcome_key"), str)
        or _HEX_64.fullmatch(header_raw["outcome_key"]) is None
        or header_bytes != _canonical_json_object_bytes(header_raw)
    ):
        if remote_line or not _pre_witness_output_is_clean(output):
            return "evidence_missing", None, None
        return "known", 0, 0
    if not remote_line and _pre_witness_output_is_clean(output):
        return "known", 0, 0
    head = _git_output(
        repository,
        ["rev-parse", "--verify", "HEAD"],
        label="resolve the registration commit",
    ).strip()
    tree = _git_output(
        repository,
        ["rev-parse", f"{head}^{{tree}}"],
        label="resolve the registration tree",
    ).strip()
    try:
        message = formal_ablation_witness_message(
            attempt_id=registration.attempt_id,
            registration_registry_sha256=registry_sha256,
            protocol_sha256=registration.protocol_sha256,
            outcome_key=header_raw["outcome_key"],
            run_header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        )
        expected_witness = formal_ablation_witness_commit_oid(
            formal_ablation_witness_commit_bytes(
                tree=tree,
                parent=head,
                message=message,
            )
        )
    except ValueError:
        return "evidence_missing", None, None
    if remote_line != f"{expected_witness}\t{registration.witness_ref}":
        if not remote_line and _pre_witness_output_is_clean(output):
            return "known", 0, 0
        return "evidence_missing", None, None
    if not results.exists() and not results.is_symlink():
        return "evidence_missing", None, None
    if results.is_symlink() or not results.is_dir():
        return "evidence_missing", None, None
    try:
        manifest, manifest_sha256 = _load_formal_corpus_manifest(
            repository / _FORMAL_CORPUS_MANIFEST_PATH,
            repository,
        )
        if manifest_sha256 != registration.manifest_sha256:
            raise ValueError("formal manifest does not match the registration")
        manifest_entries = {
            entry["name"]: entry
            for entry in manifest["tasks"]
        }
        names = list(manifest_entries)
        entries = list(results.iterdir())
    except (OSError, TypeError, ValueError):
        return "evidence_missing", None, None
    expected_started = {
        f"{task}__r{rep}.started.json"
        for task in names
        for rep in range(_FORMAL_REPETITIONS)
    }
    expected_terminal = {
        f"{task}__r{rep}.json"
        for task in names
        for rep in range(_FORMAL_REPETITIONS)
    }
    actual = {entry.name for entry in entries}
    if not actual or not actual <= expected_started | expected_terminal:
        return "evidence_missing", None, None
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError:
            return "evidence_missing", None, None
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > _MAX_RESULT_BYTES
        ):
            return "evidence_missing", None, None
    formal_fields = {
        "formal_attempt_id": registration.attempt_id,
        "formal_registration_registry_sha256": registry_sha256,
        "formal_protocol_sha256": registration.protocol_sha256,
        "formal_outcome_key": header_raw["outcome_key"],
    }
    started_bindings: dict[tuple[str, int], dict[str, Any]] = {}
    for task in names:
        for rep in range(_FORMAL_REPETITIONS):
            start_name = f"{task}__r{rep}.started.json"
            terminal_name = f"{task}__r{rep}.json"
            if start_name in actual:
                try:
                    payload = _read_bounded_bytes(
                        results / start_name,
                        max_bytes=_MAX_RESULT_BYTES,
                    )
                    marker = json.loads(payload)
                except (
                    OSError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    ValueError,
                ):
                    return "evidence_missing", None, None
                if (
                    not isinstance(marker, dict)
                    or set(marker)
                    != {
                        "schema_version",
                        "task",
                        "rep",
                        "cell_fingerprint",
                        "input_snapshot_sha256",
                        *formal_fields,
                    }
                    or marker.get("schema_version") != _CELL_ATTEMPT_SCHEMA
                    or marker.get("task") != task
                    or marker.get("rep") != rep
                    or any(
                        marker.get(field) != value
                        for field, value in formal_fields.items()
                    )
                    or any(
                        not isinstance(marker.get(field), str)
                        or _HEX_64.fullmatch(marker[field]) is None
                        for field in (
                            "cell_fingerprint",
                            "input_snapshot_sha256",
                        )
                    )
                    or payload != _canonical_json_object_bytes(marker)
                ):
                    return "evidence_missing", None, None
                manifest_entry = manifest_entries[task]
                if not _formal_input_snapshot_is_valid(
                    output
                    / "input_snapshots"
                    / marker["input_snapshot_sha256"],
                    task=task,
                    task_sha256=manifest_entry["task_sha256"],
                    corpus_sha256=manifest_entry["corpus_sha256"],
                    snapshot_sha256=marker["input_snapshot_sha256"],
                ):
                    return "evidence_missing", None, None
                started_bindings[(task, rep)] = marker
            if terminal_name not in actual:
                continue
            marker = started_bindings.get((task, rep))
            if marker is None:
                return "evidence_missing", None, None
            cached = _load_cached_cell(
                results / terminal_name,
                marker["cell_fingerprint"],
                input_snapshot_sha256=marker["input_snapshot_sha256"],
                scorer_backend="docker",
                scorer_image_id=registration.docker_image_id,
                require_call_receipts=True,
                expected_task=task,
                expected_rep=rep,
                receipt_dir=output / "llm_call_receipts",
                max_inner_attempts=registration.codex_client.max_retries + 1,
                formal_evidence=True,
                expected_formal_binding=formal_fields,
            )
            if (
                cached is None
                or len(cached.records) != 3
                or {record.condition for record in cached.records}
                != {"trust", "gate", "verify"}
                or any(
                    record.task != task or record.rep != rep
                    for record in cached.records
                )
            ):
                return "evidence_missing", None, None
    return (
        "known",
        len(started_bindings),
        len(actual & expected_terminal),
    )


def _abandon_formal_attempt_locked(
    *,
    repo_root: str | Path,
    reason_code: str,
    reason: str,
) -> dict[str, Any]:
    """Append an explicit terminal record for a consumed or unusable attempt."""
    if _REASON_CODE.fullmatch(reason_code) is None:
        raise FormalAttemptCommandError(
            "abandon reason-code must be lowercase snake_case"
        )
    if not reason or reason.strip() != reason:
        raise FormalAttemptCommandError("abandon reason must be explicit")
    repository = _repository_root(repo_root)
    _require_no_publication_transaction(
        repository,
        action="abandon a formal attempt",
    )
    _head, registry_bytes, registration = _open_registration_at_clean_head(
        repository
    )
    output = repository / registration.output_path
    report_path = output / "ablation_report.json"
    if report_path.exists() or report_path.is_symlink():
        raise FormalAttemptCommandError(
            "formal report path exists; use complete or quarantine the evidence"
        )
    try:
        with _formal_ablation_lock(output):
            recovered_operations = _recover_formal_operations(
                registration,
                output,
                allow_recovered=True,
            )
            progress_status, started_cells, terminal_cells = _partial_cell_counts(
                repository,
                registration,
            )
            if report_path.exists() or report_path.is_symlink():
                raise FormalAttemptCommandError(
                    "formal report path exists; use complete or quarantine the evidence"
                )
            abandonment = AbandonedAttempt(
                attempt_id=registration.attempt_id,
                recorded_at=now().isoformat(),
                progress_status=progress_status,
                started_cells=started_cells,
                terminal_cells=terminal_cells,
                reason_code=reason_code,
                reason=reason,
            )
            _registry_after, updated = _append_event(
                repository,
                expected=registry_bytes,
                event=abandonment,
            )
    except RuntimeError as error:
        if isinstance(error, FormalAttemptCommandError):
            raise
        raise FormalAttemptCommandError(
            "formal output lock could not be held"
        ) from error
    return {
        "state": "ABANDONED",
        "attempt_id": registration.attempt_id,
        "progress_status": progress_status,
        "started_cells": started_cells,
        "terminal_cells": terminal_cells,
        "reason_code": reason_code,
        "recovered_operations": recovered_operations,
        "current_registry_sha256": hashlib.sha256(updated).hexdigest(),
    }


def abandon_formal_attempt(
    *,
    repo_root: str | Path,
    reason_code: str,
    reason: str,
) -> dict[str, Any]:
    repository = _repository_root(repo_root)
    with formal_attempt_lock(repository):
        return _abandon_formal_attempt_locked(
            repo_root=repository,
            reason_code=reason_code,
            reason=reason,
        )


def formal_attempt_status(*, repo_root: str | Path) -> dict[str, Any]:
    """Inspect commit, witness, and output readiness without writing state."""
    repository = _repository_root(repo_root)
    payload = _registry_bytes(repository)
    try:
        registry = parse_formal_ablation_attempt_registry(payload)
    except ValueError as error:
        raise FormalAttemptCommandError(
            "formal attempt registry is invalid"
        ) from error
    registration = registry.open_registration()
    last = registry.events[-1] if registry.events else None
    publication = _publication_inspection(repository)
    if publication.status != "CLEAN":
        return {
            "schema_version": registry.schema_version,
            "event_count": len(registry.events),
            "registry_sha256": hashlib.sha256(payload).hexdigest(),
            "state": publication.status,
            "attempt_id": (
                publication.attempt_id
                or getattr(registration, "attempt_id", None)
                or getattr(last, "attempt_id", None)
            ),
            "publication_transaction_state": publication.transaction_state,
            "reason": publication.reason,
            "ready_to_run": False,
        }
    registry_matches_head = False
    head: str | None = None
    branch: str | None = None
    try:
        head = _git_output(
            repository,
            ["rev-parse", "--verify", "HEAD"],
            label="resolve status HEAD",
        ).strip()
        branch = _git_output(
            repository,
            ["symbolic-ref", "--quiet", "--short", "HEAD"],
            label="resolve status branch",
        ).strip()
        committed = _git_output(
            repository,
            [
                "show",
                f"{head}:{FORMAL_ABLATION_ATTEMPTS_PATH.as_posix()}",
            ],
            label="read status registry at HEAD",
        ).encode("utf-8")
        registry_matches_head = committed == payload
    except (FormalAttemptCommandError, RuntimeError):
        pass

    if registration is None:
        pending = not registry_matches_head and last is not None
        return {
            "schema_version": registry.schema_version,
            "event_count": len(registry.events),
            "registry_sha256": hashlib.sha256(payload).hexdigest(),
            "registry_matches_head": registry_matches_head,
            "state": (
                "TERMINAL_PENDING_COMMIT"
                if pending
                else last.event if last is not None else "EMPTY"
            ),
            "attempt_id": getattr(last, "attempt_id", None),
            "ready_to_run": False,
        }

    registration_commit_valid = False
    if registry_matches_head and head is not None:
        try:
            parents = _git_output(
                repository,
                ["rev-list", "--parents", "-n", "1", head],
                label="inspect status registration parent",
            ).strip().split()
            changed = _git_output(
                repository,
                ["diff-tree", "--no-commit-id", "--name-only", "-r", head],
                label="inspect status registration paths",
            ).splitlines()
            registration_commit_valid = (
                parents == [head, registration.source_commit]
                and changed == [FORMAL_ABLATION_ATTEMPTS_PATH.as_posix()]
            )
        except FormalAttemptCommandError:
            pass

    remote_status = "unavailable"
    witness_status = "unavailable"
    witness_line = ""
    witness_url: str | None = None
    try:
        git_path = str(_trusted_control_executable("git")["path"])
        configured_url = _formal_witness_remote_url(
            git_path,
            repository_root=repository,
            remote_name=registration.witness_remote_name,
        )
        if (
            configured_url == registration.witness_remote_url
            and head is not None
            and branch is not None
        ):
            if registration.witness_credential_helper is None:
                raise FormalAttemptCommandError(
                    "status registration has no credential helper"
                )
            _witness_credential_helper(
                configured_url,
                expected=registration.witness_credential_helper,
            )
            witness_url = configured_url
            remote_branch = _anonymous_git_output(
                [
                    "ls-remote",
                    "--heads",
                    witness_url,
                    f"refs/heads/{branch}",
                ],
                label="inspect status registration branch",
            ).strip()
            fields = remote_branch.split()
            if fields == [head, f"refs/heads/{branch}"]:
                remote_status = "exact"
            elif fields == [
                registration.source_commit,
                f"refs/heads/{branch}",
            ]:
                remote_status = "source_only"
            elif remote_branch:
                remote_status = "changed"
            else:
                remote_status = "absent"
            witness_line = _anonymous_git_output(
                [
                    "ls-remote",
                    "--refs",
                    witness_url,
                    registration.witness_ref,
                ],
                label="inspect status witness ref",
            ).strip()
            witness_status = "present" if witness_line else "absent"
        else:
            remote_status = "changed"
            witness_status = "changed"
    except (FormalAttemptCommandError, RuntimeError):
        pass

    output = repository / registration.output_path
    unsafe_output = False
    for candidate in (
        repository / "runs",
        repository / "runs" / "formal_ablation",
        output,
    ):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            unsafe_output = True
            break
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            unsafe_output = True
            break
    report_path = output / "ablation_report.json"
    header_path = output / _FORMAL_RUN_HEADER_NAME
    if unsafe_output:
        output_status = "quarantined"
    elif report_path.exists() or report_path.is_symlink():
        if report_path.is_symlink():
            output_status = "quarantined"
        elif (
            registry_matches_head
            and registration_commit_valid
        ):
            output_status = "report_present"
            from .release_claims import validate_formal_ablation_output

            try:
                validate_formal_ablation_output(
                    output,
                    repo_root=repository,
                )
            except ValueError:
                output_status = "quarantined"
            else:
                output_status = "complete_ready"
        else:
            output_status = "report_present"
    elif header_path.exists() or header_path.is_symlink() or witness_line:
        try:
            progress_status, _started, _terminal = _partial_cell_counts(
                repository,
                registration,
            )
        except FormalAttemptCommandError:
            output_status = "quarantined"
            witness_status = "changed"
        else:
            if progress_status == "evidence_missing":
                output_status = "abandon_required"
                witness_status = "present" if witness_line else "absent"
            elif not witness_line:
                output_status = "abandon_required"
                witness_status = "absent"
            else:
                output_status = "started"
                witness_status = "exact"
    elif output.exists():
        output_status = "abandon_required"
    else:
        output_status = "absent"

    ready_to_run = (
        registry_matches_head
        and registration_commit_valid
        and remote_status == "exact"
        and witness_status == "absent"
        and output_status == "absent"
    )
    if not registry_matches_head or not registration_commit_valid:
        state = "PENDING_COMMIT"
    elif output_status == "complete_ready":
        state = "COMPLETE_READY"
    elif output_status == "started":
        state = "STARTED"
    elif output_status == "abandon_required":
        state = "ABANDON_REQUIRED"
    elif output_status == "quarantined" or witness_status == "changed":
        state = "QUARANTINED"
    elif ready_to_run:
        state = "REGISTERED_READY"
    elif (
        registry_matches_head
        and registration_commit_valid
        and remote_status == "source_only"
        and witness_status == "absent"
        and output_status == "absent"
    ):
        state = "REGISTERED_PENDING_PUSH"
    else:
        state = "REGISTERED_NOT_READY"
    response: dict[str, Any] = {
        "schema_version": registry.schema_version,
        "event_count": len(registry.events),
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
        "registry_matches_head": registry_matches_head,
        "registration_commit_valid": registration_commit_valid,
        "registration_remote_status": remote_status,
        "witness_status": witness_status,
        "output_status": output_status,
        "state": state,
        "ready_to_run": ready_to_run,
        "attempt_id": registration.attempt_id,
        "protocol_sha256": registration.protocol_sha256,
        "source_commit": registration.source_commit,
        "output_path": registration.output_path,
        "docker_image_id": registration.docker_image_id,
        "witness_remote_url": witness_url or registration.witness_remote_url,
    }
    return response
