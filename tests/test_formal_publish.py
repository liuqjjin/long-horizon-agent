from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import lha.formal_publish as publish
from lha.ablation import (
    _input_snapshot_digest,
    _repo_digest,
    _source_file_digests,
    _source_tree_digest,
)
from lha.ablation_attempts import (
    FormalAblationAttemptRegistry,
    FormalAblationProtocol,
    FormalCodexClientConfig,
    FormalGitCredentialHelper,
    RegisteredAttempt,
    formal_ablation_attempt_registry_bytes,
    formal_ablation_protocol_sha256,
    formal_codex_client_sha256,
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _synthetic_trusted_source_guard(
    repository: Path,
    _registration: RegisteredAttempt,
) -> None:
    result = subprocess.run(
        ["git", "show", "HEAD:src/lha/sentinel.py"],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    if (repository / "src" / "lha" / "sentinel.py").read_bytes() != result.stdout:
        raise publish.FormalPublishError(
            "formal checkout differs from the trusted HEAD inputs"
        )


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass
class _Fixture:
    root: Path
    output: Path
    registration: RegisteredAttempt
    registry_before: bytes
    raw: dict[str, Any]
    legacy_report: bytes


def _make_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    monkeypatch.setattr(publish, "_FORMAL_TASK_COUNT", 2)
    monkeypatch.setattr(publish, "_FORMAL_REPETITIONS", 2)
    root = tmp_path / "repo"
    (root / "src" / "lha").mkdir(parents=True)
    (root / "src" / "lha" / "sentinel.py").write_text("VALUE = 1\n")
    benchmarks = root / "benchmarks"
    benchmarks.mkdir()
    legacy_report = b'{"schema_version":2,"legacy":true}\n'
    (benchmarks / "ablation_report.json").write_bytes(legacy_report)
    (benchmarks / "ablation_report.md").write_text("# legacy ablation\n")
    (benchmarks / "horizon_report.json").write_text('{"legacy":true}\n')
    (benchmarks / "horizon_report.md").write_text("# legacy horizon\n")
    (benchmarks / "horizon_curve.svg").write_text("<svg><!-- legacy --></svg>\n")
    (root / ".gitignore").write_text("runs/\n")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")

    task_names = ["task_one", "task_two"]
    task_paths: dict[str, str] = {}
    corpus_paths: dict[str, str] = {}
    task_sha: dict[str, str] = {}
    corpus_sha: dict[str, str] = {}
    snapshot_sha: dict[str, str] = {}
    for task in task_names:
        task_path = root / "data" / "tasks" / f"{task}.yaml"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text(f"kind: issue_to_pr\ntitle: {task}\ntarget_repo: data/bench/{task}\n")
        corpus = root / "data" / "bench" / task
        corpus.mkdir(parents=True)
        (corpus / "module.py").write_text(f"TASK = {task!r}\n")
        task_paths[task] = task_path.relative_to(root).as_posix()
        corpus_paths[task] = corpus.relative_to(root).as_posix()
        task_sha[task] = _digest(task_path.read_bytes())
        corpus_sha[task] = _repo_digest(corpus)
        snapshot_sha[task] = _input_snapshot_digest(
            task_sha[task],
            corpus_sha[task],
        )

    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "config", "user.email", "fixture@example.com")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "source")
    source_commit = _git(root, "rev-parse", "HEAD")
    source_tree_sha = _source_tree_digest(_source_file_digests(root / "src" / "lha"))
    client = FormalCodexClientConfig(
        no_tools=True,
        sandbox_mode="read-only",
        permission_model="profile",
        permission_profile="lha-read",
        credential_barrier="verified",
        externally_sandboxed=False,
        max_retries=2,
        timeout_s=300.0,
        retry_backoff_s=1.0,
    )
    credential_helper = FormalGitCredentialHelper(
        host="example.com",
        executable_path="/usr/bin/gh",
        executable_sha256="7" * 64,
        version="gh version 1.0",
        command="!/usr/bin/gh auth git-credential",
    )
    protocol = FormalAblationProtocol(
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha,
        manifest_sha256="1" * 64,
        model="gpt-test",
        reasoning_effort="low",
        docker_image_id="sha256:" + "2" * 64,
        codex_cli_version="codex 1.0",
        codex_cli_executable_sha256="3" * 64,
        codex_client=client,
        codex_client_sha256=formal_codex_client_sha256(client),
        witness_credential_helper=credential_helper,
    )
    attempt_id = "4" * 64
    registration = RegisteredAttempt(
        attempt_id=attempt_id,
        protocol_sha256=formal_ablation_protocol_sha256(protocol),
        source_commit=protocol.source_commit,
        source_tree_sha256=protocol.source_tree_sha256,
        manifest_sha256=protocol.manifest_sha256,
        output_path=f"runs/formal_ablation/{attempt_id}",
        model=protocol.model,
        reasoning_effort=protocol.reasoning_effort,
        docker_image_id=protocol.docker_image_id,
        codex_cli_version=protocol.codex_cli_version,
        codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
        codex_client=protocol.codex_client,
        codex_client_sha256=protocol.codex_client_sha256,
        witness_credential_helper=protocol.witness_credential_helper,
        witness_remote_name="origin",
        witness_remote_url="https://example.com/owner/repo.git",
        registered_at="2026-07-29T00:00:00+00:00",
    )
    registry_before = formal_ablation_attempt_registry_bytes(
        FormalAblationAttemptRegistry(events=(registration,))
    )
    (benchmarks / "formal_ablation_attempts.json").write_bytes(registry_before)
    _git(root, "add", "benchmarks/formal_ablation_attempts.json")
    _git(root, "commit", "-qm", "register")
    registration_commit = _git(root, "rev-parse", "HEAD")

    output = root / registration.output_path
    output.mkdir(parents=True)
    (output / ".formal-ablation.lock").write_bytes(b"")
    for name in (
        "input_snapshots",
        "artifacts",
        "scorer_evidence",
        "llm_call_receipts",
        "results",
        "active-operations",
        "active-container-ids",
    ):
        (output / name).mkdir()
    header = json.dumps(
        {
            "schema_version": 1,
            "formal_attempt_id": attempt_id,
            "registration_registry_sha256": _digest(registry_before),
            "protocol_sha256": registration.protocol_sha256,
            "outcome_key": "5" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    (output / "formal_run.json").write_bytes(header)

    artifact_digests: set[str] = set()
    scorer_digests: set[str] = set()
    receipt_digests: set[str] = set()
    records: list[dict[str, Any]] = []
    llm_calls: list[dict[str, Any]] = []
    for task in task_names:
        snapshot = output / "input_snapshots" / snapshot_sha[task]
        snapshot.mkdir()
        task_bytes = (root / task_paths[task]).read_bytes()
        (snapshot / "task.yaml").write_bytes(task_bytes)
        shutil.copytree(root / corpus_paths[task], snapshot / "repo")
        (snapshot / "snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": task,
                    "task_sha256": task_sha[task],
                    "corpus_sha256": corpus_sha[task],
                    "snapshot_sha256": snapshot_sha[task],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for rep in range(2):
            artifact = json.dumps(
                {"task": task, "rep": rep},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            artifact_digest = _digest(artifact)
            artifact_digests.add(artifact_digest)
            (output / "artifacts" / f"{artifact_digest}.json").write_bytes(artifact)
            evidence = json.dumps(
                {"task": task, "rep": rep, "artifact": artifact_digest},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            evidence_digest = _digest(evidence)
            scorer_digests.add(evidence_digest)
            (output / "scorer_evidence" / f"{evidence_digest}.json").write_bytes(evidence)
            receipt = json.dumps(
                {"task": task, "rep": rep, "call": 0},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            receipt_digest = _digest(receipt)
            receipt_digests.add(receipt_digest)
            (output / "llm_call_receipts" / f"{receipt_digest}.json").write_bytes(receipt)
            llm_calls.append(
                {
                    "task": task,
                    "rep": rep,
                    "ordinal": 0,
                    "receipt_sha256": receipt_digest,
                    "cache_hit": False,
                }
            )
            (output / "results" / f"{task}__r{rep}.started.json").write_text(
                json.dumps({"task": task, "rep": rep, "started": True})
            )
            (output / "results" / f"{task}__r{rep}.json").write_text(
                json.dumps({"task": task, "rep": rep, "terminal": True})
            )
            for condition in ("trust", "gate", "verify"):
                correct = condition == "verify" or not (task == "task_two" and rep == 0)
                records.append(
                    {
                        "task": task,
                        "condition": condition,
                        "rep": rep,
                        "status": "DONE",
                        "claimed_success": correct,
                        "artifact_correct": correct,
                        "true_success": correct,
                        "false_success": False,
                        "repairs": 0,
                        "artifact_sha256": artifact_digest,
                        "scorer_evidence_sha256": evidence_digest,
                    }
                )

    raw: dict[str, Any] = {
        "schema_version": 4,
        "llm": "codex_cli",
        "model": "gpt-test",
        "reps": 2,
        "tasks": task_names,
        "scorer": "docker",
        "fingerprint": "6" * 64,
        "provenance": {
            "formal_attempt_id": attempt_id,
            "formal_attempt_protocol_sha256": registration.protocol_sha256,
            "formal_attempt_registry_sha256": _digest(registry_before),
            "formal_attempt_registration_commit": registration_commit,
            "formal_run_header_sha256": _digest(header),
            "task_paths": task_paths,
            "corpus_paths": corpus_paths,
            "task_files_sha256": task_sha,
            "corpus_sha256": corpus_sha,
            "input_snapshot_sha256": snapshot_sha,
        },
        "artifact_store": {
            "path": "artifacts",
            "count": len(artifact_digests),
        },
        "scorer_evidence_store": {
            "path": "scorer_evidence",
            "count": len(scorer_digests),
        },
        "llm_call_receipt_store": {
            "path": "llm_call_receipts",
            "count": len(receipt_digests),
        },
        "llm_calls": llm_calls,
        "records": records,
    }
    (output / "ablation_report.json").write_text(json.dumps(raw, indent=2))
    (output / "ablation_report.md").write_text("# formal ablation\n")
    import lha.release_claims as release_claims

    monkeypatch.setattr(
        release_claims,
        "validate_formal_ablation_output",
        lambda _output, *, repo_root: (
            raw if Path(repo_root) == root else pytest.fail("wrong publication root")
        ),
    )
    # The transaction fixture uses a two-task synthetic corpus instead of the
    # independently tested 17-task formal manifest. Focused tests below replace
    # this seam to prove publication checks it before semantic validation.
    monkeypatch.setattr(
        publish,
        "_validate_trusted_formal_checkout",
        lambda _repository, _registration: None,
    )
    assert not _git(root, "status", "--porcelain")
    return _Fixture(
        root=root,
        output=output,
        registration=registration,
        registry_before=registry_before,
        raw=raw,
        legacy_report=legacy_report,
    )


class _InjectedCrash(RuntimeError):
    pass


class _CrashAt:
    def __init__(self, phase: str):
        self.phase = phase
        self.fired = False

    def __call__(self, phase: str) -> None:
        if not self.fired and phase == self.phase:
            self.fired = True
            raise _InjectedCrash(phase)


@pytest.mark.parametrize(
    "phase",
    [
        "after_prepared",
        "after_stage",
        "after_installing",
        "after_install:ablation_report.json",
        "before_installed",
        "after_installed",
    ],
)
def test_publication_recovers_idempotently_after_each_durable_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(_InjectedCrash):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
            fault_injector=_CrashAt(phase),
        )

    summary = publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    repeated = publish.verify_installed_publication(
        repository=fixture.root,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    assert summary.registry_after_bytes == repeated.registry_after_bytes
    assert summary.completion.recorded_at == repeated.completion.recorded_at
    assert publish.inspect_formal_publication(repository=fixture.root).status == "RECOVERY_REQUIRED"


def test_registry_after_recovery_does_not_require_runs_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    summary = publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    (fixture.root / "benchmarks" / "formal_ablation_attempts.json").write_bytes(
        summary.registry_after_bytes
    )
    shutil.rmtree(fixture.output)

    recovered = publish.recover_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    assert recovered.registry_already_appended is True
    finalized = publish.finalize_formal_publication(
        repository=fixture.root,
        attempt_id=fixture.registration.attempt_id,
        observed_registry_bytes=summary.registry_after_bytes,
    )
    assert finalized.action == "COMMITTED_AND_CLEANED"
    assert publish.inspect_formal_publication(repository=fixture.root).status == "CLEAN"


def test_definitely_absent_registry_append_rolls_back_legacy_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(_InjectedCrash):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
            fault_injector=_CrashAt("after_install:ablation_report.json"),
        )
    result = publish.finalize_formal_publication(
        repository=fixture.root,
        attempt_id=fixture.registration.attempt_id,
        observed_registry_bytes=fixture.registry_before,
    )
    assert result.action == "ROLLED_BACK_AND_CLEANED"
    assert (fixture.root / "benchmarks" / "ablation_report.json").read_bytes() == (
        fixture.legacy_report
    )
    assert not _git(fixture.root, "status", "--porcelain")


def test_crash_after_rollback_is_recognized_and_cleanup_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(_InjectedCrash):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
            fault_injector=_CrashAt("after_install:ablation_report.json"),
        )
    with pytest.raises(_InjectedCrash):
        publish.finalize_formal_publication(
            repository=fixture.root,
            attempt_id=fixture.registration.attempt_id,
            observed_registry_bytes=fixture.registry_before,
            fault_injector=_CrashAt("after_rollback"),
        )
    inspection = publish.inspect_formal_publication(repository=fixture.root)
    assert inspection.status == "RECOVERY_REQUIRED"
    result = publish.finalize_formal_publication(
        repository=fixture.root,
        attempt_id=fixture.registration.attempt_id,
        observed_registry_bytes=fixture.registry_before,
    )
    assert result.action == "ROLLED_BACK_AND_CLEANED"
    assert publish.inspect_formal_publication(repository=fixture.root).status == "CLEAN"


def test_uncertain_append_and_unknown_target_never_roll_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(_InjectedCrash):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
            fault_injector=_CrashAt("after_install:ablation_report.json"),
        )
    target = fixture.root / "benchmarks" / "ablation_report.json"
    target.write_bytes(b"user bytes after crash\n")
    with pytest.raises(publish.FormalPublishUncertainError):
        publish.finalize_formal_publication(
            repository=fixture.root,
            attempt_id=fixture.registration.attempt_id,
            observed_registry_bytes=None,
        )
    with pytest.raises(publish.FormalPublishError, match="third-party bytes"):
        publish.finalize_formal_publication(
            repository=fixture.root,
            attempt_id=fixture.registration.attempt_id,
            observed_registry_bytes=fixture.registry_before,
        )
    assert target.read_bytes() == b"user bytes after crash\n"
    assert publish.inspect_formal_publication(repository=fixture.root).status == "QUARANTINED"


@pytest.mark.parametrize(
    "relative",
    [
        "formal_run.json",  # no tracked predecessor: rollback would unlink it
        "ablation_report.json",  # tracked predecessor: rollback would replace it
    ],
)
def test_rollback_content_cas_preserves_bytes_injected_after_global_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    target = fixture.root / "benchmarks" / relative
    injected = b"third-party bytes after global preflight\n"
    original = publish._preflight_rollback_targets
    fired = False

    def preflight_then_inject(
        repository: Path,
        journal: publish._PublicationJournal,
    ) -> dict[str, publish.DirectoryIdentity]:
        nonlocal fired
        identities = original(repository, journal)
        if not fired:
            fired = True
            target.write_bytes(injected)
        return identities

    monkeypatch.setattr(
        publish,
        "_preflight_rollback_targets",
        preflight_then_inject,
    )
    with pytest.raises(publish.FormalPublishError):
        publish.finalize_formal_publication(
            repository=fixture.root,
            attempt_id=fixture.registration.attempt_id,
            observed_registry_bytes=fixture.registry_before,
        )

    assert target.read_bytes() == injected
    journal = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    assert journal.is_file()
    assert (fixture.root / "benchmarks" / ".formal-publish").is_dir()
    assert publish.inspect_formal_publication(repository=fixture.root).status == "QUARANTINED"


def test_rollback_preserves_unknown_empty_directory_injected_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    third_party = (
        fixture.root
        / "benchmarks"
        / "results"
        / "third-party-empty"
    )
    original = publish._preflight_rollback_targets

    def preflight_then_inject(
        repository: Path,
        journal: publish._PublicationJournal,
    ) -> dict[str, publish.DirectoryIdentity]:
        identities = original(repository, journal)
        third_party.mkdir()
        return identities

    monkeypatch.setattr(
        publish,
        "_preflight_rollback_targets",
        preflight_then_inject,
    )
    with pytest.raises(
        publish.FormalPublishError,
        match="extra directory",
    ):
        publish.finalize_formal_publication(
            repository=fixture.root,
            attempt_id=fixture.registration.attempt_id,
            observed_registry_bytes=fixture.registry_before,
        )

    assert third_party.is_dir()
    journal_path = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    assert journal_path.is_file()
    assert (fixture.root / "benchmarks" / ".formal-publish").is_dir()
    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"


def test_rollback_preserves_known_directory_replaced_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    results = fixture.root / "benchmarks" / "results"
    detached = fixture.root / "benchmarks" / "third-party-results"
    original = publish._preflight_rollback_targets

    def preflight_then_replace(
        repository: Path,
        journal: publish._PublicationJournal,
    ) -> dict[str, publish.DirectoryIdentity]:
        identities = original(repository, journal)
        results.rename(detached)
        results.mkdir()
        return identities

    monkeypatch.setattr(
        publish,
        "_preflight_rollback_targets",
        preflight_then_replace,
    )
    with pytest.raises(
        publish.FormalPublishError,
        match="identity changed",
    ):
        publish.finalize_formal_publication(
            repository=fixture.root,
            attempt_id=fixture.registration.attempt_id,
            observed_registry_bytes=fixture.registry_before,
        )

    assert results.is_dir()
    assert detached.is_dir()
    assert any(detached.iterdir())
    journal_path = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    assert journal_path.is_file()


@pytest.mark.parametrize(
    "relative",
    [
        "formal_run.json",  # missing target raced into the create path
        "ablation_report.json",  # tracked target changed before replacement
    ],
)
def test_install_content_cas_preserves_bytes_injected_before_target_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    target = fixture.root / "benchmarks" / relative
    injected = b"third-party bytes before install\n"
    fired = False

    def inject(phase: str) -> None:
        nonlocal fired
        if not fired and phase == f"before_install:{relative}":
            fired = True
            target.write_bytes(injected)

    with pytest.raises(publish.FormalPublishError, match="cannot install"):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
            fault_injector=inject,
        )

    assert fired is True
    assert target.read_bytes() == injected
    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"


def test_source_drift_during_install_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(_InjectedCrash):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
            fault_injector=_CrashAt("after_installing"),
        )
    (fixture.output / "ablation_report.md").write_text("# changed source\n")
    with pytest.raises(publish.FormalPublishError, match="rolled back"):
        publish.recover_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )
    assert (fixture.root / "benchmarks" / "ablation_report.json").read_bytes() == (
        fixture.legacy_report
    )
    assert publish.inspect_formal_publication(repository=fixture.root).status == "CLEAN"


@pytest.mark.parametrize(
    "index_flag",
    ["--assume-unchanged", "--skip-worktree"],
)
def test_publication_rejects_hidden_source_before_semantic_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_flag: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    source = fixture.root / "src" / "lha" / "sentinel.py"
    _git(
        fixture.root,
        "update-index",
        index_flag,
        "--",
        "src/lha/sentinel.py",
    )
    source.write_text("VALUE = 99\n")
    assert _git(fixture.root, "status", "--porcelain=v1") == ""

    import lha.release_claims as release_claims

    monkeypatch.setattr(
        publish,
        "_validate_trusted_formal_checkout",
        _synthetic_trusted_source_guard,
    )
    monkeypatch.setattr(
        release_claims,
        "validate_formal_ablation_output",
        lambda *_args, **_kwargs: pytest.fail(
            "semantic validation ran before trusted HEAD validation"
        ),
    )
    with pytest.raises(
        publish.FormalPublishError,
        match="trusted HEAD inputs",
    ):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )
    assert (fixture.root / "benchmarks" / "ablation_report.json").read_bytes() == (
        fixture.legacy_report
    )
    assert publish.inspect_formal_publication(repository=fixture.root).status == "CLEAN"


def test_verify_rechecks_trusted_head_immediately_before_registry_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    original_whitelist = publish._require_transaction_dirty_whitelist
    source = fixture.root / "src" / "lha" / "sentinel.py"

    def mutate_after_whitelist(
        repository: Path,
        journal: publish._PublicationJournal,
        *,
        allow_registry: bool,
    ) -> None:
        original_whitelist(
            repository,
            journal,
            allow_registry=allow_registry,
        )
        _git(
            repository,
            "update-index",
            "--assume-unchanged",
            "--",
            "src/lha/sentinel.py",
        )
        source.write_text("VALUE = 99\n")
        assert "src/lha/sentinel.py" not in _git(
            repository,
            "status",
            "--porcelain=v1",
        )

    monkeypatch.setattr(
        publish,
        "_validate_trusted_formal_checkout",
        _synthetic_trusted_source_guard,
    )
    monkeypatch.setattr(
        publish,
        "_require_transaction_dirty_whitelist",
        mutate_after_whitelist,
    )
    with pytest.raises(
        publish.FormalPublishError,
        match="trusted HEAD inputs",
    ):
        publish.verify_installed_publication(
            repository=fixture.root,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )
    assert (
        fixture.root / "benchmarks" / "formal_ablation_attempts.json"
    ).read_bytes() == fixture.registry_before


def test_publication_time_validator_and_second_collection_reject_seal_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    result = fixture.output / "results" / "task_one__r0.json"
    original = result.read_bytes()

    import lha.release_claims as release_claims

    def validate_then_change(
        output: Path,
        *,
        repo_root: Path,
    ) -> dict[str, Any]:
        assert Path(output) == fixture.output
        assert Path(repo_root) == fixture.root
        result.write_bytes(b'{"changed_after_validation":true}\n')
        return fixture.raw

    monkeypatch.setattr(
        release_claims,
        "validate_formal_ablation_output",
        validate_then_change,
    )
    with pytest.raises(
        publish.FormalPublishError,
        match="source changed during publication-time validation",
    ):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )
    assert result.read_bytes() != original
    assert (fixture.root / "benchmarks" / "ablation_report.json").read_bytes() == (
        fixture.legacy_report
    )
    assert publish.inspect_formal_publication(repository=fixture.root).status == "CLEAN"


def test_publication_rejects_validator_result_drift_before_creating_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    import lha.release_claims as release_claims

    changed = dict(fixture.raw)
    changed["fingerprint"] = "7" * 64
    monkeypatch.setattr(
        release_claims,
        "validate_formal_ablation_output",
        lambda _output, *, repo_root: (
            changed
            if Path(repo_root) == fixture.root
            else pytest.fail("wrong publication root")
        ),
    )

    with pytest.raises(
        publish.FormalPublishError,
        match="returned different formal report data",
    ):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )
    assert (fixture.root / "benchmarks" / "ablation_report.json").read_bytes() == (
        fixture.legacy_report
    )
    assert publish.inspect_formal_publication(repository=fixture.root).status == "CLEAN"


def test_recovery_uses_bound_mapping_without_repeating_full_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(_InjectedCrash):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
            fault_injector=_CrashAt("after_stage"),
        )
    result = fixture.output / "results" / "task_one__r0.json"
    result.write_bytes(b'{"tampered":true}\n')

    import lha.release_claims as release_claims

    def reject_tampered_output(
        _output: Path,
        *,
        repo_root: Path,
    ) -> dict[str, Any]:
        assert Path(repo_root) == fixture.root
        pytest.fail("recovery must not rerun the clean-worktree formal validator")

    monkeypatch.setattr(
        release_claims,
        "validate_formal_ablation_output",
        reject_tampered_output,
    )
    with pytest.raises(publish.FormalPublishError, match="rolled back"):
        publish.recover_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )
    assert (fixture.root / "benchmarks" / "ablation_report.json").read_bytes() == (
        fixture.legacy_report
    )
    assert publish.inspect_formal_publication(repository=fixture.root).status == "CLEAN"


def test_installed_registry_before_rederives_horizon_from_formal_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    relative = "horizon_curve.svg"
    injected = b"<svg>self-consistent journal tamper</svg>\n"
    target = fixture.root / "benchmarks" / relative
    stage = (
        fixture.root
        / "benchmarks"
        / ".formal-publish"
        / fixture.registration.attempt_id
        / "stage"
        / relative
    )
    target.write_bytes(injected)
    stage.write_bytes(injected)
    journal_path = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    journal = json.loads(journal_path.read_text())
    journal["desired"][relative] = {
        "sha256": _digest(injected),
        "size": len(injected),
    }
    journal_path.write_text(
        json.dumps(journal, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(publish.FormalPublishError, match="source changed"):
        publish.recover_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )
    assert target.read_bytes() == injected
    assert stage.read_bytes() == injected
    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"


def test_installed_registry_before_quarantines_missing_formal_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    shutil.rmtree(fixture.output)

    with pytest.raises(publish.FormalPublishError, match="output"):
        publish.recover_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )
    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"


def test_fresh_install_rejects_dirty_extra_and_existing_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    (fixture.root / "unrelated.txt").write_text("user work\n")
    with pytest.raises(publish.FormalPublishError, match="clean worktree"):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )
    (fixture.root / "unrelated.txt").unlink()
    journal_dir = fixture.root / ".git" / "lha-formal-publications"
    journal_dir.mkdir()
    (journal_dir / f"{'9' * 64}.json").write_text("{}")
    with pytest.raises(publish.FormalPublishError, match="existing or malformed"):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )


def test_initial_journal_write_once_preserves_raced_unknown_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    original = publish.anchored_write_once_bytes
    injected = b"unknown initial journal bytes\n"

    def race_initial_create(
        path: str | Path,
        data: bytes,
        *,
        anchor: str | Path,
        mode: int = 0o600,
    ) -> bool:
        target = Path(path)
        target.write_bytes(injected)
        target.chmod(0o600)
        return original(path, data, anchor=anchor, mode=mode)

    monkeypatch.setattr(
        publish,
        "anchored_write_once_bytes",
        race_initial_create,
    )
    with pytest.raises(publish.FormalPublishError, match="journal"):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )

    journal_path = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    assert journal_path.read_bytes() == injected
    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"


def test_journal_transition_content_cas_preserves_raced_unknown_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    original = publish.anchored_replace_bytes_if_current
    injected = b"unknown journal transition bytes\n"
    fired = False

    def race_transition(
        path: str | Path,
        data: bytes,
        *,
        anchor: str | Path,
        expected_current: tuple[bytes, ...],
        expected_missing: bool = False,
        mode: int | None = None,
    ) -> None:
        nonlocal fired
        target = Path(path)
        if not fired and target.parent.name == "lha-formal-publications":
            fired = True
            target.write_bytes(injected)
            target.chmod(0o600)
        original(
            path,
            data,
            anchor=anchor,
            expected_current=expected_current,
            expected_missing=expected_missing,
            mode=mode,
        )

    monkeypatch.setattr(
        publish,
        "anchored_replace_bytes_if_current",
        race_transition,
    )
    with pytest.raises(publish.FormalPublishError, match="journal"):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )

    journal_path = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    assert fired is True
    assert journal_path.read_bytes() == injected
    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"


@pytest.mark.parametrize("directory", ["stage", "backup"])
def test_workspace_write_cas_preserves_unknown_stage_or_backup_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    original = publish._write_workspace_file
    injected = b"unknown workspace bytes\n"
    injected_path: Path | None = None

    def inject_before_write(
        repository: Path,
        path: Path,
        payload: bytes,
    ) -> None:
        nonlocal injected_path
        if injected_path is None and directory in path.parts:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(injected)
            path.chmod(0o600)
            injected_path = path
        original(repository, path, payload)

    monkeypatch.setattr(publish, "_write_workspace_file", inject_before_write)
    with pytest.raises(publish.FormalPublishError, match="stage"):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )

    assert injected_path is not None
    assert injected_path.read_bytes() == injected
    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"


def test_journal_requires_private_owner_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(_InjectedCrash):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
            fault_injector=_CrashAt("after_prepared"),
        )
    journal_path = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    journal_path.chmod(0o644)

    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"
    with pytest.raises(publish.FormalPublishError, match="journal"):
        publish.recover_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("completion", "registration_registry_sha256"), "7" * 64),
        (("completion", "report_sha256"), "7" * 64),
        (("completion", "report_fingerprint"), "7" * 64),
        (("output_path",), "runs/formal_ablation/" + "7" * 64),
        (("source_tree_sha256",), "7" * 64),
        (("source_evidence_sha256",), "7" * 64),
        (("desired", "ablation_report.json", "sha256"), "7" * 64),
        (("semantically_validated",), False),
        (("semantically_validated",), None),
    ],
)
def test_journal_rejects_internal_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    value: object,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(_InjectedCrash):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
            fault_injector=_CrashAt("after_prepared"),
        )
    path = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    raw = json.loads(path.read_text())
    target = raw
    for field in field_path[:-1]:
        target = target[field]
    if value is None:
        target.pop(field_path[-1])
    else:
        target[field_path[-1]] = value
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")

    inspection = publish.inspect_formal_publication(repository=fixture.root)
    assert inspection.status == "QUARANTINED"
    assert path.is_file()


@pytest.mark.parametrize("mutation", ["new-entry", "changed-payload"])
def test_workspace_cleanup_preserves_entries_changed_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    summary = publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    (fixture.root / "benchmarks" / "formal_ablation_attempts.json").write_bytes(
        summary.registry_after_bytes
    )
    workspace = (
        fixture.root
        / "benchmarks"
        / ".formal-publish"
        / fixture.registration.attempt_id
    )
    changed = (
        workspace / "unexpected.bin"
        if mutation == "new-entry"
        else workspace / "stage" / "horizon_curve.svg"
    )
    injected = b"workspace bytes injected after verification\n"
    original = publish._verify_workspace_no_extras
    calls = 0

    def verify_then_inject(
        repository: Path,
        journal: publish._PublicationJournal,
    ) -> None:
        nonlocal calls
        original(repository, journal)
        calls += 1
        if calls == 1:
            changed.write_bytes(injected)

    monkeypatch.setattr(
        publish,
        "_verify_workspace_no_extras",
        verify_then_inject,
    )
    with pytest.raises(publish.FormalPublishError, match="workspace"):
        publish.cleanup_formal_publication(
            repository=fixture.root,
            attempt_id=fixture.registration.attempt_id,
        )

    assert changed.read_bytes() == injected
    journal_path = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    assert journal_path.is_file()
    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"


def test_journal_cleanup_content_cas_preserves_raced_unknown_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    summary = publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    (fixture.root / "benchmarks" / "formal_ablation_attempts.json").write_bytes(
        summary.registry_after_bytes
    )
    original = publish._remove_journal
    injected = b"unknown journal bytes before cleanup\n"

    def inject_before_unlink(
        repository: Path,
        journal: publish._PublicationJournal,
    ) -> None:
        path = (
            repository
            / ".git"
            / "lha-formal-publications"
            / f"{journal.attempt_id}.json"
        )
        path.write_bytes(injected)
        path.chmod(0o600)
        original(repository, journal)

    monkeypatch.setattr(publish, "_remove_journal", inject_before_unlink)
    with pytest.raises(publish.FormalPublishError, match="journal"):
        publish.cleanup_formal_publication(
            repository=fixture.root,
            attempt_id=fixture.registration.attempt_id,
        )

    journal_path = (
        fixture.root
        / ".git"
        / "lha-formal-publications"
        / f"{fixture.registration.attempt_id}.json"
    )
    assert journal_path.read_bytes() == injected
    assert publish.inspect_formal_publication(
        repository=fixture.root
    ).status == "QUARANTINED"


def test_safe_empty_journal_and_workspace_roots_are_clean_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    journal_root = fixture.root / ".git" / "lha-formal-publications"
    workspace_root = fixture.root / "benchmarks" / ".formal-publish"
    journal_root.mkdir(mode=0o700)
    workspace_root.mkdir(mode=0o700)

    assert publish.inspect_formal_publication(repository=fixture.root).status == "CLEAN"
    summary = publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    assert summary.state == "INSTALLED"


@pytest.mark.parametrize("kind", ["extra", "symlink", "hardlink", "fifo"])
def test_publication_rejects_extra_links_hardlinks_and_fifo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    artifacts = fixture.output / "artifacts"
    source = next(artifacts.iterdir())
    if kind == "extra":
        (artifacts / "extra.json").write_text("{}")
    elif kind == "symlink":
        (artifacts / "extra.json").symlink_to(source)
    elif kind == "hardlink":
        os.link(source, artifacts / "extra.json")
    else:
        os.mkfifo(artifacts / "extra.json")
    with pytest.raises(publish.FormalPublishError):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )


def test_snapshot_metadata_and_corpus_source_are_rechecked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    snapshot_root = fixture.output / "input_snapshots"
    snapshot = next(snapshot_root.iterdir())
    metadata = json.loads((snapshot / "snapshot.json").read_text())
    metadata["task"] = "other"
    (snapshot / "snapshot.json").write_text(json.dumps(metadata))
    with pytest.raises(publish.FormalPublishError, match="metadata"):
        publish.install_formal_publication(
            repository=fixture.root,
            output=fixture.output,
            registration=fixture.registration,
            registration_registry_bytes=fixture.registry_before,
            raw_report=fixture.raw,
        )


def test_horizon_is_fixed_to_seed_zero_and_benchmarks_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    summary = publish.install_formal_publication(
        repository=fixture.root,
        output=fixture.output,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    horizon = json.loads((fixture.root / "benchmarks" / "horizon_report.json").read_text())
    assert horizon["source"] == "benchmarks/ablation_report.json"
    first = (fixture.root / "benchmarks" / "horizon_report.json").read_bytes()
    verified = publish.verify_installed_publication(
        repository=fixture.root,
        registration=fixture.registration,
        registration_registry_bytes=fixture.registry_before,
        raw_report=fixture.raw,
    )
    assert first == (fixture.root / "benchmarks" / "horizon_report.json").read_bytes()
    assert verified.horizon_files == 3
    assert verified.registry_after_bytes == summary.registry_after_bytes
