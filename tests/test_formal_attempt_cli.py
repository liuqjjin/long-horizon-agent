from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lha.ablation as ablation
import lha.formal_attempt_cli as commands
import lha.release_claims as release_claims
from lha.ablation_attempts import (
    AbandonedAttempt,
    CompletedAttempt,
    FormalAblationAttemptRegistry,
    FormalAblationProtocol,
    FormalCodexClientConfig,
    FormalGitCredentialHelper,
    RegisteredAttempt,
    formal_ablation_attempt_registry_bytes,
    formal_ablation_protocol_sha256,
    formal_ablation_witness_commit_bytes,
    formal_ablation_witness_commit_oid,
    formal_ablation_witness_message,
    formal_attempt_lock,
    formal_codex_client_sha256,
    parse_formal_ablation_attempt_registry,
)
from lha.cli import build_parser
from lha.config import Config
from lha.formal_publish import (
    FormalPublishFinalizeSummary,
    FormalPublishInspection,
    FormalPublishSummary,
)

_TIME = "2026-07-29T09:00:00+08:00"


def _credential_helper() -> FormalGitCredentialHelper:
    path = "/opt/homebrew/bin/gh"
    return FormalGitCredentialHelper(
        host="github.com",
        executable_path=path,
        executable_sha256="8" * 64,
        version="gh version 2.92.0",
        command=f"!{path} auth git-credential",
    )


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / ".git").mkdir(parents=True)
    (root / "src" / "lha").mkdir(parents=True)
    (root / "benchmarks").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    registry = FormalAblationAttemptRegistry(events=())
    (root / "benchmarks" / "formal_ablation_attempts.json").write_bytes(
        formal_ablation_attempt_registry_bytes(registry)
    )
    return root


def _protocol() -> FormalAblationProtocol:
    client = FormalCodexClientConfig(
        max_retries=2,
        timeout_s=300.0,
        retry_backoff_s=1.0,
    )
    return FormalAblationProtocol(
        source_commit="a" * 40,
        source_tree_sha256="b" * 64,
        manifest_sha256="c" * 64,
        model="gpt-5.4-mini",
        reasoning_effort="low",
        docker_image_id="sha256:" + "d" * 64,
        codex_cli_version="codex-cli 0.141.0",
        codex_cli_executable_sha256="e" * 64,
        codex_client=client,
        codex_client_sha256=formal_codex_client_sha256(client),
        witness_credential_helper=_credential_helper(),
    )


def _registration(attempt_id: str = "f" * 64) -> RegisteredAttempt:
    protocol = _protocol()
    return RegisteredAttempt(
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
        witness_remote_name="formal-witness",
        witness_remote_url="https://github.com/example/lha.git",
        registered_at=_TIME,
    )


def _write_registry(root: Path, *events) -> bytes:
    payload = formal_ablation_attempt_registry_bytes(
        FormalAblationAttemptRegistry(events=events)
    )
    (root / "benchmarks" / "formal_ablation_attempts.json").write_bytes(payload)
    return payload


def test_cli_exposes_explicit_attempt_lifecycle():
    parser = build_parser()

    registered = parser.parse_args(
        [
            "ablation-attempt",
            "register",
            "--model",
            "gpt-5.4-mini",
            "--reasoning-effort",
            "low",
            "--docker-image-id",
            "sha256:" + "a" * 64,
        ]
    )
    abandoned = parser.parse_args(
        [
            "ablation-attempt",
            "abandon",
            "--reason-code",
            "operator_stopped",
            "--reason",
            "运行被明确终止",
        ]
    )

    assert registered.witness_remote == "formal-witness"
    assert abandoned.reason_code == "operator_stopped"
    assert parser.parse_args(["ablation-attempt", "status"]).attempt_cmd == "status"
    assert parser.parse_args(["ablation-attempt", "complete"]).attempt_cmd == "complete"


def test_status_is_read_only(tmp_path: Path):
    root = _repository(tmp_path)
    before = _write_registry(root, _registration())

    status = commands.formal_attempt_status(repo_root=root)

    assert status["state"] == "PENDING_COMMIT"
    assert status["registry_matches_head"] is False
    assert status["ready_to_run"] is False
    assert status["attempt_id"] == "f" * 64
    assert status["registry_sha256"] == hashlib.sha256(before).hexdigest()
    assert (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes() == before


def test_status_reports_a_recoverable_publication_before_git_probes(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    before = _write_registry(root, _registration())
    monkeypatch.setattr(
        commands,
        "_publication_inspection",
        lambda _repository: FormalPublishInspection(
            status="RECOVERY_REQUIRED",
            attempt_id="f" * 64,
            transaction_state="INSTALLING",
            reason="resume the exact publication transaction",
        ),
    )
    monkeypatch.setattr(
        commands,
        "_git_output",
        lambda *_args, **_kwargs: pytest.fail("status performed a Git probe"),
    )

    status = commands.formal_attempt_status(repo_root=root)

    assert status["state"] == "RECOVERY_REQUIRED"
    assert status["publication_transaction_state"] == "INSTALLING"
    assert status["ready_to_run"] is False
    assert (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes() == before


def test_state_changes_refuse_an_active_publication(tmp_path: Path, monkeypatch):
    root = _repository(tmp_path)
    monkeypatch.setattr(
        commands,
        "_publication_inspection",
        lambda _repository: FormalPublishInspection(
            status="RECOVERY_REQUIRED",
            attempt_id="f" * 64,
            transaction_state="PREPARED",
        ),
    )

    with pytest.raises(commands.FormalAttemptCommandError, match="recovery_required"):
        commands._require_no_publication_transaction(
            root,
            action="change formal attempt state",
        )


@pytest.mark.parametrize(
    ("remote_commit", "expected_state", "ready"),
    [
        ("source", "REGISTERED_PENDING_PUSH", False),
        ("registration", "REGISTERED_READY", True),
    ],
)
def test_status_requires_registration_commit_on_remote_branch(
    tmp_path: Path,
    monkeypatch,
    remote_commit: str,
    expected_state: str,
    ready: bool,
):
    root = _repository(tmp_path)
    registration = _registration()
    payload = _write_registry(root, registration)
    head = "1" * 40
    branch = "codex/formal"

    def git_output(_root, arguments, **_kwargs):
        if arguments[:2] == ["rev-parse", "--verify"]:
            return head
        if arguments[:3] == ["symbolic-ref", "--quiet", "--short"]:
            return branch
        if arguments[0] == "show":
            return payload.decode("utf-8")
        if arguments[0] == "rev-list":
            return f"{head} {registration.source_commit}"
        if arguments[0] == "diff-tree":
            return "benchmarks/formal_ablation_attempts.json\n"
        raise AssertionError(arguments)

    def anonymous(arguments, **_kwargs):
        if "--heads" in arguments:
            commit = (
                registration.source_commit
                if remote_commit == "source"
                else head
            )
            return f"{commit}\trefs/heads/{branch}\n"
        return ""

    monkeypatch.setattr(commands, "_git_output", git_output)
    monkeypatch.setattr(commands, "_anonymous_git_output", anonymous)
    monkeypatch.setattr(
        commands,
        "_formal_witness_remote_url",
        lambda *_args, **_kwargs: registration.witness_remote_url,
    )
    monkeypatch.setattr(
        commands,
        "_witness_credential_helper",
        lambda *_args, **_kwargs: registration.witness_credential_helper,
    )

    status = commands.formal_attempt_status(repo_root=root)

    assert status["state"] == expected_state
    assert status["ready_to_run"] is ready


def test_register_rejects_a_docker_tag_before_inspection(tmp_path: Path, monkeypatch):
    root = _repository(tmp_path)
    inspected = False

    def inspect(_image: str) -> str:
        nonlocal inspected
        inspected = True
        return ""

    monkeypatch.setattr(commands, "_resolve_docker_image_id", inspect)
    with pytest.raises(commands.FormalAttemptCommandError, match="not a tag"):
        commands.register_formal_attempt(
            repo_root=root,
            config=Config(),
            model="gpt-5.4-mini",
            reasoning_effort="low",
            docker_image_id="lha:release",
            witness_remote_name="formal-witness",
        )
    assert not inspected


def test_register_resolves_and_atomically_appends_every_input(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    initial = (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes()
    protocol = _protocol()
    image_id = protocol.docker_image_id
    monkeypatch.setattr(commands, "_resolve_docker_image_id", lambda image: image)
    monkeypatch.setattr(commands, "_probe_docker_image", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        commands,
        "_anonymous_git_output",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        commands,
        "_clean_head",
        lambda _root: (protocol.source_commit, "codex/release"),
    )
    monkeypatch.setattr(
        commands,
        "_committed_registry",
        lambda _root, *, head: (
            initial,
            parse_formal_ablation_attempt_registry(initial),
        ),
    )
    monkeypatch.setattr(
        commands,
        "_https_witness_remote",
        lambda *_args, **_kwargs: "https://github.com/example/lha.git",
    )
    helper_checks = 0

    def bind_helper(_url, *, expected=None):
        nonlocal helper_checks
        helper_checks += 1
        if expected is not None:
            assert expected == protocol.witness_credential_helper
        return protocol.witness_credential_helper

    monkeypatch.setattr(commands, "_witness_credential_helper", bind_helper)
    monkeypatch.setattr(
        commands,
        "_preflight_formal_git_credential_helper",
        lambda *_args, **_kwargs: {
            "host": "github.com",
            "fields": ("host", "password", "protocol", "username"),
        },
    )
    monkeypatch.setattr(
        commands,
        "_load_formal_corpus_manifest",
        lambda *_args: ({}, protocol.manifest_sha256),
    )
    trusted_head_checks = 0

    def validate_trusted_head(*_args, **_kwargs):
        nonlocal trusted_head_checks
        trusted_head_checks += 1
        return {"module.py": "1" * 64}

    monkeypatch.setattr(
        commands,
        "_validate_formal_head_checkout",
        validate_trusted_head,
    )
    monkeypatch.setattr(
        commands,
        "_source_tree_digest",
        lambda _files: protocol.source_tree_sha256,
    )
    monkeypatch.setattr(
        commands,
        "_codex_protocol",
        lambda **_kwargs: (
            protocol.codex_cli_version,
            protocol.codex_cli_executable_sha256,
            protocol.codex_client,
        ),
    )

    result = commands.register_formal_attempt(
        repo_root=root,
        config=Config(),
        model=protocol.model,
        reasoning_effort=protocol.reasoning_effort,
        docker_image_id=image_id,
        witness_remote_name="formal-witness",
        attempt_id_factory=lambda: "1" * 64,
    )

    updated = (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes()
    registry = parse_formal_ablation_attempt_registry(updated)
    registration = registry.open_registration()
    assert registration is not None
    assert registration.attempt_id == "1" * 64
    assert registration.docker_image_id == image_id
    assert registration.witness_remote_url.startswith("https://")
    assert result["registration_registry_sha256"] == hashlib.sha256(updated).hexdigest()
    assert trusted_head_checks == 2
    assert helper_checks == 2


def test_append_event_rejects_a_stale_registry(tmp_path: Path):
    root = _repository(tmp_path)
    registration = _registration()
    current = (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes()

    with pytest.raises(commands.FormalAttemptCommandError, match="changed"):
        commands._append_event(
            root,
            expected=current + b"stale",
            event=registration,
        )

    assert parse_formal_ablation_attempt_registry(current).events == ()
    assert (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes() == current


def test_append_event_recognizes_an_installed_update_after_io_error(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    current = (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes()

    def installed_then_failed(path, update, *, anchor, mode=None):
        del anchor, mode
        Path(path).write_bytes(update(current))
        raise OSError("directory fsync result was lost")

    monkeypatch.setattr(
        commands,
        "anchored_update_bytes",
        installed_then_failed,
    )

    registry, updated = commands._append_event(
        root,
        expected=current,
        event=registration,
    )

    assert registry.events == (registration,)
    assert (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes() == updated


def test_missing_local_evidence_after_remote_witness_is_not_reported_as_zero(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    monkeypatch.setattr(
        commands,
        "_anonymous_git_output",
        lambda *_args, **_kwargs: (
            "a" * 40 + "\t" + registration.witness_ref + "\n"
        ),
    )

    assert commands._partial_cell_counts(root, registration) == (
        "evidence_missing",
        None,
        None,
    )


def test_no_witness_and_no_output_is_known_zero(tmp_path: Path, monkeypatch):
    root = _repository(tmp_path)
    registration = _registration()
    monkeypatch.setattr(
        commands,
        "_anonymous_git_output",
        lambda *_args, **_kwargs: "",
    )

    assert commands._partial_cell_counts(root, registration) == ("known", 0, 0)


def test_durable_header_without_witness_is_known_zero(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    registry_sha256 = hashlib.sha256(
        (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes()
    ).hexdigest()
    output = root / registration.output_path
    output.mkdir(parents=True)
    (output / ".formal-ablation.lock").write_bytes(b"")
    header = {
        "schema_version": 1,
        "formal_attempt_id": registration.attempt_id,
        "registration_registry_sha256": registry_sha256,
        "protocol_sha256": registration.protocol_sha256,
        "outcome_key": "1" * 64,
    }
    (output / "formal_run.json").write_bytes(
        commands._canonical_json_object_bytes(header)
    )
    monkeypatch.setattr(
        commands,
        "_anonymous_git_output",
        lambda *_args, **_kwargs: "",
    )

    assert commands._partial_cell_counts(root, registration) == ("known", 0, 0)


def test_partial_header_without_witness_is_known_zero(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    output = root / registration.output_path
    output.mkdir(parents=True)
    (output / ".formal-ablation.lock").write_bytes(b"")
    (output / "formal_run.json").write_bytes(b'{"schema_version":')
    monkeypatch.setattr(
        commands,
        "_anonymous_git_output",
        lambda *_args, **_kwargs: "",
    )

    assert commands._partial_cell_counts(root, registration) == ("known", 0, 0)


def test_damaged_header_with_remote_witness_records_missing_evidence(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    output = root / registration.output_path
    output.mkdir(parents=True)
    (output / ".formal-ablation.lock").write_bytes(b"")
    (output / "formal_run.json").write_bytes(b'{"schema_version":')
    monkeypatch.setattr(
        commands,
        "_anonymous_git_output",
        lambda *_args, **_kwargs: (
            "a" * 40 + "\t" + registration.witness_ref + "\n"
        ),
    )

    assert commands._partial_cell_counts(root, registration) == (
        "evidence_missing",
        None,
        None,
    )


def test_missing_header_with_local_evidence_is_recorded_as_missing(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    evidence = root / registration.output_path / "results"
    evidence.mkdir(parents=True)
    (evidence / "case__r0.started.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        commands,
        "_anonymous_git_output",
        lambda *_args, **_kwargs: "",
    )

    assert commands._partial_cell_counts(root, registration) == (
        "evidence_missing",
        None,
        None,
    )


def test_abandon_refuses_a_complete_validated_result(tmp_path: Path, monkeypatch):
    root = _repository(tmp_path)
    registration = _registration()
    registry_bytes = _write_registry(root, registration)
    report = root / registration.output_path / "ablation_report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        commands,
        "_open_registration_at_clean_head",
        lambda _root: ("b" * 40, registry_bytes, registration),
    )
    monkeypatch.setattr(
        release_claims,
        "validate_formal_ablation_output",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(commands.FormalAttemptCommandError, match="use.*complete"):
        commands.abandon_formal_attempt(
            repo_root=root,
            reason_code="operator_stopped",
            reason="不接受这次结果",
        )

    assert (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes() == (
        registry_bytes
    )


def test_abandon_records_unknown_progress_instead_of_fabricated_counts(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    registry_bytes = _write_registry(root, registration)
    monkeypatch.setattr(
        commands,
        "_open_registration_at_clean_head",
        lambda _root: ("b" * 40, registry_bytes, registration),
    )
    monkeypatch.setattr(
        commands,
        "_partial_cell_counts",
        lambda *_args: ("evidence_missing", None, None),
    )
    monkeypatch.setattr(
        commands,
        "_recover_formal_operations",
        lambda *_args, **_kwargs: 0,
    )

    result = commands.abandon_formal_attempt(
        repo_root=root,
        reason_code="local_evidence_lost",
        reason="远端启动见证存在，但本地证据目录不可用",
    )

    registry = parse_formal_ablation_attempt_registry(
        (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes()
    )
    terminal = registry.events[-1]
    assert isinstance(terminal, AbandonedAttempt)
    assert terminal.progress_status == "evidence_missing"
    assert terminal.started_cells is None
    assert terminal.terminal_cells is None
    assert result["progress_status"] == "evidence_missing"


def test_abandon_accepts_derived_markdown_without_json_commit_marker(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    registry_bytes = _write_registry(root, registration)
    output = root / registration.output_path
    output.mkdir(parents=True)
    (output / "ablation_report.md").write_text(
        "# incomplete formal report\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        commands,
        "_open_registration_at_clean_head",
        lambda _root: ("b" * 40, registry_bytes, registration),
    )
    monkeypatch.setattr(
        commands,
        "_partial_cell_counts",
        lambda *_args: ("known", 204, 204),
    )
    monkeypatch.setattr(
        commands,
        "_recover_formal_operations",
        lambda *_args, **_kwargs: 0,
    )

    result = commands.abandon_formal_attempt(
        repo_root=root,
        reason_code="report_commit_interrupted",
        reason="Markdown 已写入，但 JSON 完成标志未落盘",
    )

    assert result["state"] == "ABANDONED"
    assert not (output / "ablation_report.json").exists()


def test_complete_uses_validated_report_even_when_it_contains_error_cells(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    registry_bytes = _write_registry(root, registration)
    output = root / registration.output_path
    output.mkdir(parents=True)
    fingerprint = "9" * 64
    report_bytes = json.dumps(
        {
            "fingerprint": fingerprint,
            "provenance": {
                "formal_attempt_id": registration.attempt_id,
                "formal_attempt_protocol_sha256": registration.protocol_sha256,
                "formal_attempt_registry_sha256": hashlib.sha256(
                    registry_bytes
                ).hexdigest(),
            },
            "records": [{"status": "ERROR"}],
        }
    ).encode()
    (output / "ablation_report.json").write_bytes(report_bytes)
    monkeypatch.setattr(
        commands,
        "_open_registration_at_clean_head",
        lambda _root: ("b" * 40, registry_bytes, registration),
    )
    completion = CompletedAttempt(
        attempt_id=registration.attempt_id,
        protocol_sha256=registration.protocol_sha256,
        registration_registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        recorded_at=_TIME,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        report_fingerprint=fingerprint,
    )
    registry_after = formal_ablation_attempt_registry_bytes(
        FormalAblationAttemptRegistry(events=(registration, completion))
    )
    publication = FormalPublishSummary(
        state="INSTALLED",
        attempt_id=registration.attempt_id,
        completion=completion,
        registry_before_bytes=registry_bytes,
        registry_after_bytes=registry_after,
        registry_before_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        registry_after_sha256=hashlib.sha256(registry_after).hexdigest(),
        report_sha256=completion.report_sha256,
        report_fingerprint=fingerprint,
        evidence_files=10,
        evidence_bytes=100,
        horizon_files=3,
        horizon_bytes=30,
        registry_already_appended=False,
    )
    import lha.formal_publish as publishing

    monkeypatch.setattr(
        publishing,
        "install_formal_publication",
        lambda **_kwargs: publication,
    )
    monkeypatch.setattr(
        publishing,
        "verify_installed_publication",
        lambda **_kwargs: publication,
    )
    monkeypatch.setattr(
        publishing,
        "finalize_formal_publication",
        lambda **_kwargs: FormalPublishFinalizeSummary(
            attempt_id=registration.attempt_id,
            action="COMMITTED_AND_CLEANED",
        ),
    )
    monkeypatch.setattr(
        commands,
        "_validate_registration_checkout",
        lambda *_args, **_kwargs: None,
    )

    result = commands._complete_validated_formal_attempt(
        repository=root,
        output=output,
        registration=registration,
        registration_head="b" * 40,
        registry_bytes=registry_bytes,
        raw=json.loads(report_bytes),
        recovered_operations=0,
    )

    assert result["state"] == "COMPLETED"
    assert result["scheduled_cells"] == 204
    registry = parse_formal_ablation_attempt_registry(
        (root / "benchmarks" / "formal_ablation_attempts.json").read_bytes()
    )
    assert registry.events[-1].event == "COMPLETED"


def test_complete_cleans_an_appended_publication_without_reopening_output(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    before = _write_registry(root, registration)
    completion = CompletedAttempt(
        attempt_id=registration.attempt_id,
        protocol_sha256=registration.protocol_sha256,
        registration_registry_sha256=hashlib.sha256(before).hexdigest(),
        recorded_at=_TIME,
        report_sha256="8" * 64,
        report_fingerprint="9" * 64,
    )
    after = formal_ablation_attempt_registry_bytes(
        FormalAblationAttemptRegistry(events=(registration, completion))
    )
    (root / "benchmarks" / "formal_ablation_attempts.json").write_bytes(after)
    monkeypatch.setattr(
        commands,
        "_publication_inspection",
        lambda _repository: FormalPublishInspection(
            status="RECOVERY_REQUIRED",
            attempt_id=registration.attempt_id,
            transaction_state="INSTALLED",
        ),
    )
    monkeypatch.setattr(
        commands,
        "_open_registration_at_head_during_publication",
        lambda _repository: ("1" * 40, before, registration),
    )
    monkeypatch.setattr(
        commands,
        "_validate_registration_checkout",
        lambda *_args, **_kwargs: None,
    )
    expected = {
        "state": "COMPLETED",
        "publication_recovery": "COMMITTED_AND_CLEANED",
    }
    monkeypatch.setattr(
        commands,
        "_cleanup_appended_formal_publication",
        lambda **_kwargs: expected,
    )
    monkeypatch.setattr(
        commands,
        "_formal_ablation_lock",
        lambda *_args, **_kwargs: pytest.fail("output lock was reopened"),
    )

    assert commands._complete_formal_attempt_locked(repo_root=root) is expected


def test_complete_resumes_an_installed_publication_without_reopening_output(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    before = _write_registry(root, registration)
    (root / "benchmarks" / "ablation_report.json").write_text(
        '{"fingerprint":"' + "9" * 64 + '"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        commands,
        "_publication_inspection",
        lambda _repository: FormalPublishInspection(
            status="RECOVERY_REQUIRED",
            attempt_id=registration.attempt_id,
            transaction_state="INSTALLED",
        ),
    )
    monkeypatch.setattr(
        commands,
        "_open_registration_at_head_during_publication",
        lambda _repository: ("1" * 40, before, registration),
    )
    monkeypatch.setattr(
        commands,
        "_validate_registration_checkout",
        lambda *_args, **_kwargs: None,
    )
    expected = {"state": "COMPLETED"}
    monkeypatch.setattr(
        commands,
        "_complete_validated_formal_attempt",
        lambda **_kwargs: expected,
    )
    monkeypatch.setattr(
        commands,
        "_formal_ablation_lock",
        lambda *_args, **_kwargs: pytest.fail("output lock was reopened"),
    )

    assert commands._complete_formal_attempt_locked(repo_root=root) is expected


def test_repository_formal_lock_rejects_a_second_owner(tmp_path: Path):
    root = _repository(tmp_path)

    with formal_attempt_lock(root):
        with pytest.raises(RuntimeError, match="another formal attempt"):
            with formal_attempt_lock(root):
                pass


def test_repository_formal_lock_rejects_name_replacement_after_flock(
    tmp_path: Path,
    monkeypatch,
):
    import lha.ablation_attempts as attempts

    root = _repository(tmp_path)
    real_flock = attempts.fcntl.flock
    replaced = False

    def replacing_flock(descriptor, operation):
        nonlocal replaced
        real_flock(descriptor, operation)
        if operation & attempts.fcntl.LOCK_EX and not replaced:
            replaced = True
            lock = root / ".git" / attempts.FORMAL_ATTEMPT_LOCK_NAME
            lock.unlink()
            lock.write_bytes(b"")
            lock.chmod(0o600)

    monkeypatch.setattr(attempts.fcntl, "flock", replacing_flock)

    with pytest.raises(OSError, match="name changed"):
        with formal_attempt_lock(root):
            pass


def test_partial_counts_require_current_canonical_cell_bindings(
    tmp_path: Path,
    monkeypatch,
):
    root = _repository(tmp_path)
    registration = _registration()
    registry_bytes = _write_registry(root, registration)
    output = root / registration.output_path
    results = output / "results"
    output.mkdir(parents=True)
    outcome = "7" * 64
    header = commands._canonical_json_object_bytes(
        {
            "schema_version": commands._FORMAL_RUN_HEADER_SCHEMA,
            "formal_attempt_id": registration.attempt_id,
            "registration_registry_sha256": hashlib.sha256(
                registry_bytes
            ).hexdigest(),
            "protocol_sha256": registration.protocol_sha256,
            "outcome_key": outcome,
        }
    )
    (output / "formal_run.json").write_bytes(header)
    head = "1" * 40
    tree = "2" * 40
    message = formal_ablation_witness_message(
        attempt_id=registration.attempt_id,
        registration_registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        protocol_sha256=registration.protocol_sha256,
        outcome_key=outcome,
        run_header_sha256=hashlib.sha256(header).hexdigest(),
    )
    witness = formal_ablation_witness_commit_oid(
        formal_ablation_witness_commit_bytes(
            tree=tree,
            parent=head,
            message=message,
        )
    )
    monkeypatch.setattr(
        commands,
        "_anonymous_git_output",
        lambda *_args, **_kwargs: (
            f"{witness}\t{registration.witness_ref}\n"
        ),
    )
    monkeypatch.setattr(
        commands,
        "_git_output",
        lambda _root, args, **_kwargs: tree if "^{tree}" in args[-1] else head,
    )
    monkeypatch.setattr(
        commands,
        "_load_formal_corpus_manifest",
        lambda *_args: (
            {
                "tasks": [
                    {
                        "name": "case",
                        "task_sha256": "a" * 64,
                        "corpus_sha256": "b" * 64,
                    }
                ]
            },
            registration.manifest_sha256,
        ),
    )
    formal_fields = {
        "formal_attempt_id": registration.attempt_id,
        "formal_registration_registry_sha256": hashlib.sha256(
            registry_bytes
        ).hexdigest(),
        "formal_protocol_sha256": registration.protocol_sha256,
        "formal_outcome_key": outcome,
    }
    marker = {
        "schema_version": commands._CELL_ATTEMPT_SCHEMA,
        "task": "case",
        "rep": 0,
        "cell_fingerprint": "4" * 64,
        "input_snapshot_sha256": "5" * 64,
        **formal_fields,
    }
    assert commands._partial_cell_counts(root, registration) == (
        "evidence_missing",
        None,
        None,
    )
    results.mkdir()
    assert commands._partial_cell_counts(root, registration) == (
        "evidence_missing",
        None,
        None,
    )
    terminal = {
        "schema_version": ablation._CACHE_SCHEMA,
        "fingerprint": marker["cell_fingerprint"],
        "terminal_error": False,
        "records": [
            {"task": "case", "rep": 0, "condition": condition}
            for condition in ("trust", "gate", "verify")
        ],
        "llm_call_receipts": ["6" * 64],
        **formal_fields,
    }
    start_path = results / "case__r0.started.json"
    terminal_path = results / "case__r0.json"
    start_path.write_bytes(commands._canonical_json_object_bytes(marker))
    terminal_path.write_bytes(commands._canonical_json_object_bytes(terminal))
    monkeypatch.setattr(
        commands,
        "_formal_input_snapshot_is_valid",
        lambda *_args, **_kwargs: True,
    )

    def load_cached(path, fingerprint, **_kwargs):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("fingerprint") != fingerprint:
            return None
        return SimpleNamespace(
            records=[
                SimpleNamespace(condition=condition, task="case", rep=0)
                for condition in ("trust", "gate", "verify")
            ]
        )

    monkeypatch.setattr(commands, "_load_cached_cell", load_cached)

    assert commands._partial_cell_counts(root, registration) == ("known", 1, 1)

    monkeypatch.setattr(commands, "_load_cached_cell", lambda *_args, **_kwargs: None)
    assert commands._partial_cell_counts(root, registration) == (
        "evidence_missing",
        None,
        None,
    )
    monkeypatch.setattr(commands, "_load_cached_cell", load_cached)

    terminal["fingerprint"] = "8" * 64
    terminal_path.write_bytes(commands._canonical_json_object_bytes(terminal))
    assert commands._partial_cell_counts(root, registration) == (
        "evidence_missing",
        None,
        None,
    )
