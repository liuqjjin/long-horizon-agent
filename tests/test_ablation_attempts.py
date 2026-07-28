from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from lha.ablation_attempts import (
    AbandonedAttempt,
    CompletedAttempt,
    FormalAblationAttemptRegistry,
    FormalAblationProtocol,
    FormalCodexClientConfig,
    RegisteredAttempt,
    UnregisteredRunRecorded,
    formal_ablation_protocol_sha256,
    formal_ablation_selection_sha256,
    formal_codex_client_sha256,
    parse_formal_ablation_attempt_registry,
    registry_has_prefix,
)

_TIME = "2026-07-28T12:00:00+08:00"


def _protocol() -> FormalAblationProtocol:
    client = FormalCodexClientConfig(
        max_retries=2,
        timeout_s=300.0,
        retry_backoff_s=1.0,
    )
    return FormalAblationProtocol(
        source_commit="a" * 40,
        source_tree_sha256="f" * 64,
        manifest_sha256="b" * 64,
        model="gpt-5.6",
        reasoning_effort="xhigh",
        docker_image_id="sha256:" + "c" * 64,
        codex_cli_version="codex-cli 1.2.3",
        codex_cli_executable_sha256="9" * 64,
        codex_client=client,
        codex_client_sha256=formal_codex_client_sha256(client),
    )


def _registered(
    attempt_id: str,
    *,
    protocol: FormalAblationProtocol | None = None,
) -> RegisteredAttempt:
    selected = protocol or _protocol()
    return RegisteredAttempt(
        attempt_id=attempt_id,
        protocol_sha256=formal_ablation_protocol_sha256(selected),
        source_commit=selected.source_commit,
        source_tree_sha256=selected.source_tree_sha256,
        manifest_sha256=selected.manifest_sha256,
        output_path=f"runs/formal_ablation/{attempt_id}",
        model=selected.model,
        reasoning_effort=selected.reasoning_effort,
        docker_image_id=selected.docker_image_id,
        codex_cli_version=selected.codex_cli_version,
        codex_cli_executable_sha256=selected.codex_cli_executable_sha256,
        codex_client=selected.codex_client,
        codex_client_sha256=selected.codex_client_sha256,
        witness_remote_name="formal-witness",
        witness_remote_url="git@github.com:example/lha-formal-witness.git",
        registered_at=_TIME,
    )


def _completed(attempt_id: str) -> CompletedAttempt:
    return CompletedAttempt(
        attempt_id=attempt_id,
        protocol_sha256=formal_ablation_protocol_sha256(_protocol()),
        registration_registry_sha256="9" * 64,
        recorded_at=_TIME,
        report_sha256="d" * 64,
        report_fingerprint="e" * 64,
    )


def test_open_registration_blocks_a_new_attempt():
    first = _registered("1" * 64)
    second = _registered("2" * 64)

    with pytest.raises(ValidationError, match="must end before a new attempt"):
        FormalAblationAttemptRegistry(events=(first, second))


def test_abandoned_attempt_consumes_its_selection():
    first = _registered("1" * 64)
    second = _registered("2" * 64)

    with pytest.raises(ValidationError, match="consumed.*cannot be registered"):
        FormalAblationAttemptRegistry(
            events=(
                first,
                AbandonedAttempt(
                    attempt_id=first.attempt_id,
                    recorded_at=_TIME,
                    started_cells=28,
                    terminal_cells=28,
                    reason_code="operator_stopped",
                    reason="运行已经发生，不能选择性重试",
                ),
                second,
            )
        )


def test_abandoned_attempt_allows_a_materially_changed_selection():
    first = _registered("1" * 64)
    changed = _protocol().model_copy(
        update={
            "source_commit": "d" * 40,
            "source_tree_sha256": "e" * 64,
        }
    )
    second = _registered("2" * 64, protocol=changed)
    registry = FormalAblationAttemptRegistry(
        events=(
            first,
            AbandonedAttempt(
                attempt_id=first.attempt_id,
                recorded_at=_TIME,
                started_cells=28,
                terminal_cells=28,
                reason_code="source_changed",
                reason="实现代码已经修改",
            ),
            second,
        )
    )

    assert registry.open_registration() == second


def test_completed_protocol_cannot_be_registered_again():
    first = _registered("1" * 64)
    second = _registered("2" * 64)

    with pytest.raises(ValidationError, match="consumed.*cannot be registered"):
        FormalAblationAttemptRegistry(
            events=(
                first,
                _completed(first.attempt_id),
                second,
            )
        )


def test_terminal_attempt_cannot_change_state():
    registration = _registered("1" * 64)

    with pytest.raises(ValidationError, match="cannot change state"):
        FormalAblationAttemptRegistry(
            events=(
                registration,
                AbandonedAttempt(
                    attempt_id=registration.attempt_id,
                    recorded_at=_TIME,
                    started_cells=12,
                    terminal_cells=10,
                    reason_code="operator_stopped",
                    reason="运行中止",
                ),
                _completed(registration.attempt_id),
            )
        )


def test_unregistered_disclosure_never_opens_an_attempt():
    protocol = _protocol()
    disclosure = UnregisteredRunRecorded(
        attempt_id="1" * 64,
        protocol_sha256=formal_ablation_protocol_sha256(protocol),
        source_commit=protocol.source_commit,
        source_tree_sha256=protocol.source_tree_sha256,
        manifest_sha256=protocol.manifest_sha256,
        output_path="runs/old-formal-ablation",
        model=protocol.model,
        reasoning_effort=protocol.reasoning_effort,
        docker_image_id=protocol.docker_image_id,
        codex_cli_version=protocol.codex_cli_version,
        codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
        codex_client=protocol.codex_client,
        codex_client_sha256=protocol.codex_client_sha256,
        recorded_at=_TIME,
        reason="旧运行发生在登记规则加入之前，仅作披露",
        published_report_path=(
            "benchmarks/formal_ablation_history/"
            f"{protocol.source_commit}/ablation_report.json"
        ),
        report_sha256="3" * 64,
        report_fingerprint="4" * 64,
        scheduled_cells=204,
        usable_cells=204,
        error_cells=0,
        trust_delivered_correct=190,
        trust_delivered_wrong=14,
        gate_delivered_correct=190,
        gate_delivered_wrong=0,
        gate_intercepted_wrong=14,
        gate_rejected_correct=0,
        verify_delivered_correct=204,
        verify_delivered_wrong=0,
        verify_not_delivered=0,
    )
    registry = FormalAblationAttemptRegistry(events=(disclosure,))

    assert registry.open_registration() is None
    assert registry.registration(disclosure.attempt_id) is None


def test_unregistered_disclosure_consumes_the_same_selection():
    protocol = _protocol()
    disclosure = UnregisteredRunRecorded(
        attempt_id="1" * 64,
        protocol_sha256=formal_ablation_protocol_sha256(protocol),
        source_commit=protocol.source_commit,
        source_tree_sha256=protocol.source_tree_sha256,
        manifest_sha256=protocol.manifest_sha256,
        output_path="runs/old-formal-ablation",
        model=protocol.model,
        reasoning_effort=protocol.reasoning_effort,
        docker_image_id=protocol.docker_image_id,
        codex_cli_version=protocol.codex_cli_version,
        codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
        codex_client=protocol.codex_client,
        codex_client_sha256=protocol.codex_client_sha256,
        recorded_at=_TIME,
        reason="旧运行已经使用这一实验条件",
        published_report_path=(
            "benchmarks/formal_ablation_history/"
            f"{protocol.source_commit}/ablation_report.json"
        ),
        report_sha256="3" * 64,
        report_fingerprint="4" * 64,
        scheduled_cells=204,
        usable_cells=204,
        error_cells=0,
        trust_delivered_correct=190,
        trust_delivered_wrong=14,
        gate_delivered_correct=190,
        gate_delivered_wrong=0,
        gate_intercepted_wrong=14,
        gate_rejected_correct=0,
        verify_delivered_correct=204,
        verify_delivered_wrong=0,
        verify_not_delivered=0,
    )

    with pytest.raises(ValidationError, match="consumed.*cannot be registered"):
        FormalAblationAttemptRegistry(
            events=(disclosure, _registered("2" * 64, protocol=protocol))
        )


def test_empty_commit_cannot_change_a_consumed_selection():
    first_protocol = _protocol()
    second_protocol = first_protocol.model_copy(update={"source_commit": "d" * 40})
    first = _registered("1" * 64, protocol=first_protocol)
    second = _registered("2" * 64, protocol=second_protocol)

    with pytest.raises(ValidationError, match="consumed.*cannot be registered"):
        FormalAblationAttemptRegistry(
            events=(
                first,
                AbandonedAttempt(
                    attempt_id=first.attempt_id,
                    recorded_at=_TIME,
                    started_cells=0,
                    terminal_cells=0,
                    reason_code="preflight_failed",
                    reason="预检失败",
                ),
                second,
            )
        )


def test_protocol_digest_cannot_be_replaced():
    protocol = _protocol()

    with pytest.raises(ValidationError, match="protocol_sha256"):
        RegisteredAttempt(
            attempt_id="1" * 64,
            protocol_sha256="0" * 64,
            source_commit=protocol.source_commit,
            source_tree_sha256=protocol.source_tree_sha256,
            manifest_sha256=protocol.manifest_sha256,
            output_path="runs/formal_ablation/" + "1" * 64,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
            codex_client=protocol.codex_client,
            codex_client_sha256=protocol.codex_client_sha256,
            witness_remote_name="formal-witness",
            witness_remote_url="git@github.com:example/lha-formal-witness.git",
            registered_at=_TIME,
        )


def test_codex_client_digest_cannot_be_replaced():
    protocol = _protocol()

    with pytest.raises(ValidationError, match="codex_client_sha256"):
        FormalAblationProtocol(
            **protocol.model_dump(
                mode="python",
                exclude={"codex_client_sha256"},
            ),
            codex_client_sha256="0" * 64,
        )


def test_codex_retry_configuration_changes_the_selection():
    first = _protocol()
    changed_client = first.codex_client.model_copy(
        update={"max_retries": first.codex_client.max_retries + 1}
    )
    second = first.model_copy(
        update={
            "codex_client": changed_client,
            "codex_client_sha256": formal_codex_client_sha256(
                changed_client
            ),
        }
    )

    assert formal_ablation_protocol_sha256(first) != formal_ablation_protocol_sha256(
        second
    )
    assert formal_ablation_selection_sha256(first) != formal_ablation_selection_sha256(
        second
    )


def test_registered_output_is_derived_from_attempt_id():
    protocol = _protocol()

    with pytest.raises(ValidationError, match="output_path"):
        RegisteredAttempt(
            attempt_id="1" * 64,
            protocol_sha256=formal_ablation_protocol_sha256(protocol),
            source_commit=protocol.source_commit,
            source_tree_sha256=protocol.source_tree_sha256,
            manifest_sha256=protocol.manifest_sha256,
            output_path="runs/formal_ablation/another-directory",
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
            codex_client=protocol.codex_client,
            codex_client_sha256=protocol.codex_client_sha256,
            witness_remote_name="formal-witness",
            witness_remote_url="git@github.com:example/lha-formal-witness.git",
            registered_at=_TIME,
        )


def test_unregistered_disclosure_requires_complete_condition_counts():
    protocol = _protocol()
    fields = {
        "attempt_id": "1" * 64,
        "protocol_sha256": formal_ablation_protocol_sha256(protocol),
        "source_commit": protocol.source_commit,
        "source_tree_sha256": protocol.source_tree_sha256,
        "manifest_sha256": protocol.manifest_sha256,
        "output_path": "runs/old-formal-ablation",
        "model": protocol.model,
        "reasoning_effort": protocol.reasoning_effort,
        "docker_image_id": protocol.docker_image_id,
        "codex_cli_version": protocol.codex_cli_version,
        "codex_cli_executable_sha256": protocol.codex_cli_executable_sha256,
        "codex_client": protocol.codex_client,
        "codex_client_sha256": protocol.codex_client_sha256,
        "recorded_at": _TIME,
        "reason": "旧运行只作披露",
        "published_report_path": (
            "benchmarks/formal_ablation_history/"
            f"{protocol.source_commit}/ablation_report.json"
        ),
        "report_sha256": "3" * 64,
        "report_fingerprint": "4" * 64,
        "scheduled_cells": 204,
        "usable_cells": 204,
        "error_cells": 0,
        "trust_delivered_correct": 190,
        "trust_delivered_wrong": 14,
        "gate_delivered_correct": 190,
        "gate_delivered_wrong": 0,
        "gate_intercepted_wrong": 13,
        "gate_rejected_correct": 0,
        "verify_delivered_correct": 204,
        "verify_delivered_wrong": 0,
        "verify_not_delivered": 0,
    }

    with pytest.raises(ValidationError, match="gate or verify counts"):
        UnregisteredRunRecorded(**fields)


def test_registry_rejects_unknown_fields():
    registration = _registered("1" * 64)
    raw = {
        "schema_version": 1,
        "events": [
            {
                **registration.model_dump(mode="json"),
                "authorization_override": True,
            }
        ],
    }

    with pytest.raises(ValueError, match="registry is invalid"):
        parse_formal_ablation_attempt_registry(
            json.dumps(raw).encode("utf-8")
        )


def test_current_registry_must_retain_historical_prefix():
    registration = _registered("1" * 64)
    historical = FormalAblationAttemptRegistry(events=(registration,))
    current = FormalAblationAttemptRegistry(
        events=(registration, _completed(registration.attempt_id))
    )
    rewritten = FormalAblationAttemptRegistry(
        events=(
            _registered("2" * 64),
        )
    )

    assert registry_has_prefix(current, historical)
    assert not registry_has_prefix(rewritten, historical)
