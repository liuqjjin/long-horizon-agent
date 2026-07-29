"""Benchmark adapter contracts; no test here calls a model."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import signal
import stat
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import pytest

import lha
from lha.bench import (
    Prediction,
    cluster_bootstrap_ci,
    eval_command,
    mcnemar_exact,
    paired_cluster_sign_flip_exact,
    parse_report,
    write_predictions,
)
from lha.bench import terminal_bench as tb
from lha.bench import terminal_public_evidence as tpe
from lha.bench.swebench import DATASET, prediction_from_run


# --- SWE-bench predictions ---------------------------------------------------
def test_predictions_jsonl_uses_official_fields(tmp_path):
    preds = [
        Prediction("astropy__astropy-1", "diff --git a b\n", "lha"),
        Prediction("django__django-2", "", "lha"),
    ]
    path = write_predictions(preds, tmp_path / "preds.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert set(rows[0]) == {"instance_id", "model_name_or_path", "model_patch"}
    assert rows[0]["instance_id"] == "astropy__astropy-1"
    assert rows[1]["model_patch"] == ""


def test_duplicate_instance_id_refused(tmp_path):
    preds = [Prediction("x-1", "a", "lha"), Prediction("x-1", "b", "lha")]
    with pytest.raises(ValueError, match="duplicate"):
        write_predictions(preds, tmp_path / "preds.jsonl")


def test_prediction_from_run_reads_the_frozen_patch(tmp_path):
    (tmp_path / "patch.diff").write_text("diff --git a/f.py b/f.py\n")
    p = prediction_from_run(tmp_path, "repo__repo-1", "lha")
    assert p.model_patch.startswith("diff --git")

    empty = tmp_path / "empty_run"
    empty.mkdir()
    assert prediction_from_run(empty, "repo__repo-2", "lha").model_patch == ""

    placeholder = tmp_path / "placeholder_run"
    placeholder.mkdir()
    (placeholder / "patch.diff").write_text("(no diff)\n")
    assert prediction_from_run(placeholder, "repo__repo-3", "lha").model_patch == ""


def test_eval_command_is_the_official_invocation(tmp_path):
    cmd = eval_command(tmp_path / "p.jsonl", "run1", max_workers=4)
    joined = " ".join(cmd)
    assert "-m swebench.harness.run_evaluation" in joined
    assert cmd[cmd.index("--dataset_name") + 1] == DATASET
    assert cmd[cmd.index("--run_id") + 1] == "run1"
    assert cmd[cmd.index("--max_workers") + 1] == "4"
    assert "--namespace" not in cmd  # default: upstream images

    local = eval_command(tmp_path / "p.jsonl", "run1", namespace="")
    assert local[local.index("--namespace") + 1] == ""  # arm64: build locally


def test_parse_report_keeps_errors_in_the_denominator(tmp_path):
    # The shape the official harness writes (schema_version 2): one resolved,
    # one unresolved, one eval error, one empty patch.
    report = {
        "total_instances": 500,
        "submitted_instances": 4,
        "completed_instances": 2,
        "resolved_instances": 1,
        "unresolved_instances": 1,
        "empty_patch_instances": 1,
        "error_instances": 1,
        "completed_ids": ["gold-1", "wrong-2"],
        "incomplete_ids": [],
        "empty_patch_ids": ["noop-3"],
        "submitted_ids": ["gold-1", "wrong-2", "noop-3", "crash-4"],
        "resolved_ids": ["gold-1"],
        "unresolved_ids": ["wrong-2"],
        "error_ids": ["crash-4"],
        "schema_version": 2,
    }
    path = tmp_path / "lha.run1.json"
    path.write_text(json.dumps(report))

    s = parse_report(path)
    assert (s.resolved, s.unresolved, s.empty_patch, s.error) == (1, 1, 1, 1)
    assert s.resolved_rate == pytest.approx(0.25)  # 1/4 — the error is NOT dropped
    assert s.error_rate == pytest.approx(0.25)
    assert s.resolved_ids == ["gold-1"] and s.error_ids == ["crash-4"]
    assert "1/4 resolved" in s.to_markdown()


# --- paired statistics -------------------------------------------------------
def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)  # symmetric
    assert mcnemar_exact(5, 5) == 1.0  # capped two-sided
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_cluster_bootstrap_ci_degenerate_and_deterministic():
    same = {"t1": [1.0, 1.0], "t2": [1.0], "t3": [1.0, 1.0, 1.0]}
    assert cluster_bootstrap_ci(same, n=200) == (1.0, 1.0)

    mixed = {"t1": [1.0, 0.0], "t2": [0.0], "t3": [1.0]}
    a = cluster_bootstrap_ci(mixed, n=500, seed=7)
    b = cluster_bootstrap_ci(mixed, n=500, seed=7)
    assert a == b
    assert a is not None and 0.0 <= a[0] <= a[1] <= 1.0
    assert cluster_bootstrap_ci(dict(reversed(list(mixed.items()))), n=500, seed=7) == a

    assert cluster_bootstrap_ci({}) is None
    assert cluster_bootstrap_ci({"t": []}) is None


def test_cluster_sign_flip_does_not_treat_repetitions_as_independent():
    result = paired_cluster_sign_flip_exact(
        {
            "repeated_failure": [(False, True)] * 12,
            "unchanged": [(True, True)] * 12,
        }
    )

    assert result.clusters == 2
    assert result.nonzero_clusters == 1
    assert result.mean_difference == pytest.approx(0.5)
    assert result.p_value == 1.0
    assert mcnemar_exact(12, 0) == pytest.approx(0.00048828125)


def test_cluster_sign_flip_uses_task_means_and_is_two_sided():
    result = paired_cluster_sign_flip_exact(
        {
            "large_effect": [(False, True)] * 7 + [(True, True)] * 5,
            "small_effect": [(False, True)] * 3 + [(True, True)] * 9,
        }
    )

    assert result.clusters == 2
    assert result.nonzero_clusters == 2
    assert result.mean_difference == pytest.approx(5 / 12)
    assert result.p_value == 0.5


@pytest.mark.parametrize(
    "pairs",
    [
        {},
        {"": [(False, True)]},
        {"task": []},
        {"task": [(0, True)]},
    ],
)
def test_cluster_sign_flip_rejects_invalid_clusters(pairs):
    with pytest.raises(ValueError):
        paired_cluster_sign_flip_exact(pairs)


# --- Terminal-Bench agent (Harbor stubbed; no model calls) -------------------
def _codex_stream(*items, usage=None):
    rows = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        *items,
        {
            "type": "item.completed",
            "item": {"id": "final", "type": "agent_message", "text": "done"},
        },
        {
            "type": "turn.completed",
            "usage": usage
            or {
                "input_tokens": 10,
                "cached_input_tokens": 2,
                "output_tokens": 3,
                "reasoning_output_tokens": 1,
            },
        },
    ]
    return "\n".join(json.dumps(row) for row in rows)


def test_codex_exec_is_tool_enabled_and_uses_harbor_as_outer_sandbox():
    headers = {
        "X-LHA-Evaluation-ID": "a" * 32,
        "X-LHA-Attempt-ID": "b" * 64,
        "X-LHA-Container-ID": "c" * 64,
    }
    cmd = tb.codex_exec_command(
        "gpt-5.5",
        "xhigh",
        "configure the service",
        proxy_base_url="https://lha-terminal-proxy:8080",
        binding_headers=headers,
    )
    assert "CODEX_HOME=/tmp/lha_codex_runtime" in cmd
    assert "/usr/local/bin/codex exec" in cmd
    assert "exec env LHA_TERMINAL_PROXY_CAPABILITY=" in cmd
    assert "--sandbox danger-full-access" in cmd
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert "--json" in cmd
    assert "env_key" in cmd
    assert "LHA_TERMINAL_PROXY_CAPABILITY" in cmd
    assert "CODEX_CA_CERTIFICATE=/proc/self/fd/3" in cmd
    assert "SSL_CERT_FILE=/proc/self/fd/3" in cmd
    assert "exec 3</tmp/.lha_terminal_proxy_ca.pem" in cmd
    assert "request_max_retries = 1" in cmd
    assert "stream_max_retries = 0" in cmd
    assert "OpenAI" in cmd
    assert "LHA Terminal broker" not in cmd
    assert "version" in cmd
    assert "0.141.0" in cmd
    assert "Authorization" not in cmd
    assert "access_token" not in cmd
    assert "account_id" not in cmd
    assert "--disable multi_agent" in cmd
    assert "--disable multi_agent_v2" in cmd
    assert "--disable collab" not in cmd
    assert "configure the service" in cmd

    with pytest.raises(ValueError, match="binding headers"):
        tb.codex_exec_command(
            "gpt-5.5",
            "xhigh",
            "configure the service",
            binding_headers={"X-LHA-Evaluation-ID": "a" * 32},
        )
    with pytest.raises(ValueError, match="no client stream retry"):
        tb.codex_exec_command(
            "gpt-5.5",
            "xhigh",
            "configure the service",
            binding_headers=headers,
            stream_max_retries=1,
        )


def test_codex_stream_allows_completed_tools_and_rejects_protocol_damage():
    stream = _codex_stream(
        {
            "type": "item.started",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "command": "make",
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "command": "make",
                "aggregated_output": "ok\n",
                "exit_code": 0,
                "status": "completed",
            },
        },
    )
    audit = tb.audit_codex_jsonl(stream)
    assert audit.tool_calls == 1
    assert (audit.input_tokens, audit.cached_input_tokens, audit.output_tokens) == (10, 2, 3)
    assert audit.reasoning_output_tokens == 1

    # A real Codex 0.141 run emits file_change as a paired tool lifecycle.
    file_change = tb.audit_codex_jsonl(
        _codex_stream(
            {
                "type": "item.started",
                "item": {
                    "id": "patch-1",
                    "type": "file_change",
                    "status": "in_progress",
                    "changes": [{"path": "answer.txt", "kind": "add"}],
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "patch-1",
                    "type": "file_change",
                    "status": "completed",
                    "changes": [{"path": "answer.txt", "kind": "add"}],
                },
            }
        )
    )
    assert file_change.tool_calls == 1

    with pytest.raises(RuntimeError, match="invalid JSON"):
        tb.audit_codex_jsonl(stream + "\n{broken")
    with pytest.raises(RuntimeError, match="does not match 0.141"):
        tb.audit_codex_jsonl(
            "\n".join(
                (
                    json.dumps({"type": "thread.started"}),
                    json.dumps({"type": "turn.started"}),
                    json.dumps({"type": "future.event"}),
                )
            )
        )
    unfinished = "\n".join(
        (
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "id": "tool-1",
                        "type": "command_execution",
                        "command": "make",
                        "aggregated_output": "",
                        "exit_code": None,
                        "status": "in_progress",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_output_tokens": 0,
                    },
                }
            ),
        )
    )
    with pytest.raises(RuntimeError, match="unfinished items"):
        tb.audit_codex_jsonl(unfinished)
    with pytest.raises(RuntimeError, match="does not match 0.141"):
        tb.audit_codex_jsonl(
            _codex_stream(
                {
                    "type": "item.completed",
                    "item": {"id": "future-1", "type": "future_tool"},
                }
            )
        )


def test_terminal_bench_21_subset_is_deterministic_and_disjoint():
    ids = [f"task-{index:02d}" for index in range(30)]
    subset = tb.preregister_instances(ids)
    expected = sorted(ids, key=lambda item: (hashlib.sha256(item.encode()).hexdigest(), item))
    assert tb.DATASET == "terminal-bench/terminal-bench-2-1"
    assert list(subset.scored_instance_ids) == expected[:20]
    assert list(subset.smoke_instance_ids) == expected[20:23]
    assert set(subset.scored_instance_ids).isdisjoint(subset.smoke_instance_ids)

    with pytest.raises(ValueError, match="at least 23"):
        tb.preregister_instances(ids[:22])
    with pytest.raises(ValueError, match="unique"):
        tb.preregister_instances(ids + [ids[0]])

    duplicate_scored = subset.model_dump(mode="json")
    duplicate_scored["scored_instance_ids"][-1] = duplicate_scored[
        "scored_instance_ids"
    ][0]
    with pytest.raises(ValueError, match="scored instance ids must be unique"):
        tb.RegisteredSubset.model_validate(duplicate_scored)

    overlapping = subset.model_dump(mode="json")
    overlapping["smoke_instance_ids"][0] = overlapping["scored_instance_ids"][0]
    with pytest.raises(ValueError, match="must be disjoint"):
        tb.RegisteredSubset.model_validate(overlapping)


def _image_map(instance_ids, character="a"):
    return {item: "sha256:" + character * 64 for item in instance_ids}


def _content_map(instance_ids, character="b"):
    return {item: "sha256:" + character * 64 for item in instance_ids}


def _checksum_map(instance_ids, character="e"):
    return {item: character * 64 for item in instance_ids}


def test_protocol_records_exact_provenance_and_contains_no_secret(tmp_path):
    wheel = tmp_path / "lha.whl"
    wheel.write_bytes(b"wheel bytes")
    codex_binary = tmp_path / "codex"
    codex_binary.write_bytes(b"codex linux binary")
    protocol = tb.create_protocol(
        evaluation_id="1" * 32,
        output_root=tmp_path / "jobs",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        codex_cli_version="codex-cli 0.141.0",
        codex_target="x86_64-unknown-linux-musl",
        codex_binary_path=codex_binary,
        broker_image_id="sha256:" + "f" * 64,
        wheel_path=wheel,
    )
    assert protocol.schema_version == 14
    assert protocol.budgets.codex_timeout_s == 1800
    assert protocol.budgets.max_tool_calls == 128
    assert protocol.budgets.max_model_requests == 60
    assert protocol.budgets.stream_max_retries == 0
    assert protocol.budgets.broker_stream_max_retries == 12
    assert protocol.budgets.broker_stream_max_retries_per_request == 4
    assert protocol.budgets.max_jsonl_line_bytes == 2 * 1024 * 1024
    assert protocol.budgets.codex_exec_runs == 1
    assert protocol.budgets.scored_runs_per_task == 1
    assert protocol.budgets.infrastructure_retries == 0
    assert protocol.budgets.task_retries == 0
    assert protocol.budgets.harbor_agent_timeout_multiplier == 4
    assert len(protocol.task_content_digests) == 23
    assert len(protocol.task_checksums) == 23
    assert len(protocol.task_image_digests) == 23
    assert len(protocol.corpus_instance_ids) == 89
    assert protocol.evaluation_id == "1" * 32
    assert protocol.output_root == str((tmp_path / "jobs").resolve())
    assert protocol.broker_image_id == "sha256:" + "f" * 64
    assert protocol.codex_binary_sha256 == hashlib.sha256(
        b"codex linux binary"
    ).hexdigest()
    assert protocol.wheel_sha256 == hashlib.sha256(b"wheel bytes").hexdigest()

    legacy = protocol.model_dump(mode="json")
    legacy["schema_version"] = 13
    legacy["budgets"]["max_jsonl_line_bytes"] = 60 * 1024
    assert tb.TerminalBenchProtocol.model_validate(legacy).schema_version == 13
    legacy["budgets"]["max_jsonl_line_bytes"] = 2 * 1024 * 1024
    with pytest.raises(ValueError, match="schema does not match"):
        tb.TerminalBenchProtocol.model_validate(legacy)

    path = tb.write_protocol(protocol, tmp_path / "protocol.json")
    raw = path.read_text()
    assert "auth" not in raw.lower()
    assert tb.TerminalBenchProtocol.model_validate_json(raw) == protocol


def test_protocol_is_derived_from_the_complete_packaged_official_corpus(
    monkeypatch,
    tmp_path,
):
    corpus = tb.load_terminal_bench_corpus()
    assert len(corpus.tasks) == 89
    assert corpus.resolution_failures == ()
    subset = tb.preregister_instances(tuple(corpus.tasks))
    assert subset.scored_instance_ids[0] == "terminal-bench/regex-log"
    assert subset.scored_instance_ids[-1] == (
        "terminal-bench/model-extraction-relu-logits"
    )
    assert subset.smoke_instance_ids == (
        "terminal-bench/sam-cell-seg",
        "terminal-bench/sqlite-with-gcov",
        "terminal-bench/password-recovery",
    )

    original_resource = tb._CORPUS_RESOURCE
    tampered = tmp_path / "corpus.json"
    tampered.write_bytes(original_resource.read_bytes() + b" ")
    monkeypatch.setattr(tb, "_CORPUS_RESOURCE", tampered)
    with pytest.raises(RuntimeError, match="digest changed"):
        tb.load_terminal_bench_corpus()


def test_public_corpus_snapshot_is_the_source_of_packaged_metadata():
    repository = Path(__file__).resolve().parents[1]
    snapshot_path = repository / "benchmarks" / "terminal_bench_2_1_resolution.json"
    snapshot_payload = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_payload)
    resource = json.loads(tb._CORPUS_RESOURCE.read_bytes())
    assert len(snapshot["tasks"]) == 89
    assert snapshot["resolution_failures"] == []
    assert resource == {
        **snapshot,
        "source_inputs_sha256": hashlib.sha256(snapshot_payload).hexdigest(),
    }


def test_harbor_commands_bind_each_exact_instance_and_never_put_auth_in_argv(tmp_path):
    inputs = _agent_inputs(tmp_path)
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        (tmp_path / "protocol.json").read_text()
    )
    smoke = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=inputs["protocol_path"],
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    scored = tb.build_harbor_commands(
        protocol,
        "scored",
        protocol_path=inputs["protocol_path"],
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    assert len(smoke) == 3
    assert len(scored) == 20
    assert [row.instance_id for row in smoke] == list(
        protocol.subset.smoke_instance_ids
    )
    for row in (*smoke, *scored):
        argv = list(row.argv)
        assert argv[argv.index("--dataset") + 1] == (
            f"{tb.DATASET}@{protocol.dataset_version}"
        )
        assert argv[argv.index("--include-task-name") + 1] == row.instance_id
        assert argv[argv.index("--n-tasks") + 1] == "1"
        assert argv[argv.index("--max-retries") + 1] == "0"
        assert argv[argv.index("--agent-timeout-multiplier") + 1] == "4"
        assert tb.AGENT_IMPORT_PATH in argv
        assert inputs["auth_path"] not in argv
        assert tb._argv_key_values(argv, "--agent-kwarg")["attempt_id"] == row.attempt_id
        assert argv[argv.index("--jobs-dir") + 1] == protocol.output_root
        assert Path(row.job_dir).parent == Path(protocol.output_root)
        assert row.command_sha256 == tb.command_digest(row.argv)


def _write_harbor_job(command, protocol, protocol_path):
    job_dir = Path(command.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    raw_kwargs = tb._argv_key_values(command.argv, "--agent-kwarg")
    agent_kwargs = {
        "wheel_path": raw_kwargs["wheel_path"],
        "codex_binary_path": raw_kwargs["codex_binary_path"],
        "protocol_path": raw_kwargs["protocol_path"],
        "reasoning_effort": raw_kwargs["reasoning_effort"],
        "instance_id": command.instance_id,
        "run_kind": command.run_kind,
        "attempt_id": command.attempt_id,
    }
    agent_config = {
        "name": tb.AGENT_IMPORT_PATH,
        "import_path": None,
        "model_name": protocol.model,
        "kwargs": agent_kwargs,
        "skills": [],
        "extra_allowed_hosts": [],
        "include_logs": [],
        "exclude_logs": [],
        "env": {},
        "mcp_servers": [],
        "resume_trajectory": False,
    }
    environment_config = {
        "type": "docker",
        "import_path": None,
        "kwargs": {},
    }
    job_config = {
        "n_concurrent_trials": 1,
        "agent_timeout_multiplier": 4,
        "datasets": [
            {
                "name": tb.DATASET,
                "ref": protocol.dataset_version,
                "task_names": [command.instance_id],
                "n_tasks": 1,
            }
        ],
        "agents": [agent_config],
    }
    (job_dir / "config.json").write_text(json.dumps(job_config))
    (job_dir / "lock.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "harbor": {"version": protocol.harbor_version},
                "n_concurrent_trials": 1,
                "retry": {"max_retries": 0},
                "trials": [
                    {
                        "agent_timeout_multiplier": 4,
                        "task": {
                            "name": command.instance_id,
                            "digest": protocol.task_content_digests[command.instance_id],
                        },
                        "agent": agent_config,
                        "environment": environment_config,
                    }
                ],
            }
        )
    )

    trial_dir = job_dir / "trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    stream = _codex_stream()
    audit = tb.audit_codex_jsonl(stream)
    image = tb.DockerImageAttestation(
        container_id="1" * 64,
        image_id="sha256:" + "d" * 64,
        configured_image=(
            f"registry.example/task@{protocol.task_image_digests[command.instance_id]}"
        ),
        repo_digests=(
            f"registry.example/task@{protocol.task_image_digests[command.instance_id]}",
        ),
        compose_project="lha-test",
        network_name="lha-test_default",
        container_ip="172.28.0.2",
    )
    protocol_sha256 = tb.sha256_file(protocol_path)
    smoke_seal_sha256 = (
        tb._validated_smoke_seal(protocol, protocol_sha256=protocol_sha256)
        if command.run_kind == "scored"
        else None
    )
    receipt = {
        "schema_version": 5,
        "type": "terminal_proxy_receipt",
        "evaluation_id": protocol.evaluation_id,
        "attempt_id": command.attempt_id,
        "source_container_id": image.container_id,
        "started_at": "2026-07-27T10:00:00+00:00",
        "stopped_at": "2026-07-27T10:00:09+00:00",
        "ttl_s": protocol.budgets.broker_ttl_s,
        "max_requests": protocol.budgets.max_model_requests,
        "max_buffered_response_bytes": tb.BROKER_MAX_BUFFERED_RESPONSE_BYTES,
        "request_retry_limit": protocol.budgets.request_max_retries,
        "stream_retry_limit": protocol.budgets.broker_stream_max_retries,
        "stream_retry_limit_per_request": (
            protocol.budgets.broker_stream_max_retries_per_request
        ),
        "downstream_accepted_requests": 1,
        "rejected_requests": 0,
        "rejection_reasons": {},
        "upstream_attempts": 1,
        "upstream_statuses": {"200": 1},
        "stream_retries_used": 0,
        "stream_retried_requests": 0,
        "max_stream_retries_on_request": 0,
        "upstream_error": None,
        "upstream_transport_errors": {},
        "upstream_stream_errors": {},
        "observed_content_types": ["text/event-stream"],
        "revoked": True,
        "outcome": "sigterm",
    }
    with tb.open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        started = tb.write_command_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=command.attempt_id,
            run_kind=command.run_kind,
            instance_id=command.instance_id,
            command_sha256=command.command_sha256,
        )
        tb.write_model_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=command.attempt_id,
            protocol_sha256=protocol_sha256,
            run_kind=command.run_kind,
            instance_id=command.instance_id,
            container_id=image.container_id,
        )
        events_sha256 = store.write_once("codex_events.jsonl", stream.encode())
        receipt_sha256 = store.write_json_once("broker_receipt.json", receipt)
    provenance = tb.TerminalBenchAgentProvenance(
        evaluation_id=protocol.evaluation_id,
        attempt_id=command.attempt_id,
        lha_version=lha.__version__,
        run_kind=command.run_kind,
        instance_id=command.instance_id,
        dataset_version=protocol.dataset_version,
        model=protocol.model,
        reasoning_effort=protocol.reasoning_effort,
        harbor_version=protocol.harbor_version,
        codex_cli_version=protocol.codex_cli_version,
        observed_codex_cli_version=protocol.codex_cli_version,
        codex_target=protocol.codex_target,
        observed_codex_target=protocol.codex_target,
        codex_binary_sha256=protocol.codex_binary_sha256,
        observed_codex_binary_sha256=protocol.codex_binary_sha256,
        broker_image_id=protocol.broker_image_id,
        task_content_digest=protocol.task_content_digests[command.instance_id],
        task_image_digest=protocol.task_image_digests[command.instance_id],
        image_attestation=image,
        post_quiescence_attestation=image,
        wheel_sha256=protocol.wheel_sha256,
        protocol_sha256=protocol_sha256,
        subset=protocol.subset,
        budgets=protocol.budgets,
        model_started=True,
        infrastructure_retries_used=0,
        codex_outcome="success",
        codex_return_code=0,
        broker_cleanup_state="succeeded",
        container_quiescence="restarted",
        smoke_seal_sha256=smoke_seal_sha256,
        codex_events_sha256=events_sha256,
        broker_receipt_sha256=receipt_sha256,
        broker_tls_certificate_sha256="e" * 64,
        broker_accepted_requests=1,
        broker_revoked=True,
        codex_audit=audit,
    )
    with tb.open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        terminal_sha256 = store.write_json_once("terminal.json", provenance)
        store.write_json_once(
            "command.json",
            tb.CommandEnvelope(
                evaluation_id=protocol.evaluation_id,
                attempt_id=command.attempt_id,
                run_kind=command.run_kind,
                instance_id=command.instance_id,
                command_sha256=command.command_sha256,
                started_at=started.started_at,
                finished_at="2026-07-27T10:00:10+00:00",
                process_return_code=0,
                outcome="completed",
                failure_stage=None,
                exception_sha256=None,
                model_started=True,
            ),
        )
    metadata = tb._agent_metadata(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        instance_id=command.instance_id,
        run_kind=command.run_kind,
        audit=audit,
        codex_events_sha256=events_sha256,
        image_attestation=image,
        terminal_record_sha256=terminal_sha256,
        provenance=provenance,
    )
    trial_result = {
        "task_name": command.instance_id,
        "task_checksum": protocol.task_checksums[command.instance_id],
        "config": {
            "agent": agent_config,
            "environment": environment_config,
        },
        "agent_info": {
            "name": "lha",
            "version": lha.__version__,
            "model_info": {"name": protocol.model, "provider": None},
        },
        "agent_result": {
            "n_input_tokens": audit.input_tokens,
            "n_cache_tokens": audit.cached_input_tokens,
            "n_output_tokens": audit.output_tokens,
            "metadata": metadata,
        },
        "verifier_result": {"rewards": {"reward": 1}},
        "exception_info": None,
        "started_at": "2026-07-27T10:00:00+00:00",
        "finished_at": "2026-07-27T10:00:10+00:00",
    }
    (trial_dir / "result.json").write_text(json.dumps(trial_result))
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_retries": 0,
                },
                "trial_results": [trial_result],
            }
        )
    )
    return job_dir, trial_dir


def test_harbor_result_manifest_rejects_set_drift(tmp_path):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    for command in commands:
        _write_harbor_job(command, protocol, protocol_path)
    manifest = tb.validate_harbor_results(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
        manifest_path=tmp_path / "manifest.json",
    )
    assert manifest.observed_instance_ids == protocol.subset.smoke_instance_ids
    assert manifest.dataset_version == protocol.dataset_version
    assert manifest.task_content_digests == {
        item: protocol.task_content_digests[item]
        for item in protocol.subset.smoke_instance_ids
    }
    assert manifest.task_checksums == {
        item: protocol.task_checksums[item]
        for item in protocol.subset.smoke_instance_ids
    }
    assert manifest.task_image_digests == {
        item: protocol.task_image_digests[item]
        for item in protocol.subset.smoke_instance_ids
    }

    bad = Path(commands[0].job_dir) / "trial" / "result.json"
    bad_result = json.loads(bad.read_text())
    bad_result["task_name"] = "outside"
    bad.write_text(json.dumps(bad_result))
    with pytest.raises(ValueError, match="disagree"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )

    tb.write_protocol(
        protocol.model_copy(update={"model": "gpt-5.4"}),
        protocol_path,
    )
    with pytest.raises(ValueError, match="does not contain"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )


def test_harbor_result_manifest_recovers_only_identical_evidence(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    first = tb.validate_harbor_results(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    original = manifest_path.read_bytes()
    second = tb.validate_harbor_results(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )

    assert second == first
    assert manifest_path.read_bytes() == original
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_harbor_result_manifest_rejects_noncanonical_existing_bytes(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    tb.validate_harbor_results(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    same_model_different_bytes = json.dumps(json.loads(manifest_path.read_text())).encode()
    manifest_path.write_bytes(same_model_different_bytes)

    with pytest.raises(ValueError, match="conflicts with current evidence"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
            manifest_path=manifest_path,
        )

    assert manifest_path.read_bytes() == same_model_different_bytes


def test_harbor_result_manifest_rejects_conflicting_evidence_without_overwrite(
    tmp_path,
):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    tb.validate_harbor_results(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    original = manifest_path.read_bytes()
    result_path = Path(commands[0].job_dir) / "trial" / "result.json"
    changed = json.loads(result_path.read_text())
    changed["finished_at"] = "2026-07-27T10:00:11+00:00"
    result_path.write_text(json.dumps(changed))
    job_result_path = Path(commands[0].job_dir) / "result.json"
    job_result = json.loads(job_result_path.read_text())
    job_result["trial_results"][0] = changed
    job_result_path.write_text(json.dumps(job_result))

    with pytest.raises(ValueError, match="conflicts with current evidence"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
            manifest_path=manifest_path,
        )

    assert manifest_path.read_bytes() == original
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def _prepared_results(tmp_path, run_kind):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    if run_kind == "scored":
        smoke_commands = tb.build_harbor_commands(
            protocol,
            "smoke",
            protocol_path=protocol_path,
            wheel_path=inputs["wheel_path"],
            codex_binary_path=inputs["codex_binary_path"],
        )
        for command in smoke_commands:
            _write_harbor_job(command, protocol, protocol_path)
        tb.seal_smoke_phase(
            protocol,
            smoke_commands,
            protocol_path=protocol_path,
            manifest_path=tmp_path / "smoke-manifest.json",
        )
    commands = tb.build_harbor_commands(
        protocol,
        run_kind,
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    for command in commands:
        _write_harbor_job(command, protocol, protocol_path)
    return protocol, protocol_path, commands


def _prepared_smoke_results(tmp_path):
    return _prepared_results(tmp_path, "smoke")


def _prepared_scored_results(tmp_path):
    return _prepared_results(tmp_path, "scored")


def test_harbor_result_rejects_forged_agent_model_and_effort(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    config_path = Path(commands[0].job_dir) / "config.json"

    original_config = config_path.read_text()
    config = json.loads(config_path.read_text())
    config["agents"][0]["name"] = "other.package:Agent"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="agent import"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )
    config_path.write_text(original_config)

    config_path = Path(commands[1].job_dir) / "config.json"
    original_config = config_path.read_text()
    config = json.loads(config_path.read_text())
    config["agents"][0]["model_name"] = "another-model"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="preregistered model"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )
    config_path.write_text(original_config)

    config_path = Path(commands[2].job_dir) / "config.json"
    config = json.loads(config_path.read_text())
    config["agents"][0]["kwargs"]["reasoning_effort"] = "low"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="critical agent kwargs"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )


def test_harbor_result_rejects_dataset_and_task_content_drift(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    job_dir = Path(commands[0].job_dir)
    config_path = job_dir / "config.json"
    lock_path = job_dir / "lock.json"
    trial_path = job_dir / "trial" / "result.json"
    job_result_path = job_dir / "result.json"

    original_config = config_path.read_text()
    config = json.loads(original_config)
    config["datasets"][0]["ref"] = "sha256:" + "0" * 64
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="immutable dataset version"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )
    config_path.write_text(original_config)

    job_dir = Path(commands[1].job_dir)
    lock_path = job_dir / "lock.json"
    original_lock = lock_path.read_text()
    job_lock = json.loads(original_lock)
    job_lock["trials"][0]["task"]["digest"] = "sha256:" + "0" * 64
    lock_path.write_text(json.dumps(job_lock))
    with pytest.raises(ValueError, match="registered task"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )
    lock_path.write_text(original_lock)

    job_dir = Path(commands[2].job_dir)
    trial_path = job_dir / "trial" / "result.json"
    job_result_path = job_dir / "result.json"
    original_trial = trial_path.read_text()
    original_job_result = job_result_path.read_text()
    trial = json.loads(original_trial)
    trial["task_checksum"] = "0" * 64
    trial_path.write_text(json.dumps(trial))
    job_result = json.loads(job_result_path.read_text())
    job_result["trial_results"] = [trial]
    job_result_path.write_text(json.dumps(job_result))
    with pytest.raises(ValueError, match="task content"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )
    trial_path.write_text(original_trial)
    job_result_path.write_text(original_job_result)


def test_harbor_result_rejects_forged_provenance_and_runtime_image(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    control_root = tb.terminal_control_root(
        protocol.output_root,
        protocol.evaluation_id,
    )
    provenance_path = control_root / commands[0].attempt_id / "terminal.json"

    original_provenance = provenance_path.read_bytes()
    provenance = json.loads(original_provenance)
    provenance["wheel_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance does not match"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )
    provenance_path.write_bytes(original_provenance)

    provenance_path = control_root / commands[1].attempt_id / "terminal.json"
    original_provenance = provenance_path.read_bytes()
    provenance = json.loads(original_provenance)
    provenance["image_attestation"]["configured_image"] = (
        "registry.example/task@sha256:" + "9" * 64
    )
    provenance["image_attestation"]["repo_digests"] = [
        "registry.example/task@sha256:" + "9" * 64
    ]
    provenance["post_quiescence_attestation"] = provenance["image_attestation"]
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="runtime Docker evidence"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )
    provenance_path.write_bytes(original_provenance)


def test_harbor_result_rejects_forged_or_invalid_codex_audit(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    control_root = tb.terminal_control_root(
        protocol.output_root,
        protocol.evaluation_id,
    )
    provenance_path = control_root / commands[0].attempt_id / "terminal.json"

    original_provenance = provenance_path.read_bytes()
    provenance = json.loads(original_provenance)
    provenance["codex_audit"]["tool_calls"] = 1
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="Codex audit changed"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )
    provenance_path.write_bytes(original_provenance)

    events_path = control_root / commands[1].attempt_id / "codex_events.jsonl"
    events_path.write_text("{forged")
    with pytest.raises(ValueError, match="JSONL digest changed"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )


def _successful_broker_receipt():
    receipt = {
        "downstream_accepted_requests": 2,
        "rejected_requests": 0,
        "rejection_reasons": {},
        "max_buffered_response_bytes": tb.BROKER_MAX_BUFFERED_RESPONSE_BYTES,
        "request_retry_limit": 1,
        "stream_retry_limit": tb.BROKER_STREAM_RETRY_LIMIT,
        "stream_retry_limit_per_request": (
            tb.BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
        ),
        "upstream_attempts": 2,
        "upstream_statuses": {"200": 2},
        "stream_retries_used": 0,
        "stream_retried_requests": 0,
        "max_stream_retries_on_request": 0,
        "upstream_error": None,
        "upstream_transport_errors": {},
        "upstream_stream_errors": {},
        "observed_content_types": ["text/event-stream"],
    }
    return receipt


def test_successful_codex_allows_only_registered_bounded_in_process_recovery():
    receipt = _successful_broker_receipt()
    assert tb._broker_receipt_proves_clean_success(receipt, 2, 0)
    receipt["observed_content_types"] = []
    assert tb._broker_receipt_proves_clean_success(receipt, 2, 0)
    receipt["observed_content_types"] = ["text/event-stream"]

    receipt.update(
        {
            "downstream_accepted_requests": 3,
            "rejected_requests": 1,
            "rejection_reasons": {"upstream_transport_exception": 1},
            "upstream_attempts": 4,
            "upstream_statuses": {"200": 3},
            "stream_retries_used": 1,
            "stream_retried_requests": 1,
            "max_stream_retries_on_request": 1,
            "upstream_transport_errors": {"RemoteProtocolError": 1},
            "upstream_stream_errors": {"RemoteProtocolError": 1},
        }
    )
    assert tb._broker_receipt_proves_clean_success(receipt, 3, 0)

    receipt["upstream_transport_errors"] = {"SensitiveConnectError": 1}
    assert not tb._broker_receipt_proves_clean_success(receipt, 3, 0)
    receipt["upstream_transport_errors"] = {"RemoteProtocolError": 2}
    receipt["rejected_requests"] = 2
    receipt["rejection_reasons"] = {"upstream_transport_exception": 2}
    receipt["downstream_accepted_requests"] = 4
    receipt["upstream_attempts"] = 5

    assert not tb._broker_receipt_proves_clean_success(receipt, 4, 0)


def test_successful_provenance_counts_request_retries_per_logical_request():
    budgets = tb.TerminalBenchBudgets()
    audit = tb.CodexRunAudit(
        event_counts={
            "thread.started": 1,
            "turn.started": 1,
            "item.completed": 49,
            "turn.completed": 1,
        },
        item_counts={"command_execution": 24, "agent_message": 25},
        tool_calls=24,
        reconnect_notices=0,
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=50,
        reasoning_output_tokens=10,
    )
    protocol = tb.create_protocol(
        evaluation_id="a" * 32,
        output_root=Path("/tmp/lha-terminal-request-count-test/jobs"),
        model="gpt-5.5",
        reasoning_effort="xhigh",
        codex_cli_version="codex-cli 0.141.0",
        codex_target="x86_64-unknown-linux-musl",
        codex_binary_path=Path(__file__),
        broker_image_id="sha256:" + "b" * 64,
        wheel_path=Path(__file__),
    )
    instance_id = protocol.subset.smoke_instance_ids[0]
    attempt_id = tb.terminal_attempt_id(
        protocol.evaluation_id,
        "smoke",
        instance_id,
    )
    image = tb.DockerImageAttestation(
        container_id="1" * 64,
        image_id="sha256:" + "d" * 64,
        configured_image=(
            f"registry.example/task@{protocol.task_image_digests[instance_id]}"
        ),
        repo_digests=(
            f"registry.example/task@{protocol.task_image_digests[instance_id]}",
        ),
        compose_project="lha-test",
        network_name="lha-test_default",
        container_ip="172.28.0.2",
    )
    base = {
        "evaluation_id": protocol.evaluation_id,
        "attempt_id": attempt_id,
        "lha_version": lha.__version__,
        "run_kind": "smoke",
        "instance_id": instance_id,
        "dataset_version": protocol.dataset_version,
        "model": protocol.model,
        "reasoning_effort": protocol.reasoning_effort,
        "harbor_version": protocol.harbor_version,
        "codex_cli_version": protocol.codex_cli_version,
        "observed_codex_cli_version": protocol.codex_cli_version,
        "codex_target": protocol.codex_target,
        "observed_codex_target": protocol.codex_target,
        "codex_binary_sha256": protocol.codex_binary_sha256,
        "observed_codex_binary_sha256": protocol.codex_binary_sha256,
        "broker_image_id": protocol.broker_image_id,
        "task_content_digest": protocol.task_content_digests[instance_id],
        "task_image_digest": protocol.task_image_digests[instance_id],
        "image_attestation": image,
        "post_quiescence_attestation": image,
        "wheel_sha256": protocol.wheel_sha256,
        "protocol_sha256": "c" * 64,
        "subset": protocol.subset,
        "budgets": budgets,
        "model_started": True,
        "infrastructure_retries_used": 0,
        "codex_outcome": "success",
        "codex_return_code": 0,
        "broker_cleanup_state": "succeeded",
        "container_quiescence": "restarted",
        "codex_events_sha256": "e" * 64,
        "broker_receipt_sha256": "f" * 64,
        "broker_tls_certificate_sha256": "9" * 64,
        "broker_revoked": True,
        "codex_audit": audit,
    }

    record = tb.TerminalBenchAgentProvenance(
        **base,
        broker_accepted_requests=36,
    )
    assert record.broker_accepted_requests == 36

    with pytest.raises(ValueError, match="inconsistent broker request counts"):
        tb.TerminalBenchAgentProvenance(
            **base,
            broker_accepted_requests=51,
        )


def test_successful_codex_requires_stream_notice_and_receipt_to_agree():
    receipt = _successful_broker_receipt()
    receipt["upstream_attempts"] = 3
    receipt["upstream_statuses"] = {"200": 3}
    receipt["stream_retries_used"] = 1
    receipt["stream_retried_requests"] = 1
    receipt["max_stream_retries_on_request"] = 1
    receipt["upstream_stream_errors"] = {"RemoteProtocolError": 1}

    assert tb._broker_receipt_proves_clean_success(receipt, 2, 0)
    assert not tb._broker_receipt_proves_clean_success(receipt, 2, 1)
    receipt["observed_content_types"] = ["application/json"]
    assert not tb._broker_receipt_proves_clean_success(receipt, 2, 0)

    receipt = _successful_broker_receipt()
    receipt["upstream_attempts"] = 6
    receipt["upstream_statuses"] = {"200": 6}
    receipt["stream_retries_used"] = 4
    receipt["stream_retried_requests"] = 1
    receipt["max_stream_retries_on_request"] = 4
    receipt["upstream_stream_errors"] = {"RemoteProtocolError": 4}
    assert tb._broker_receipt_proves_clean_success(receipt, 2, 0)
    assert not tb._broker_receipt_proves_clean_success(receipt, 2, 1)


def test_terminal_validator_accepts_only_fixed_consistent_broker_diagnostics():
    valid_receipt = {
        "downstream_accepted_requests": 2,
        "rejected_requests": 1,
        "rejection_reasons": {"upstream_transport_exception": 1},
        "max_buffered_response_bytes": tb.BROKER_MAX_BUFFERED_RESPONSE_BYTES,
        "request_retry_limit": 1,
        "stream_retry_limit": tb.BROKER_STREAM_RETRY_LIMIT,
        "stream_retry_limit_per_request": (
            tb.BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
        ),
        "upstream_attempts": 2,
        "upstream_statuses": {"200": 1},
        "stream_retries_used": 0,
        "stream_retried_requests": 0,
        "max_stream_retries_on_request": 0,
        "upstream_transport_errors": {"ReadTimeout": 1},
        "upstream_stream_errors": {},
        "observed_content_types": ["text/event-stream"],
    }
    receipt = dict(valid_receipt)
    assert tb._broker_receipt_diagnostics_are_valid(receipt)

    receipt["rejection_reasons"] = {"unregistered_internal_reason": 1}
    assert not tb._broker_receipt_diagnostics_are_valid(receipt)
    receipt = dict(valid_receipt)
    receipt["upstream_transport_errors"] = {"ReadTimeout: secret/path": 1}
    assert not tb._broker_receipt_diagnostics_are_valid(receipt)
    receipt = dict(valid_receipt)
    receipt["upstream_transport_errors"] = {}
    assert not tb._broker_receipt_diagnostics_are_valid(receipt)
    for field, invalid_value in (
        ("downstream_accepted_requests", 3),
        ("upstream_attempts", 3),
        ("upstream_statuses", {"200": 2}),
        ("stream_retries_used", 1),
        ("stream_retried_requests", 1),
        ("max_stream_retries_on_request", 1),
        ("max_buffered_response_bytes", tb.BROKER_MAX_BUFFERED_RESPONSE_BYTES - 1),
        ("observed_content_types", "text/event-stream"),
        ("observed_content_types", ["text/event-stream"] * 5),
        ("observed_content_types", ["x" * 257]),
        ("observed_content_types", ["text/event-stream\nsecret"]),
    ):
        receipt = dict(valid_receipt)
        receipt[field] = invalid_value
        assert not tb._broker_receipt_diagnostics_are_valid(receipt), field


def test_harbor_result_rejects_trial_model_and_metadata_drift(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    job_dir = Path(commands[0].job_dir)
    trial_path = job_dir / "trial" / "result.json"
    job_result_path = job_dir / "result.json"

    original_trial = trial_path.read_text()
    original_job_result = job_result_path.read_text()
    trial = json.loads(original_trial)
    trial["agent_info"]["model_info"]["name"] = "different-model"
    trial_path.write_text(json.dumps(trial))
    job_result = json.loads(job_result_path.read_text())
    job_result["trial_results"] = [trial]
    job_result_path.write_text(json.dumps(job_result))
    with pytest.raises(ValueError, match="different model"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )
    trial_path.write_text(original_trial)
    job_result_path.write_text(original_job_result)

    job_dir = Path(commands[1].job_dir)
    trial_path = job_dir / "trial" / "result.json"
    job_result_path = job_dir / "result.json"
    trial = json.loads(trial_path.read_text())
    trial["agent_result"]["metadata"]["protocol_sha256"] = "0" * 64
    trial_path.write_text(json.dumps(trial))
    job_result = json.loads(job_result_path.read_text())
    job_result["trial_results"] = [trial]
    job_result_path.write_text(json.dumps(job_result))
    with pytest.raises(ValueError, match="metadata does not match"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )


def test_terminal_summary_requires_bound_official_harbor_records(tmp_path):
    protocol, protocol_path, commands = _prepared_scored_results(tmp_path)
    manifest_path = tmp_path / "scored-manifest.json"
    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    batch = tb.derive_terminal_bench_records(
        protocol,
        commands,
        protocol_path=protocol_path,
        execution_manifest=manifest,
        manifest_path=manifest_path,
    )
    summary = tb.summarize_records(
        protocol,
        batch,
        commands=commands,
        protocol_path=protocol_path,
        execution_manifest=manifest,
        manifest_path=manifest_path,
    )
    assert (summary.passed, summary.failed, summary.errors) == (20, 0, 0)
    assert summary.success_rate == 1.0
    assert summary.p50_duration_s == 10
    assert summary.p95_duration_s == 10
    assert summary.protocol_errors == 0
    assert summary.mechanism_metrics == "unavailable"
    assert summary.incorrect_deliveries is None
    assert summary.intercepted_incorrect is None
    assert summary.false_rejections is None
    assert summary.repair_successes is None
    assert summary.repair_attempts is None
    assert summary.repair_success_rate is None
    markdown = summary.to_markdown()
    assert "固定 20 题子集" in markdown
    assert "不是完整排行榜成绩" in markdown
    assert "错误交付 / 拦截 / 错误拒绝：未测" in markdown
    assert "修复成功率：不适用" in markdown
    assert "不可评分 ERROR：0" in markdown
    assert "协议错误" not in markdown
    assert "0 / 0 / 0" not in markdown
    assert "0/0" not in markdown

    forged = batch.records[0].model_copy(
        update={"official_status": "FAIL", "independent_correct": False}
    )
    forged_batch = batch.model_copy(
        update={"records": (forged, *batch.records[1:])}
    )
    with pytest.raises(ValueError, match="do not match"):
        tb.summarize_records(
            protocol,
            forged_batch,
            commands=commands,
            protocol_path=protocol_path,
            execution_manifest=manifest,
            manifest_path=manifest_path,
        )


def test_terminal_summary_keeps_error_in_denominator_with_mechanism_metrics_unavailable(
    tmp_path,
):
    protocol, protocol_path, commands = _prepared_scored_results(tmp_path)
    job_dir = Path(commands[0].job_dir)
    trial_path = job_dir / "trial" / "result.json"
    root_result_path = job_dir / "result.json"
    trial = json.loads(trial_path.read_text())
    trial["verifier_result"] = None
    trial_path.write_text(json.dumps(trial))
    root_result = json.loads(root_result_path.read_text())
    root_result["trial_results"] = [trial]
    root_result_path.write_text(json.dumps(root_result))

    manifest_path = tmp_path / "scored-manifest.json"
    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    batch = tb.derive_terminal_bench_records(
        protocol,
        commands,
        protocol_path=protocol_path,
        execution_manifest=manifest,
        manifest_path=manifest_path,
    )
    summary = tb.summarize_records(
        protocol,
        batch,
        commands=commands,
        protocol_path=protocol_path,
        execution_manifest=manifest,
        manifest_path=manifest_path,
    )

    assert (summary.passed, summary.failed, summary.errors) == (19, 0, 1)
    assert summary.success_rate == pytest.approx(19 / 20)
    assert summary.protocol_errors == 1
    assert summary.incorrect_deliveries is None
    assert summary.intercepted_incorrect is None
    assert summary.false_rejections is None
    assert summary.repair_success_rate is None
    assert "- ERROR：1/20（保留在分母中）" in summary.to_markdown()
    assert "- 不可评分 ERROR：1" in summary.to_markdown()


def test_codex_protocol_error_remains_in_twenty_task_denominator(tmp_path):
    protocol, protocol_path, commands = _prepared_scored_results(tmp_path)
    command = commands[0]
    job_dir = Path(command.job_dir)
    trial_path = job_dir / "trial" / "result.json"
    root_result_path = job_dir / "result.json"
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / command.attempt_id
    )
    events_path = attempt_dir / "codex_events.jsonl"
    provenance_path = attempt_dir / "terminal.json"

    events_path.write_text("{malformed")
    provenance = json.loads(provenance_path.read_text())
    provenance["codex_outcome"] = "protocol_error"
    provenance["codex_return_code"] = 0
    provenance["codex_failure_kind"] = "codex_jsonl_invalid"
    provenance["codex_events_sha256"] = tb.sha256_file(events_path)
    provenance["codex_audit"] = None
    provenance["container_quiescence"] = "stopped"
    provenance["post_quiescence_attestation"] = None
    provenance_path.write_text(json.dumps(provenance))

    trial = json.loads(trial_path.read_text())
    trial["agent_result"] = None
    trial["verifier_result"] = None
    trial["exception_info"] = {"exception_type": "RuntimeError"}
    trial_path.write_text(json.dumps(trial))
    root_result = json.loads(root_result_path.read_text())
    root_result["trial_results"] = [trial]
    root_result_path.write_text(json.dumps(root_result))

    manifest_path = tmp_path / "scored-error-manifest.json"
    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    batch = tb.derive_terminal_bench_records(
        protocol,
        commands,
        protocol_path=protocol_path,
        execution_manifest=manifest,
        manifest_path=manifest_path,
    )
    summary = tb.summarize_records(
        protocol,
        batch,
        commands=commands,
        protocol_path=protocol_path,
        execution_manifest=manifest,
        manifest_path=manifest_path,
    )

    assert len(batch.records) == 20
    assert batch.records[0].official_status == "ERROR"
    assert batch.records[0].independent_correct is None
    assert batch.records[0].infrastructure_retries is None
    assert (summary.passed, summary.failed, summary.errors) == (19, 0, 1)
    assert summary.denominator == 20


def test_setup_error_has_bound_evidence_or_invalidates_the_manifest(tmp_path):
    protocol, protocol_path, commands = _prepared_scored_results(tmp_path)
    command = commands[0]
    job_dir = Path(command.job_dir)
    trial_path = job_dir / "trial" / "result.json"
    root_result_path = job_dir / "result.json"
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / command.attempt_id
    )
    events_path = attempt_dir / "codex_events.jsonl"
    provenance_path = attempt_dir / "terminal.json"

    events_path.unlink()
    (attempt_dir / "broker_receipt.json").unlink()
    (attempt_dir / "MODEL_STARTED.json").unlink()
    provenance = json.loads(provenance_path.read_text())
    provenance["observed_codex_cli_version"] = None
    provenance["observed_codex_target"] = None
    provenance["observed_codex_binary_sha256"] = None
    provenance["image_attestation"] = None
    provenance["post_quiescence_attestation"] = None
    provenance["model_started"] = False
    provenance["codex_outcome"] = "setup_error"
    provenance["codex_return_code"] = None
    provenance["codex_failure_kind"] = "agent_setup_failed"
    provenance["broker_cleanup_state"] = "not_started"
    provenance["container_quiescence"] = "not_started"
    provenance["codex_events_sha256"] = None
    provenance["broker_receipt_sha256"] = None
    provenance["broker_accepted_requests"] = None
    provenance["broker_revoked"] = None
    provenance["codex_audit"] = None
    provenance_path.write_text(json.dumps(provenance))
    envelope_path = attempt_dir / "command.json"
    envelope = json.loads(envelope_path.read_text())
    envelope["model_started"] = False
    envelope_path.write_text(json.dumps(envelope))

    trial = json.loads(trial_path.read_text())
    trial["agent_result"] = None
    trial["verifier_result"] = None
    trial["exception_info"] = {"exception_type": "AgentSetupError"}
    trial_path.write_text(json.dumps(trial))
    root_result = json.loads(root_result_path.read_text())
    root_result["trial_results"] = [trial]
    root_result_path.write_text(json.dumps(root_result))

    manifest_path = tmp_path / "setup-error-manifest.json"
    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    assert manifest.codex_events_sha256[command.instance_id] is None
    assert manifest.container_image_ids[command.instance_id] is None
    batch = tb.derive_terminal_bench_records(
        protocol,
        commands,
        protocol_path=protocol_path,
        execution_manifest=manifest,
        manifest_path=manifest_path,
    )
    assert batch.records[0].official_status == "ERROR"

    provenance_path.unlink()
    without_terminal = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
    )
    assert without_terminal.official_status[command.instance_id] == "ERROR"
    assert without_terminal.terminal_record_sha256[command.instance_id] is None

    envelope_path.unlink()
    without_envelope = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
    )
    assert without_envelope.official_status[command.instance_id] == "ERROR"
    assert without_envelope.command_envelope_sha256[command.instance_id] is None


def test_terminal_task_record_rejects_hand_entered_pass_with_false_truth():
    with pytest.raises(ValueError, match="must agree"):
        tb.TerminalBenchTaskRecord(
            instance_id="forged",
            protocol_sha256="0" * 64,
            execution_manifest_sha256="1" * 64,
            command_envelope_sha256="3" * 64,
            official_result_sha256="2" * 64,
            official_status="PASS",
            independent_correct=False,
        )


def test_terminal_task_record_rejects_unmeasured_gate_and_repair_values():
    common = {
        "instance_id": "forged",
        "protocol_sha256": "0" * 64,
        "execution_manifest_sha256": "1" * 64,
        "command_envelope_sha256": "3" * 64,
        "official_result_sha256": "2" * 64,
        "official_status": "PASS",
        "independent_correct": True,
    }
    with pytest.raises(ValueError):
        tb.TerminalBenchTaskRecord.model_validate({**common, "gate_accepted": True})
    with pytest.raises(ValueError):
        tb.TerminalBenchTaskRecord.model_validate({**common, "repairs": 0})


def test_terminal_records_reject_missing_or_forged_execution_manifest(tmp_path):
    protocol, protocol_path, commands = _prepared_scored_results(tmp_path)
    manifest_path = tmp_path / "scored-manifest.json"
    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    manifest_path.unlink()
    with pytest.raises(ValueError, match="manifest is unreadable"):
        tb.derive_terminal_bench_records(
            protocol,
            commands,
            protocol_path=protocol_path,
            execution_manifest=manifest,
            manifest_path=manifest_path,
        )

    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    forged = json.loads(manifest_path.read_text())
    forged["protocol_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="does not contain"):
        tb.derive_terminal_bench_records(
            protocol,
            commands,
            protocol_path=protocol_path,
            execution_manifest=manifest,
            manifest_path=manifest_path,
        )


def test_terminal_records_reject_inconsistent_official_harbor_fields(tmp_path):
    protocol, protocol_path, commands = _prepared_scored_results(tmp_path)
    job_dir = Path(commands[0].job_dir)
    trial_path = job_dir / "trial" / "result.json"
    root_result_path = job_dir / "result.json"
    trial = json.loads(trial_path.read_text())
    trial["verifier_result"]["rewards"]["reward"] = 2
    trial_path.write_text(json.dumps(trial))
    root_result = json.loads(root_result_path.read_text())
    root_result["trial_results"] = [trial]
    root_result_path.write_text(json.dumps(root_result))

    with pytest.raises(ValueError, match="binary value 0 or 1"):
        tb.validate_harbor_results(
            protocol,
            "scored",
            commands,
            protocol_path=protocol_path,
            manifest_path=tmp_path / "inconsistent-manifest.json",
        )

    trial["verifier_result"]["rewards"]["reward"] = 1
    trial["exception_info"] = {
        "exception_type": "VerifierCrash",
        "exception_message": "failed",
        "exception_traceback": "",
        "occurred_at": "2026-07-27T10:00:10+00:00",
    }
    trial_path.write_text(json.dumps(trial))
    root_result["trial_results"] = [trial]
    root_result_path.write_text(json.dumps(root_result))
    with pytest.raises(ValueError, match="both an exception and a verifier"):
        tb.validate_harbor_results(
            protocol,
            "scored",
            commands,
            protocol_path=protocol_path,
            manifest_path=tmp_path / "inconsistent-manifest.json",
        )

    trial["verifier_result"] = None
    trial["exception_info"] = None
    trial_path.write_text(json.dumps(trial))
    root_result["trial_results"] = [trial]
    root_result_path.write_text(json.dumps(root_result))
    manifest_path = tmp_path / "missing-verifier-manifest.json"
    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    batch = tb.derive_terminal_bench_records(
        protocol,
        commands,
        protocol_path=protocol_path,
        execution_manifest=manifest,
        manifest_path=manifest_path,
    )
    first = batch.records[0]
    assert (first.official_status, first.independent_correct) == ("ERROR", None)
    assert first.protocol_error == "Harbor trial omitted verifier_result"


def _exported_terminal_public_evidence(
    tmp_path,
    *,
    first_exception_info=None,
    first_reward=1,
):
    protocol, protocol_path, commands = _prepared_scored_results(tmp_path)
    if first_exception_info is not None or first_reward != 1:
        first_job = Path(commands[0].job_dir)
        trial_path = first_job / "trial" / "result.json"
        job_result_path = first_job / "result.json"
        trial = json.loads(trial_path.read_text())
        if first_exception_info is not None:
            trial["verifier_result"] = None
            trial["exception_info"] = first_exception_info
        else:
            trial["verifier_result"]["rewards"]["reward"] = first_reward
        trial_path.write_text(json.dumps(trial))
        job_result = json.loads(job_result_path.read_text())
        job_result["trial_results"] = [trial]
        job_result_path.write_text(json.dumps(job_result))
    smoke_manifest_path = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / "smoke_manifest.json"
    )
    scored_manifest_path = tmp_path / "scored-manifest.json"
    scored_manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=scored_manifest_path,
    )
    smoke_manifest = tb.HarborExecutionManifest.model_validate_json(
        smoke_manifest_path.read_bytes()
    )
    smoke_seal_path = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / "smoke_seal.json"
    )
    smoke_seal = tb.SmokeSeal.model_validate_json(smoke_seal_path.read_bytes())
    records = tb.derive_terminal_bench_records(
        protocol,
        commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    summary = tb.summarize_records(
        protocol,
        records,
        commands=commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    package = tmp_path / "public-evidence"
    exported = tpe.export_terminal_bench_public_evidence(
        protocol,
        protocol_path=protocol_path,
        smoke_manifest=smoke_manifest,
        smoke_manifest_path=smoke_manifest_path,
        smoke_seal=smoke_seal,
        smoke_seal_path=smoke_seal_path,
        scored_manifest=scored_manifest,
        scored_manifest_path=scored_manifest_path,
        records=records,
        summary=summary,
        scored_commands=commands,
        output_dir=package,
        public_path_root=tmp_path.resolve(),
        auth_parent=tmp_path / "private-auth",
    )
    return protocol, commands, package, exported


def test_terminal_public_evidence_round_trip_recomputes_fixed_twenty(tmp_path):
    protocol, _commands, package, exported = _exported_terminal_public_evidence(
        tmp_path
    )
    validated = tpe.validate_terminal_bench_public_evidence(package)

    assert validated == exported
    assert (validated.passed, validated.failed, validated.errors) == (20, 0, 0)
    assert validated.denominator == 20
    assert validated.model == protocol.model
    assert validated.reasoning_effort == protocol.reasoning_effort
    assert validated.harbor_version == protocol.harbor_version
    index = tpe.TerminalBenchPublicEvidenceIndex.model_validate_json(
        (package / "evidence.json").read_bytes()
    )
    assert [trial.instance_id for trial in index.trials] == list(
        protocol.subset.scored_instance_ids
    )
    assert len(index.trials) == 20
    assert {trial.raw_status for trial in index.trials} == {"PASS"}
    assert not (package / "codex_events.jsonl").exists()
    assert not (package / "broker_receipt.json").exists()
    assert not (package / "codex_stderr.txt").exists()
    public_text = "\n".join(
        path.read_text()
        for path in sorted(package.rglob("*"))
        if path.is_file()
    )
    assert "account-test-fixture" not in public_text
    assert "fixture-signature" not in public_text
    assert "LHA_CODEX_AUTH_FILE" not in public_text
    assert str(Path.home()) not in public_text
    assert str(Path(__file__).resolve().parents[1]) not in public_text
    assert str(tmp_path / "private-auth") not in public_text


def test_terminal_public_evidence_rejects_change_to_each_exported_file(tmp_path):
    _protocol, _commands, package, _exported = _exported_terminal_public_evidence(
        tmp_path
    )
    paths = sorted(path for path in package.rglob("*") if path.is_file())
    assert len(paths) == 28
    for path in paths:
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        try:
            with pytest.raises(ValueError):
                tpe.validate_terminal_bench_public_evidence(package)
        finally:
            path.write_bytes(original)
        tpe.validate_terminal_bench_public_evidence(package)


def test_terminal_public_evidence_rederives_records_not_only_file_hashes(tmp_path):
    _protocol, _commands, package, _exported = _exported_terminal_public_evidence(
        tmp_path
    )
    records_path = package / "records.json"
    records = tb.TerminalBenchRecordBatch.model_validate_json(
        records_path.read_bytes()
    )
    forged_record = records.records[0].model_copy(
        update={"official_status": "FAIL", "independent_correct": False}
    )
    forged_records = records.model_copy(
        update={"records": (forged_record, *records.records[1:])}
    )
    records_path.write_text(forged_records.model_dump_json(indent=2) + "\n")

    index_path = package / "evidence.json"
    index = tpe.TerminalBenchPublicEvidenceIndex.model_validate_json(
        index_path.read_bytes()
    )
    forged_index = index.model_copy(
        update={"records_sha256": tb.sha256_file(records_path)}
    )
    index_path.write_text(forged_index.model_dump_json(indent=2) + "\n")

    with pytest.raises(ValueError, match="official raw results"):
        tpe.validate_terminal_bench_public_evidence(package)


@pytest.mark.parametrize(
    ("raw_reward", "raw_status"),
    [(1, "PASS"), (0, "FAIL")],
)
def test_terminal_public_evidence_rejects_raw_outcome_downgraded_to_error(
    tmp_path,
    raw_reward,
    raw_status,
):
    _protocol, _commands, package, _exported = _exported_terminal_public_evidence(
        tmp_path,
        first_reward=raw_reward,
    )
    manifest_path = package / "scored_manifest.json"
    manifest = tb.HarborExecutionManifest.model_validate_json(
        manifest_path.read_bytes()
    )
    first_id = manifest.expected_instance_ids[0]
    assert manifest.official_status[first_id] == raw_status
    forged_manifest = manifest.model_copy(
        update={
            "official_status": {
                **manifest.official_status,
                first_id: "ERROR",
            },
            "protocol_errors": {
                **manifest.protocol_errors,
                first_id: "forged protocol error",
            },
        }
    )
    manifest_path.write_bytes(tpe._canonical_model_bytes(forged_manifest))

    index_path = package / "evidence.json"
    index = tpe.TerminalBenchPublicEvidenceIndex.model_validate_json(
        index_path.read_bytes()
    )
    index_path.write_bytes(
        tpe._canonical_model_bytes(
            index.model_copy(
                update={"scored_manifest_sha256": tb.sha256_file(manifest_path)}
            )
        )
    )

    with pytest.raises(ValueError, match="official raw result disagrees"):
        tpe.validate_terminal_bench_public_evidence(package)


def test_terminal_public_evidence_recomputes_fail_and_keeps_twenty_denominator(
    tmp_path,
):
    protocol, protocol_path, commands = _prepared_scored_results(tmp_path)
    first_job = Path(commands[0].job_dir)
    trial_path = first_job / "trial" / "result.json"
    job_result_path = first_job / "result.json"
    trial = json.loads(trial_path.read_text())
    trial["verifier_result"]["rewards"]["reward"] = 0
    trial_path.write_text(json.dumps(trial))
    job_result = json.loads(job_result_path.read_text())
    job_result["trial_results"] = [trial]
    job_result_path.write_text(json.dumps(job_result))

    scored_manifest_path = tmp_path / "scored-manifest.json"
    scored_manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=scored_manifest_path,
    )
    records = tb.derive_terminal_bench_records(
        protocol,
        commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    summary = tb.summarize_records(
        protocol,
        records,
        commands=commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    smoke_manifest_path = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / "smoke_manifest.json"
    )
    smoke_manifest = tb.HarborExecutionManifest.model_validate_json(
        smoke_manifest_path.read_bytes()
    )
    smoke_seal_path = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / "smoke_seal.json"
    )
    smoke_seal = tb.SmokeSeal.model_validate_json(smoke_seal_path.read_bytes())
    package = tmp_path / "public-evidence"
    tpe.export_terminal_bench_public_evidence(
        protocol,
        protocol_path=protocol_path,
        smoke_manifest=smoke_manifest,
        smoke_manifest_path=smoke_manifest_path,
        smoke_seal=smoke_seal,
        smoke_seal_path=smoke_seal_path,
        scored_manifest=scored_manifest,
        scored_manifest_path=scored_manifest_path,
        records=records,
        summary=summary,
        scored_commands=commands,
        output_dir=package,
        public_path_root=tmp_path.resolve(),
        auth_parent=tmp_path / "private-auth",
    )
    validated = tpe.validate_terminal_bench_public_evidence(package)
    assert (validated.passed, validated.failed, validated.errors) == (19, 1, 0)
    assert validated.denominator == 20


def test_terminal_public_evidence_redacts_error_traceback_secret_and_path(tmp_path):
    protocol, _commands, package, exported = _exported_terminal_public_evidence(
        tmp_path,
        first_exception_info={
            "exception_type": "ValidationError",
            "exception_message": "Bearer private-token-12345678",
            "exception_traceback": (
                "Traceback: /Users/example/.codex/auth.json "
                "sk-proj-privatecredential123"
            ),
            "occurred_at": "2026-07-27T10:00:10+00:00",
        },
    )

    assert (exported.passed, exported.failed, exported.errors) == (19, 0, 1)
    index = tpe.TerminalBenchPublicEvidenceIndex.model_validate_json(
        (package / "evidence.json").read_bytes()
    )
    first = index.trials[0]
    trial_path = package / first.path
    projection = tpe.PublicHarborErrorProjection.model_validate_json(
        trial_path.read_bytes()
    )
    scored_manifest = tb.HarborExecutionManifest.model_validate_json(
        (package / "scored_manifest.json").read_bytes()
    )

    assert first.payload_kind == "redacted_error"
    assert first.raw_status == "ERROR"
    assert first.exception_type == "ValidationError"
    assert first.source_sha256 == scored_manifest.trial_result_sha256[first.instance_id]
    assert first.payload_sha256 == tb.sha256_file(trial_path)
    assert first.payload_sha256 != first.source_sha256
    assert projection.source_sha256 == first.source_sha256
    assert projection.task_name == protocol.subset.scored_instance_ids[0]
    assert projection.exception_type == "ValidationError"
    assert set(json.loads(trial_path.read_text())) == {
        "schema_version",
        "kind",
        "source_sha256",
        "task_name",
        "task_checksum",
        "raw_status",
        "raw_correct",
        "raw_protocol_error",
        "exception_type",
        "duration_s",
        "infrastructure_retries",
    }
    public_text = "\n".join(
        path.read_text()
        for path in sorted(package.rglob("*"))
        if path.is_file()
    )
    assert "exception_traceback" not in public_text
    assert "private-token" not in public_text
    assert "privatecredential" not in public_text
    assert "/Users/example" not in public_text
    validated = tpe.validate_terminal_bench_public_evidence(package)
    assert validated == exported


@pytest.mark.parametrize(
    "exception_type",
    [
        "/Users/example/.codex/auth.json",
        "sk-proj-privatecredential123",
    ],
)
def test_terminal_public_evidence_rejects_unsafe_exception_type(exception_type):
    with pytest.raises(ValueError, match="unsafe for public evidence"):
        tpe._official_exception_type(
            {"exception_info": {"exception_type": exception_type}}
        )


def test_terminal_public_evidence_detects_redacted_error_tampering(tmp_path):
    _protocol, _commands, package, _exported = _exported_terminal_public_evidence(
        tmp_path,
        first_exception_info={"exception_type": "ValidationError"},
    )
    index_path = package / "evidence.json"
    index = tpe.TerminalBenchPublicEvidenceIndex.model_validate_json(
        index_path.read_bytes()
    )
    first = index.trials[0]
    trial_path = package / first.path
    projection = tpe.PublicHarborErrorProjection.model_validate_json(
        trial_path.read_bytes()
    ).model_copy(
        update={
            "exception_type": "RuntimeError",
            "raw_protocol_error": "Harbor trial exception: RuntimeError",
        }
    )
    trial_path.write_text(projection.model_dump_json(indent=2) + "\n")
    forged_first = first.model_copy(
        update={
            "payload_sha256": tb.sha256_file(trial_path),
            "exception_type": "RuntimeError",
            "raw_protocol_error": "Harbor trial exception: RuntimeError",
        }
    )
    index_path.write_text(
        index.model_copy(
            update={"trials": (forged_first, *index.trials[1:])}
        ).model_dump_json(indent=2)
        + "\n"
    )

    with pytest.raises(ValueError, match="raw ERROR explanation changed"):
        tpe.validate_terminal_bench_public_evidence(package)


def test_terminal_public_evidence_refuses_credentials_and_full_logs(tmp_path):
    protocol, protocol_path, commands = _prepared_scored_results(tmp_path)
    first_job = Path(commands[0].job_dir)
    trial_path = first_job / "trial" / "result.json"
    job_result_path = first_job / "result.json"
    trial = json.loads(trial_path.read_text())
    trial["access_token"] = "secret-test-token"
    trial_path.write_text(json.dumps(trial))
    job_result = json.loads(job_result_path.read_text())
    job_result["trial_results"] = [trial]
    job_result_path.write_text(json.dumps(job_result))

    scored_manifest_path = tmp_path / "scored-manifest.json"
    scored_manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=scored_manifest_path,
    )
    records = tb.derive_terminal_bench_records(
        protocol,
        commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    summary = tb.summarize_records(
        protocol,
        records,
        commands=commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    smoke_manifest_path = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / "smoke_manifest.json"
    )
    smoke_manifest = tb.HarborExecutionManifest.model_validate_json(
        smoke_manifest_path.read_bytes()
    )
    smoke_seal_path = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / "smoke_seal.json"
    )
    smoke_seal = tb.SmokeSeal.model_validate_json(smoke_seal_path.read_bytes())
    with pytest.raises(ValueError, match="credential field"):
        tpe.export_terminal_bench_public_evidence(
            protocol,
            protocol_path=protocol_path,
            smoke_manifest=smoke_manifest,
            smoke_manifest_path=smoke_manifest_path,
            smoke_seal=smoke_seal,
            smoke_seal_path=smoke_seal_path,
            scored_manifest=scored_manifest,
            scored_manifest_path=scored_manifest_path,
            records=records,
            summary=summary,
            scored_commands=commands,
            output_dir=tmp_path / "public-evidence",
            public_path_root=tmp_path.resolve(),
            auth_parent=tmp_path / "private-auth",
        )
    with pytest.raises(ValueError, match="full private log"):
        tpe._assert_public_payload(
            {"stderr": "complete private process output"},
            label="trial",
        )
    with pytest.raises(ValueError, match="private data"):
        tpe._assert_public_payload(
            {"protocol_path": "/Users/example/.codex/auth.json"},
            label="trial",
        )


class _FailedHarborProcess:
    """Small Popen double for host-runner accounting tests."""

    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __init__(self, argv, *, env, cwd, start_new_session, pass_fds):
        assert start_new_session is True
        assert len(pass_fds) == 1
        assert stat.S_ISREG(os.fstat(pass_fds[0]).st_mode)
        assert Path(cwd) == Path.cwd()
        self.argv = tuple(argv)
        self.environment = dict(env)
        self.returncode = None
        type(self).calls.append((self.argv, self.environment))

    def wait(self, timeout):
        assert timeout > 0
        self.returncode = 1
        return self.returncode

    def poll(self):
        return self.returncode


def test_host_cleanup_kills_group_after_leader_has_already_exited(monkeypatch):
    class ExitedLeader:
        pid = 424242
        returncode = -signal.SIGTERM

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert timeout > 0
            return self.returncode

    group_exists = True
    delivered: list[signal.Signals] = []

    def fake_killpg(process_group, requested_signal):
        nonlocal group_exists
        assert process_group == ExitedLeader.pid
        if requested_signal == 0:
            if not group_exists:
                raise ProcessLookupError
            return
        delivered.append(requested_signal)
        if requested_signal == signal.SIGKILL:
            group_exists = False

    monkeypatch.setattr(tb.os, "killpg", fake_killpg)

    tb._stop_host_process(ExitedLeader())

    assert delivered == [signal.SIGTERM, signal.SIGKILL]
    assert group_exists is False


def test_host_command_slot_is_consumed_exactly_once(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    command = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=inputs["protocol_path"],
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )[0]
    _FailedHarborProcess.calls = []
    monkeypatch.setattr(tb.subprocess, "Popen", _FailedHarborProcess)

    envelope = tb.run_harbor_command_once(
        protocol,
        command,
        protocol_path=inputs["protocol_path"],
        auth_path=inputs["auth_path"],
    )
    assert envelope.outcome == "error"
    assert envelope.failure_stage == "environment_setup"
    assert envelope.model_started is False
    assert len(_FailedHarborProcess.calls) == 1
    argv, environment = _FailedHarborProcess.calls[0]
    assert inputs["auth_path"] not in argv
    assert environment["LHA_CODEX_AUTH_FILE"] == inputs["auth_path"]

    with pytest.raises(tb.ControlRecordExists, match="already consumed"):
        tb.run_harbor_command_once(
            protocol,
            command,
            protocol_path=inputs["protocol_path"],
            auth_path=inputs["auth_path"],
        )
    assert len(_FailedHarborProcess.calls) == 1


def test_scored_phase_refuses_to_start_before_smoke_is_sealed(
    monkeypatch,
    tmp_path,
):
    inputs = _agent_inputs(tmp_path)
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    commands = tb.build_harbor_commands(
        protocol,
        "scored",
        protocol_path=inputs["protocol_path"],
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    called = []

    def should_not_run(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("a scored Harbor command started before smoke was sealed")

    monkeypatch.setattr(tb, "run_harbor_command_once", should_not_run)
    with pytest.raises(tb.ControlStoreError, match="smoke seal"):
        tb.run_terminal_phase(
            protocol,
            "scored",
            commands,
            protocol_path=inputs["protocol_path"],
            auth_path=inputs["auth_path"],
        )
    assert called == []


def test_phase_resume_skips_only_complete_prefix(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    _write_harbor_job(commands[0], protocol, protocol_path)
    called: list[str] = []

    def complete_pending(_protocol, command, **_kwargs):
        called.append(command.instance_id)
        _write_harbor_job(command, protocol, protocol_path)
        envelope, *_ = tb._load_attempt_control(
            protocol,
            command,
            protocol_sha256=tb.sha256_file(protocol_path),
        )
        return envelope

    monkeypatch.setattr(tb, "run_harbor_command_once", complete_pending)
    envelopes = tb.run_terminal_phase(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
        auth_path=inputs["auth_path"],
    )
    assert len(envelopes) == 3
    assert called == [command.instance_id for command in commands[1:]]


def test_phase_resume_blocks_unattested_partial_and_refuses_out_of_order_control(
    monkeypatch,
    tmp_path,
):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    with tb.open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        commands[0].attempt_id,
    ) as store:
        tb.write_command_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=commands[0].attempt_id,
            run_kind="smoke",
            instance_id=commands[0].instance_id,
            command_sha256=commands[0].command_sha256,
        )
    called: list[str] = []
    monkeypatch.setattr(
        tb,
        "run_harbor_command_once",
        lambda *_args, **_kwargs: called.append("called"),
    )
    with pytest.raises(tb.ControlStoreError, match="cleanup cannot be proven"):
        tb.run_terminal_phase(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
            auth_path=inputs["auth_path"],
        )
    assert called == []

    other_inputs = _agent_inputs(tmp_path / "out-of-order")
    other_path = Path(other_inputs["protocol_path"])
    other_protocol = tb.TerminalBenchProtocol.model_validate_json(other_path.read_text())
    other_commands = tb.build_harbor_commands(
        other_protocol,
        "smoke",
        protocol_path=other_path,
        wheel_path=other_inputs["wheel_path"],
        codex_binary_path=other_inputs["codex_binary_path"],
    )
    _write_harbor_job(other_commands[1], other_protocol, other_path)
    with pytest.raises(tb.ControlStoreError, match="out of order"):
        tb.run_terminal_phase(
            other_protocol,
            "smoke",
            other_commands,
            protocol_path=other_path,
            auth_path=other_inputs["auth_path"],
        )
    assert called == []


def test_smoke_resume_stops_after_an_existing_failed_envelope(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    _FailedHarborProcess.calls = []
    monkeypatch.setattr(tb.subprocess, "Popen", _FailedHarborProcess)
    first = tb.run_harbor_command_once(
        protocol,
        commands[0],
        protocol_path=protocol_path,
        auth_path=inputs["auth_path"],
    )
    assert first.outcome == "error"
    called: list[str] = []
    monkeypatch.setattr(
        tb,
        "run_harbor_command_once",
        lambda *_args, **_kwargs: called.append("called"),
    )
    envelopes = tb.run_terminal_phase(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
        auth_path=inputs["auth_path"],
    )
    assert envelopes == (first,)
    assert called == []


def test_smoke_stops_after_zero_exit_harbor_trial_exception(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    called: list[str] = []

    def complete_with_trial_exception(_protocol, command, **_kwargs):
        called.append(command.instance_id)
        job_dir, trial_dir = _write_harbor_job(command, protocol, protocol_path)
        trial_path = trial_dir / "result.json"
        trial_result = json.loads(trial_path.read_text())
        trial_result["verifier_result"] = None
        trial_result["exception_info"] = {"exception_type": "RuntimeError"}
        trial_path.write_text(json.dumps(trial_result))
        job_result_path = job_dir / "result.json"
        job_result = json.loads(job_result_path.read_text())
        job_result["trial_results"] = [trial_result]
        job_result_path.write_text(json.dumps(job_result))
        envelope, *_ = tb._load_attempt_control(
            protocol,
            command,
            protocol_sha256=tb.sha256_file(protocol_path),
        )
        return envelope

    monkeypatch.setattr(tb, "run_harbor_command_once", complete_with_trial_exception)
    envelopes = tb.run_terminal_phase(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
        auth_path=inputs["auth_path"],
    )

    assert len(envelopes) == 1
    assert envelopes[0].outcome == "completed"
    assert envelopes[0].process_return_code == 0
    assert called == [commands[0].instance_id]
    manifest = tb.validate_harbor_results(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
    )
    first_id = commands[0].instance_id
    assert manifest.official_status[first_id] == "ERROR"
    assert manifest.protocol_errors[first_id] == "Harbor trial exception: RuntimeError"


def test_scored_resume_counts_abandoned_attempt_and_continues_pending(
    monkeypatch,
    tmp_path,
):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    smoke_commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    for command in smoke_commands:
        _write_harbor_job(command, protocol, protocol_path)
    tb.seal_smoke_phase(protocol, smoke_commands, protocol_path=protocol_path)
    commands = tb.build_harbor_commands(
        protocol,
        "scored",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    with tb.open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        commands[0].attempt_id,
    ) as store:
        tb.write_command_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=commands[0].attempt_id,
            run_kind="scored",
            instance_id=commands[0].instance_id,
            command_sha256=commands[0].command_sha256,
        )
        tb.write_model_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=commands[0].attempt_id,
            protocol_sha256=tb.sha256_file(protocol_path),
            run_kind="scored",
            instance_id=commands[0].instance_id,
            container_id="c" * 64,
        )
    cleanup_calls: list[dict[str, object]] = []

    class CleanupController:
        def __init__(self, **_kwargs):
            pass

        def cleanup_abandoned(self, **kwargs):
            cleanup_calls.append(kwargs)

    monkeypatch.setattr(tb, "TerminalProxyController", CleanupController)
    called: list[str] = []

    def complete_pending(_protocol, command, **_kwargs):
        called.append(command.instance_id)
        return tb.CommandEnvelope(
            evaluation_id=protocol.evaluation_id,
            attempt_id=command.attempt_id,
            run_kind="scored",
            instance_id=command.instance_id,
            command_sha256=command.command_sha256,
            started_at="2026-07-27T10:00:00+00:00",
            finished_at="2026-07-27T10:00:01+00:00",
            process_return_code=0,
            outcome="completed",
            failure_stage=None,
            exception_sha256=None,
            model_started=False,
        )

    monkeypatch.setattr(tb, "run_harbor_command_once", complete_pending)
    envelopes = tb.run_terminal_phase(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        auth_path=inputs["auth_path"],
    )
    assert len(envelopes) == 20
    assert envelopes[0].outcome == "interrupted"
    assert cleanup_calls == [
        {
            "evaluation_id": protocol.evaluation_id,
            "attempt_id": commands[0].attempt_id,
            "source_container_id": "c" * 64,
            "expected_task_image_digest": protocol.task_image_digests[
                commands[0].instance_id
            ],
        }
    ]
    assert called == [command.instance_id for command in commands[1:]]


def test_abandoned_attempt_rejects_model_marker_from_another_protocol(tmp_path):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    command = commands[0]
    with tb.open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        tb.write_command_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=command.attempt_id,
            run_kind="smoke",
            instance_id=command.instance_id,
            command_sha256=command.command_sha256,
        )
        tb.write_model_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=command.attempt_id,
            protocol_sha256="f" * 64,
            run_kind="smoke",
            instance_id=command.instance_id,
            container_id="c" * 64,
        )
    with pytest.raises(tb.ControlStoreError, match="changed its binding"):
        tb.run_terminal_phase(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
            auth_path=inputs["auth_path"],
        )


def test_abandoned_attempt_cleanup_failure_blocks_without_writing_envelope(
    monkeypatch,
    tmp_path,
):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    command = commands[0]
    with tb.open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        tb.write_command_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=command.attempt_id,
            run_kind="smoke",
            instance_id=command.instance_id,
            command_sha256=command.command_sha256,
        )
        tb.write_model_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=command.attempt_id,
            protocol_sha256=tb.sha256_file(protocol_path),
            run_kind="smoke",
            instance_id=command.instance_id,
            container_id="c" * 64,
        )

    class FailedCleanupController:
        def __init__(self, **_kwargs):
            pass

        def cleanup_abandoned(self, **_kwargs):
            raise tb.TerminalProxyError("Docker daemon unavailable")

    monkeypatch.setattr(tb, "TerminalProxyController", FailedCleanupController)
    with pytest.raises(tb.ControlStoreError, match="cleanup could not be proven"):
        tb.run_terminal_phase(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
            auth_path=inputs["auth_path"],
        )
    with tb.open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        assert not store.has("command.json")


def test_missing_harbor_jobs_are_twenty_error_rows_after_one_shot_execution(
    monkeypatch,
    tmp_path,
):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    smoke_commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    for command in smoke_commands:
        _write_harbor_job(command, protocol, protocol_path)
    tb.seal_smoke_phase(
        protocol,
        smoke_commands,
        protocol_path=protocol_path,
        manifest_path=tmp_path / "smoke-manifest.json",
    )

    scored_commands = tb.build_harbor_commands(
        protocol,
        "scored",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    _FailedHarborProcess.calls = []
    monkeypatch.setattr(tb.subprocess, "Popen", _FailedHarborProcess)
    envelopes = tb.run_terminal_phase(
        protocol,
        "scored",
        scored_commands,
        protocol_path=protocol_path,
        auth_path=inputs["auth_path"],
    )
    assert len(envelopes) == 20
    assert len(_FailedHarborProcess.calls) == 20
    assert all(envelope.outcome == "error" for envelope in envelopes)

    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        scored_commands,
        protocol_path=protocol_path,
    )
    assert set(manifest.official_status.values()) == {"ERROR"}
    assert all(value is None for value in manifest.trial_result_sha256.values())
    assert all(value is not None for value in manifest.protocol_errors.values())
    assert len(manifest.command_envelope_sha256) == 20


def test_unstarted_scored_commands_remain_twenty_error_rows(tmp_path):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    smoke_commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )
    for command in smoke_commands:
        _write_harbor_job(command, protocol, protocol_path)
    tb.seal_smoke_phase(protocol, smoke_commands, protocol_path=protocol_path)
    scored_commands = tb.build_harbor_commands(
        protocol,
        "scored",
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
    )

    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        scored_commands,
        protocol_path=protocol_path,
    )
    assert len(manifest.official_status) == 20
    assert set(manifest.official_status.values()) == {"ERROR"}
    assert all(value is None for value in manifest.command_envelope_sha256.values())
    assert all(
        value == "registered Harbor command was not started"
        for value in manifest.protocol_errors.values()
    )


def test_smoke_seal_is_idempotent_immutable_and_binds_terminal_records(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    seal, manifest = tb.seal_smoke_phase(
        protocol,
        commands,
        protocol_path=protocol_path,
        manifest_path=tmp_path / "smoke-manifest.json",
    )
    assert seal.smoke_instance_ids == protocol.subset.smoke_instance_ids
    assert seal.terminal_record_sha256 == {
        instance_id: manifest.terminal_record_sha256[instance_id]
        for instance_id in protocol.subset.smoke_instance_ids
    }
    repeated_seal, repeated_manifest = tb.seal_smoke_phase(
        protocol,
        commands,
        protocol_path=protocol_path,
    )
    assert repeated_seal == seal
    assert repeated_manifest == manifest

    command = commands[0]
    terminal_path = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / command.attempt_id
        / "terminal.json"
    )
    original = terminal_path.read_bytes()
    terminal_path.write_bytes(original + b" ")
    with pytest.raises(tb.ControlStoreError, match="changed after sealing"):
        tb._validated_smoke_seal(
            protocol,
            protocol_sha256=tb.sha256_file(protocol_path),
        )


def test_smoke_seal_recovers_after_manifest_was_published(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    manifest = tb.validate_harbor_results(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
    )
    with tb.open_control_store(protocol.output_root, protocol.evaluation_id) as store:
        store.write_json_once("smoke_manifest.json", manifest)

    protocol_sha256 = tb.sha256_file(protocol_path)
    registration = tb._control_registration(
        protocol,
        protocol_sha256=protocol_sha256,
    )
    tb.initialize_control_store(
        evaluation_id=protocol.evaluation_id,
        protocol_sha256=protocol_sha256,
        output_root=protocol.output_root,
        attempts=registration.attempts,
    )
    seal, resumed_manifest = tb.seal_smoke_phase(
        protocol,
        commands,
        protocol_path=protocol_path,
    )

    assert resumed_manifest == manifest
    assert seal.manifest_sha256 == hashlib.sha256(
        (
            tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
            / "smoke_manifest.json"
        ).read_bytes()
    ).hexdigest()


def test_smoke_seal_recovers_an_interrupted_manifest_write(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    with tb.open_control_store(protocol.output_root, protocol.evaluation_id) as store:
        pending = store._pending_name("smoke_manifest.json")
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=store._fd,
        )
        try:
            os.write(descriptor, b"{")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    protocol_sha256 = tb.sha256_file(protocol_path)
    registration = tb._control_registration(
        protocol,
        protocol_sha256=protocol_sha256,
    )
    tb.initialize_control_store(
        evaluation_id=protocol.evaluation_id,
        protocol_sha256=protocol_sha256,
        output_root=protocol.output_root,
        attempts=registration.attempts,
    )
    seal, manifest = tb.seal_smoke_phase(
        protocol,
        commands,
        protocol_path=protocol_path,
    )

    assert seal.manifest_sha256
    assert manifest.run_kind == "smoke"


def test_smoke_seal_is_serialized_across_controllers(tmp_path, monkeypatch):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    original_validate = tb.validate_harbor_results
    first_entered = Event()
    release_first = Event()
    blocked = False

    def block_first(*args, **kwargs):
        nonlocal blocked
        if not blocked:
            blocked = True
            first_entered.set()
            assert release_first.wait(timeout=5)
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(tb, "validate_harbor_results", block_first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            tb.seal_smoke_phase,
            protocol,
            commands,
            protocol_path=protocol_path,
        )
        assert first_entered.wait(timeout=5)
        second = pool.submit(
            tb.seal_smoke_phase,
            protocol,
            commands,
            protocol_path=protocol_path,
        )
        with pytest.raises(tb.ControlStoreError, match="already active"):
            second.result(timeout=5)
        release_first.set()
        seal, manifest = first.result(timeout=5)

    assert seal.manifest_sha256
    assert manifest.run_kind == "smoke"


def test_smoke_seal_detects_every_bound_evidence_file_change(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    tb.seal_smoke_phase(
        protocol,
        commands,
        protocol_path=protocol_path,
        manifest_path=tmp_path / "display-smoke-manifest.json",
    )
    control_root = tb.terminal_control_root(
        protocol.output_root,
        protocol.evaluation_id,
    )
    command = commands[0]
    attempt_root = control_root / command.attempt_id
    job_root = Path(command.job_dir)
    paths = (
        control_root / "smoke_manifest.json",
        attempt_root / "COMMAND_STARTED.json",
        attempt_root / "command.json",
        attempt_root / "terminal.json",
        attempt_root / "codex_events.jsonl",
        attempt_root / "broker_receipt.json",
        job_root / "config.json",
        job_root / "lock.json",
        job_root / "result.json",
        job_root / "trial" / "result.json",
    )
    for path in paths:
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        try:
            with pytest.raises((tb.ControlStoreError, ValueError)):
                tb._validated_smoke_seal(
                    protocol,
                    protocol_sha256=tb.sha256_file(protocol_path),
                )
        finally:
            path.write_bytes(original)
        assert (
            tb._validated_smoke_seal(
                protocol,
                protocol_sha256=tb.sha256_file(protocol_path),
            )
            == tb.sha256_file(control_root / "smoke_seal.json")
        )


def test_build_agent_without_harbor_raises_with_hint(monkeypatch):
    for mod in list(sys.modules):
        if mod == "harbor" or mod.startswith("harbor."):
            monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, "harbor", None)  # forces ImportError
    with pytest.raises(ImportError, match="3.12"):
        tb.build_agent()


def _stub_harbor(monkeypatch) -> None:
    class BaseInstalledAgent:
        def __init__(self, logs_dir, *args, extra_env=None, **kwargs):
            self.logs_dir = logs_dir

    chain = [
        "harbor",
        "harbor.agents",
        "harbor.agents.installed",
        "harbor.agents.installed.base",
        "harbor.environments",
        "harbor.environments.docker",
        "harbor.environments.docker.docker",
    ]
    for name in chain:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    setattr(
        sys.modules["harbor.agents.installed.base"],
        "BaseInstalledAgent",
        BaseInstalledAgent,
    )
    setattr(
        sys.modules["harbor.environments.docker.docker"],
        "DockerEnvironment",
        _FakeEnv,
    )


def test_harbor_stream_reader_uses_the_registered_codex_line_boundary():
    class StreamEnvironment:
        def __init__(self):
            self.observed_limit = None

        async def _collect_streamed_output(self, process, **_kwargs):
            self.observed_limit = process.stdout._limit
            return "finished"

    async def exercise():
        environment = StreamEnvironment()
        reader = asyncio.StreamReader(limit=64 * 1024)
        process = types.SimpleNamespace(stdout=reader)
        original_limit = reader._limit
        with tb._harbor_stream_line_limit(
            environment,
            maximum_line_bytes=2 * 1024 * 1024,
        ):
            assert await environment._collect_streamed_output(process) == "finished"
        assert reader._limit == original_limit
        assert "_collect_streamed_output" not in environment.__dict__
        return environment.observed_limit

    assert asyncio.run(exercise()) == (2 * 1024 * 1024) + (64 * 1024)


def test_nested_asyncio_line_overrun_is_classified_from_exception_context():
    overrun = asyncio.LimitOverrunError("line too long", consumed=70_000)
    wrapped = ValueError("Separator is not found, and chunk exceeds the limit")
    wrapped.__context__ = overrun

    assert tb._contains_exception(wrapped, asyncio.LimitOverrunError)


class _FakeEnv:
    """Records setup and returns a synthetic tool-enabled Codex event stream."""

    def __init__(
        self,
        event_stream: str | None,
        *,
        run_returncode: int = 0,
        fail_commands: int = 0,
        stderr_stream: str = "",
        output_fragments: list[str] | None = None,
    ):
        self.commands: list[str] = []
        self.exec_calls: list[dict[str, object]] = []
        self.uploads: list[tuple[str, str]] = []
        self._event_stream = event_stream
        self._run_returncode = run_returncode
        self._fail_commands = fail_commands
        self._stderr_stream = stderr_stream
        self._output_fragments = output_fragments
        self.compose_commands: list[list[str]] = []
        self.uploaded_payloads: dict[str, bytes] = {}
        self._output_callback = None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append(command)
        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_sec": timeout_sec,
                "user": user,
            }
        )
        if self._fail_commands:
            self._fail_commands -= 1
            return types.SimpleNamespace(return_code=127, stdout="", stderr="infrastructure")
        stdout = ""
        return_code = 0
        if "uname -m" in command and "/usr/local/bin/codex --version" in command:
            stdout = "\n".join(
                (
                    "x86_64",
                    hashlib.sha256(b"standalone codex binary").hexdigest(),
                    "codex-cli 0.141.0",
                )
            )
        if "codex exec" in command and self._event_stream is not None:
            stdout = self._event_stream
            return_code = self._run_returncode
            if self._output_callback is not None:
                fragments = (
                    self._event_stream.splitlines(keepends=True)
                    if self._output_fragments is None
                    else self._output_fragments
                )
                for fragment in fragments:
                    await self._output_callback(fragment, "stdout")
        if f"cat {tb._CODEX_STDERR_PATH}" in command:
            stdout = self._stderr_stream
        return types.SimpleNamespace(return_code=return_code, stdout=stdout, stderr="")

    async def upload_file(self, source_path, target_path):
        self.uploads.append((str(source_path), str(target_path)))
        self.uploaded_payloads[str(target_path)] = Path(source_path).read_bytes()

    @contextmanager
    def scoped_output_callback(self, callback):
        previous = self._output_callback
        self._output_callback = callback
        try:
            yield
        finally:
            self._output_callback = previous

    async def _run_docker_compose_command(self, command, **_kwargs):
        self.compose_commands.append(list(command))
        return types.SimpleNamespace(
            return_code=0,
            stdout="1" * 64 + "\n",
            stderr="",
        )


def _agent_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wheel = tmp_path / "lha.whl"
    wheel.write_bytes(b"not a real wheel: the Harbor environment is stubbed")
    auth = tmp_path / "private-auth" / "auth.json"
    auth.parent.mkdir()
    jwt_header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    jwt_claims = (
        base64.urlsafe_b64encode(
            json.dumps({"exp": int(time.time()) + 86_400}).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": f"{jwt_header}.{jwt_claims}.fixture-signature",
                    "account_id": "account-test-fixture",
                }
            }
        )
    )
    auth.chmod(0o600)
    codex_binary = tmp_path / "codex"
    codex_binary.write_bytes(b"standalone codex binary")
    protocol = tb.create_protocol(
        evaluation_id="2" * 32,
        output_root=tmp_path / "jobs",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        codex_cli_version="codex-cli 0.141.0",
        codex_target="x86_64-unknown-linux-musl",
        codex_binary_path=codex_binary,
        broker_image_id="sha256:" + "f" * 64,
        wheel_path=wheel,
    )
    protocol_path = tb.write_protocol(protocol, tmp_path / "protocol.json")
    smoke_commands = tb.build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=wheel,
        codex_binary_path=codex_binary,
    )
    scored_commands = tb.build_harbor_commands(
        protocol,
        "scored",
        protocol_path=protocol_path,
        wheel_path=wheel,
        codex_binary_path=codex_binary,
    )
    tb.initialize_terminal_evaluation(
        protocol,
        (*smoke_commands, *scored_commands),
        protocol_path=protocol_path,
    )
    instance_id = protocol.subset.smoke_instance_ids[0]
    attempt_id = tb.terminal_attempt_id(protocol.evaluation_id, "smoke", instance_id)
    return {
        "wheel_path": str(wheel),
        "codex_binary_path": str(codex_binary),
        "auth_path": str(auth),
        "model": "gpt-5.5",
        "reasoning_effort": "xhigh",
        "protocol_path": str(protocol_path),
        "instance_id": instance_id,
        "run_kind": "smoke",
        "attempt_id": attempt_id,
    }


def _make_agent(monkeypatch, logs_dir, inputs):
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    task_image_digest = protocol.task_image_digests.get(
        inputs["instance_id"],
        next(iter(protocol.task_image_digests.values())),
    )
    image = tb.DockerImageAttestation(
        container_id="1" * 64,
        image_id="sha256:" + "d" * 64,
        configured_image=f"registry.example/lha-task@{task_image_digest}",
        repo_digests=(f"registry.example/lha-task@{task_image_digest}",),
        compose_project="lha-test",
        network_name="lha-test_default",
        container_ip="172.28.0.2",
    )

    async def attest(_environment):
        return image

    async def restart(_environment, before):
        assert before == image
        return image

    async def kill(_environment, container_id):
        assert container_id == image.container_id

    class FakeHandle:
        base_url = "https://lha-terminal-proxy:8080"
        tls_certificate_pem = b"fixture certificate"
        tls_certificate_sha256 = hashlib.sha256(tls_certificate_pem).hexdigest()

        def capability_environment(self):
            return {"LHA_TERMINAL_PROXY_CAPABILITY": "capability_test_fixture_1234567890"}

        def binding_headers(self):
            return {
                "X-LHA-Evaluation-ID": "2" * 32,
                "X-LHA-Attempt-ID": inputs["attempt_id"],
                "X-LHA-Container-ID": image.container_id,
            }

    class FakeProxyController:
        def __init__(self):
            self.starts = []
            self.stops = []

        def start(self, **kwargs):
            self.starts.append(kwargs)
            return FakeHandle()

        def stop(self, handle):
            self.stops.append(handle)
            return {
                "schema_version": 5,
                "type": "terminal_proxy_receipt",
                "evaluation_id": "2" * 32,
                "attempt_id": inputs["attempt_id"],
                "source_container_id": image.container_id,
                "started_at": "2026-07-27T10:00:00+00:00",
                "stopped_at": "2026-07-27T10:00:01+00:00",
                "ttl_s": 2100,
                "max_requests": 60,
                "max_buffered_response_bytes": tb.BROKER_MAX_BUFFERED_RESPONSE_BYTES,
                "request_retry_limit": 1,
                "stream_retry_limit": tb.BROKER_STREAM_RETRY_LIMIT,
                "stream_retry_limit_per_request": (
                    tb.BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
                ),
                "downstream_accepted_requests": 1,
                "rejected_requests": 0,
                "rejection_reasons": {},
                "upstream_attempts": 1,
                "upstream_statuses": {"200": 1},
                "stream_retries_used": 0,
                "stream_retried_requests": 0,
                "max_stream_retries_on_request": 0,
                "upstream_error": None,
                "upstream_transport_errors": {},
                "upstream_stream_errors": {},
                "observed_content_types": ["text/event-stream"],
                "revoked": True,
                "outcome": "sigterm",
            }

    controller = FakeProxyController()
    monkeypatch.setattr(tb, "_attest_harbor_docker_image", attest)
    monkeypatch.setattr(tb, "_restart_and_confirm_main", restart)
    monkeypatch.setattr(tb, "_kill_and_confirm_main", kill)
    monkeypatch.setattr(tb, "TerminalProxyController", lambda **_kwargs: controller)
    _stub_harbor(monkeypatch)
    agent = tb.build_agent()(logs_dir=logs_dir, **inputs)
    agent._test_proxy_controller = controller
    short_name = inputs["instance_id"].rsplit("/", 1)[-1]
    agent.session_id = f"{short_name[:32].rstrip('_-')}__abc1234__agent"
    return agent


def test_agent_install_uploads_wheel_and_records_secret_free_provenance(
    monkeypatch, tmp_path
):
    inputs = _agent_inputs(tmp_path)
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
    assert type(agent).name() == "lha"
    assert agent.version() == lha.__version__

    env = _FakeEnv(None)
    asyncio.run(agent.install(env))
    assert env.uploads == [(inputs["codex_binary_path"], "/tmp/.lha_codex_binary.upload")]
    install = next(
        command for command in env.commands if "/usr/local/bin/codex" in command
    )
    assert "-m 4755" in install
    assert "/usr/local/bin/codex" in install
    wrapper_install = next(
        command for command in env.commands if "/tmp/.lha-privileged-bash" in command
    )
    assert "if [ -e /proc/self/fd/3 ]; then exec 3<&-; fi" in wrapper_install
    assert "install -o 0 -g 0 -m 4755 /bin/bash" in wrapper_install
    assert inputs["auth_path"] not in json.dumps(env.uploads)
    assert "/tmp/.lha_terminal_proxy_capability" not in env.uploaded_payloads
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    with tb.open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        inputs["attempt_id"],
    ) as store:
        assert not store.has("MODEL_STARTED.json")
        assert not store.has("terminal.json")
    assert agent._codex_version == "codex-cli 0.141.0"
    version_check = next(
        command
        for command in env.commands
        if command.endswith("/usr/local/bin/codex --version 2>/dev/null")
    )
    assert "/usr/local/bin/codex --version 2>/dev/null" in version_check
    assert agent._image_attestation is not None
    assert agent._image_attestation.image_id == "sha256:" + "d" * 64
    assert not (tmp_path / "logs").exists()
    raw_protocol = Path(inputs["protocol_path"]).read_text()
    assert "secret-test-fixture" not in raw_protocol
    assert inputs["auth_path"] not in raw_protocol


def test_agent_refuses_runtime_image_without_registered_digest(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)

    async def forged_attestation(_environment):
        return tb.DockerImageAttestation(
            container_id="1" * 64,
            image_id="sha256:" + "d" * 64,
            configured_image="registry.example/other:latest",
            repo_digests=("registry.example/other@sha256:" + "9" * 64,),
            compose_project="lha-test",
            network_name="lha-test_default",
            container_ip="172.28.0.2",
        )

    monkeypatch.setattr(tb, "_attest_harbor_docker_image", forged_attestation)
    env = _FakeEnv(None)
    with pytest.raises(RuntimeError, match="task-image digest"):
        asyncio.run(agent.install(env))
    assert env.uploads == []
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    provenance_path = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / inputs["attempt_id"]
        / "terminal.json"
    )
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        provenance_path.read_text()
    )
    assert provenance.codex_outcome == "setup_error"
    assert provenance.codex_failure_kind == "agent_setup_failed"
    assert provenance.image_attestation is None


def test_agent_install_without_wheel_fails_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("LHA_WHEEL", raising=False)
    inputs = _agent_inputs(tmp_path / "inputs")
    inputs["wheel_path"] = ""
    agent = _make_agent(monkeypatch, tmp_path, inputs)
    with pytest.raises(RuntimeError, match="wheel"):
        asyncio.run(agent.install(_FakeEnv(None)))


def test_agent_refuses_provenance_drift_before_upload(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    inputs["model"] = "different-model"
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
    env = _FakeEnv(None)
    with pytest.raises(RuntimeError, match="model does not match"):
        asyncio.run(agent.install(env))
    assert env.uploads == []

    inputs = _agent_inputs(tmp_path / "second")
    (tmp_path / "second" / "lha.whl").write_bytes(b"changed after preregistration")
    agent = _make_agent(monkeypatch, tmp_path / "other-logs", inputs)
    with pytest.raises(RuntimeError, match="wheel does not match"):
        asyncio.run(agent.install(_FakeEnv(None)))

    inputs = _agent_inputs(tmp_path / "third")
    Path(inputs["codex_binary_path"]).write_bytes(b"changed after preregistration")
    agent = _make_agent(monkeypatch, tmp_path / "binary-logs", inputs)
    with pytest.raises(RuntimeError, match="Codex binary does not match"):
        asyncio.run(agent.install(_FakeEnv(None)))


def test_agent_binds_the_registered_instance_to_harbor_session(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path / "outside")
    inputs["instance_id"] = "terminal-bench/not-preregistered"
    agent = _make_agent(monkeypatch, tmp_path / "outside-logs", inputs)
    env = _FakeEnv(None)
    with pytest.raises(RuntimeError, match="outside the smoke set"):
        asyncio.run(agent.install(env))
    assert env.uploads == []

    inputs = _agent_inputs(tmp_path / "wrong-session")
    agent = _make_agent(monkeypatch, tmp_path / "session-logs", inputs)
    agent.session_id = "different-task__abc1234__agent"
    env = _FakeEnv(None)
    with pytest.raises(RuntimeError, match="does not match the registered instance"):
        asyncio.run(agent.install(env))
    assert env.uploads == []


def test_agent_runs_real_tool_stream_in_place_and_populates_usage(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    stream = _codex_stream(
        {
            "type": "item.started",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "command": "make",
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "command": "make",
                "aggregated_output": "ok\n",
                "exit_code": 0,
                "status": "completed",
            },
        },
    )
    env = _FakeEnv(stream)
    ctx = types.SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        metadata=None,
    )
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
    asyncio.run(agent.install(env))
    asyncio.run(agent.run("configure it", env, ctx))
    assert sum("codex exec" in command for command in env.commands) == 1
    codex_call = next(
        call for call in env.exec_calls if "codex exec" in str(call["command"])
    )
    assert codex_call["user"] == "60000:60000"
    assert not any("lha run" in command for command in env.commands)
    assert not any(inputs["auth_path"] in command for command in env.commands)
    assert all(target != "/tmp/.lha_codex_auth.upload" for _, target in env.uploads)
    assert tb._CAPABILITY_STAGING in env.uploaded_payloads
    assert tb._TLS_CERT_STAGING in env.uploaded_payloads
    assert b"secret-test-fixture" not in env.uploaded_payloads[
        tb._CAPABILITY_STAGING
    ]
    assert (ctx.n_input_tokens, ctx.n_cache_tokens, ctx.n_output_tokens) == (10, 2, 3)
    assert ctx.metadata["instance_id"] == inputs["instance_id"]
    assert ctx.metadata["codex_tool_calls"] == 1
    assert ctx.metadata["codex_reasoning_output_tokens"] == 1
    assert not (tmp_path / "logs" / "codex_events.jsonl").exists()
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / inputs["attempt_id"]
    )
    assert (attempt_dir / "codex_events.jsonl").read_text() == stream
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (attempt_dir / "terminal.json").read_text()
    )
    assert provenance.codex_outcome == "success"
    assert provenance.broker_tls_certificate_sha256 == hashlib.sha256(
        b"fixture certificate"
    ).hexdigest()
    assert provenance.broker_cleanup_state == "succeeded"
    assert provenance.broker_revoked is True
    assert provenance.container_quiescence == "restarted"
    assert provenance.codex_audit is not None
    assert provenance.codex_audit.tool_calls == 1
    assert provenance.codex_events_sha256 == ctx.metadata["codex_events_sha256"]
    assert provenance.image_attestation is not None
    assert provenance.image_attestation.image_id == ctx.metadata["container_image_id"]
    assert provenance.post_quiescence_attestation == provenance.image_attestation
    assert len(agent._test_proxy_controller.starts) == 1
    assert len(agent._test_proxy_controller.stops) == 1
    with pytest.raises(RuntimeError, match="exactly one Codex run"):
        asyncio.run(agent.run("run twice", env, ctx))


def test_agent_stops_at_the_129th_tool_without_retry(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    events = []
    for index in range(1, 130):
        item_id = f"tool-{index}"
        events.append(
            {
                "type": "item.started",
                "item": {
                    "id": item_id,
                    "type": "command_execution",
                    "command": "true",
                    "aggregated_output": "",
                    "exit_code": None,
                    "status": "in_progress",
                },
            }
        )
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "id": item_id,
                    "type": "command_execution",
                    "command": "true",
                    "aggregated_output": "",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        )
    env = _FakeEnv(_codex_stream(*events))
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
    asyncio.run(agent.install(env))

    with pytest.raises(RuntimeError, match="tool call 129"):
        asyncio.run(agent.run("use too many tools", env, types.SimpleNamespace()))
    assert sum("codex exec" in command for command in env.commands) == 1
    assert len(agent._test_proxy_controller.starts) == 1
    assert len(agent._test_proxy_controller.stops) == 1

    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / inputs["attempt_id"]
    )
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (attempt_dir / "terminal.json").read_text()
    )
    assert provenance.codex_outcome == "protocol_error"
    assert provenance.codex_failure_kind == "codex_tool_budget_exceeded"
    assert provenance.container_quiescence == "stopped"
    assert provenance.codex_audit is None


def test_agent_run_protocol_error_is_not_retried_and_credentials_are_cleaned(
    monkeypatch, tmp_path
):
    inputs = _agent_inputs(tmp_path)
    env = _FakeEnv("{broken")
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
    asyncio.run(agent.install(env))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        asyncio.run(agent.run("fix it", env, types.SimpleNamespace()))
    assert sum("codex exec" in command for command in env.commands) == 1
    assert all(target != "/tmp/.lha_codex_auth.upload" for _, target in env.uploads)
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / inputs["attempt_id"]
    )
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (attempt_dir / "terminal.json").read_text()
    )
    assert provenance.codex_outcome == "protocol_error"
    assert provenance.codex_failure_kind == "codex_jsonl_invalid"
    assert provenance.broker_cleanup_state == "succeeded"
    assert provenance.container_quiescence == "stopped"

    reported_inputs = _agent_inputs(tmp_path / "reported-inputs")
    reported_stream = "\n".join(
        json.dumps(row)
        for row in (
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "turn.started"},
            {"type": "error", "message": "upstream request failed"},
        )
    )
    reported = _FakeEnv(
        reported_stream,
        run_returncode=1,
        stderr_stream="request failed without credentials\n",
    )
    agent = _make_agent(monkeypatch, tmp_path / "reported-logs", reported_inputs)
    asyncio.run(agent.install(reported))
    with pytest.raises(RuntimeError, match="reported error"):
        asyncio.run(agent.run("fix it", reported, types.SimpleNamespace()))
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(reported_inputs["protocol_path"]).read_text()
    )
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / reported_inputs["attempt_id"]
    )
    assert (attempt_dir / "codex_events.jsonl").read_text() == reported_stream
    assert (attempt_dir / "codex_stderr.txt").read_text() == (
        "request failed without credentials\n"
    )
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (attempt_dir / "terminal.json").read_text()
    )
    assert provenance.codex_outcome == "protocol_error"
    assert provenance.codex_failure_kind == "codex_reported_error"
    assert provenance.codex_return_code == 1
    assert provenance.broker_cleanup_state == "succeeded"
    assert provenance.container_quiescence == "stopped"

    nonzero_inputs = _agent_inputs(tmp_path / "nonzero-inputs")
    nonzero = _FakeEnv("", run_returncode=2)
    agent = _make_agent(monkeypatch, tmp_path / "nonzero-logs", nonzero_inputs)
    asyncio.run(agent.install(nonzero))
    with pytest.raises(RuntimeError, match="exited 2"):
        asyncio.run(agent.run("fix it", nonzero, types.SimpleNamespace()))
    assert sum("codex exec" in command for command in nonzero.commands) == 1
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(nonzero_inputs["protocol_path"]).read_text()
    )
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / nonzero_inputs["attempt_id"]
    )
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (attempt_dir / "terminal.json").read_text()
    )
    assert provenance.codex_outcome == "process_error"
    assert provenance.codex_return_code == 2
    assert provenance.container_quiescence == "stopped"


@pytest.mark.parametrize("callback_mode", ["fragmented-lines", "final-buffer"])
def test_agent_rejects_capability_split_across_stream_fragments(
    monkeypatch,
    tmp_path,
    callback_mode,
):
    inputs = _agent_inputs(tmp_path)
    capability = "capability_test_fixture_1234567890"
    midpoint = len(capability) // 2
    first = capability[:midpoint]
    second = capability[midpoint:]
    if callback_mode == "fragmented-lines":
        event_stream = first + "\n" + second
        output_fragments = [first + "\n", second]
    else:
        event_stream = capability
        output_fragments = []
    env = _FakeEnv(
        event_stream,
        output_fragments=output_fragments,
    )
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
    asyncio.run(agent.install(env))

    with pytest.raises(tb.CapabilityExposureError, match="capability"):
        asyncio.run(agent.run("fix it", env, types.SimpleNamespace()))

    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / inputs["attempt_id"]
    )
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (attempt_dir / "terminal.json").read_text()
    )
    assert provenance.codex_outcome == "protocol_error"
    assert provenance.codex_failure_kind == "codex_capability_exposed"
    assert (attempt_dir / "codex_events.jsonl").read_text() == ""
    assert capability not in "\n".join(
        path.read_text()
        for path in attempt_dir.iterdir()
        if path.is_file()
    )


def test_agent_callback_aborts_when_total_stream_limit_is_exceeded(
    monkeypatch,
    tmp_path,
):
    inputs = _agent_inputs(tmp_path)
    detector_type = tb._BoundedSecretStreamDetector
    monkeypatch.setattr(
        tb,
        "_BoundedSecretStreamDetector",
        lambda secret, *, max_total_bytes: detector_type(
            secret,
            max_total_bytes=64,
        ),
    )
    env = _FakeEnv("x" * 65, output_fragments=["x" * 65])
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
    asyncio.run(agent.install(env))

    with pytest.raises(tb.CodexEventError, match="exceeded the registered byte limit"):
        asyncio.run(agent.run("fix it", env, types.SimpleNamespace()))

    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / inputs["attempt_id"]
    )
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (attempt_dir / "terminal.json").read_text()
    )
    assert provenance.codex_outcome == "protocol_error"
    assert provenance.codex_failure_kind == "codex_jsonl_invalid"
    assert (attempt_dir / "codex_events.jsonl").read_text() == ""


def test_agent_setup_failure_is_not_retried_after_task_container_start(
    monkeypatch,
    tmp_path,
):
    inputs = _agent_inputs(tmp_path)
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
    env = _FakeEnv(_codex_stream(), fail_commands=1)
    with pytest.raises(RuntimeError, match="non-zero status"):
        asyncio.run(agent.install(env))
    assert len(env.commands) == 1
    assert sum("codex exec" in command for command in env.commands) == 0
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / inputs["attempt_id"]
    )
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (attempt_dir / "terminal.json").read_text()
    )
    assert provenance.infrastructure_retries_used == 0
    assert provenance.codex_outcome == "setup_error"


def test_agent_cancellation_waits_for_credential_cleanup(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)

    class BlockingEnv(_FakeEnv):
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            if "codex exec" in command:
                self.commands.append(command)
                await asyncio.Event().wait()
            return await super().exec(
                command,
                cwd=cwd,
                env=env,
                timeout_sec=timeout_sec,
                user=user,
            )

    async def scenario():
        env = BlockingEnv(None)
        agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
        await agent.install(env)
        task = asyncio.create_task(agent.run("fix it", env, types.SimpleNamespace()))
        while not any("codex exec" in command for command in env.commands):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return env, agent

    env, agent = asyncio.run(scenario())
    assert all(target != "/tmp/.lha_codex_auth.upload" for _, target in env.uploads)
    assert len(agent._test_proxy_controller.stops) == 1
    protocol = tb.TerminalBenchProtocol.model_validate_json(
        Path(inputs["protocol_path"]).read_text()
    )
    attempt_dir = (
        tb.terminal_control_root(protocol.output_root, protocol.evaluation_id)
        / inputs["attempt_id"]
    )
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (attempt_dir / "terminal.json").read_text()
    )
    assert provenance.codex_outcome == "execution_error"
    assert provenance.codex_failure_kind == "codex_cancelled"
    assert provenance.broker_cleanup_state == "succeeded"
    assert provenance.container_quiescence == "stopped"


# --- the --json usage summary the adapter consumes ---------------------------
def test_usage_totals_sums_the_trace(tmp_path):
    from lha.cli import _usage_totals

    trace = tmp_path / "llm_trace.jsonl"
    trace.write_text(
        json.dumps({"usage": {"input_tokens": 5, "output_tokens": 2, "cost_usd": 0.01}})
        + "\n"
        + json.dumps({"usage": None})
        + "\n"
        + '{"torn'
        + "\n"
        + json.dumps({"usage": {"input_tokens": 3, "output_tokens": 1, "cost_usd": 0.02}})
        + "\n"
    )
    totals = _usage_totals(tmp_path)
    assert totals == {
        "calls": 3,
        "input_tokens": 8,
        "output_tokens": 3,
        "cost_usd": pytest.approx(0.03),
    }
    assert _usage_totals(tmp_path / "nope") is None
