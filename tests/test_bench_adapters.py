"""Benchmark adapter contracts; no test here calls a model."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

import lha
from lha.bench import (
    Prediction,
    cluster_bootstrap_ci,
    eval_command,
    mcnemar_exact,
    parse_report,
    write_predictions,
)
from lha.bench import terminal_bench as tb
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

    assert cluster_bootstrap_ci({}) is None
    assert cluster_bootstrap_ci({"t": []}) is None


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
            or {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 3},
        },
    ]
    return "\n".join(json.dumps(row) for row in rows)


def test_codex_exec_is_tool_enabled_and_uses_harbor_as_outer_sandbox():
    cmd = tb.codex_exec_command("gpt-5.4-mini", "high", "configure the service")
    assert "CODEX_HOME=/tmp/lha_codex_home" in cmd
    assert "codex exec" in cmd
    assert "--sandbox danger-full-access" in cmd
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert "--json" in cmd
    assert "LHA_" not in cmd  # not the text-only LHA ablation patch path
    assert "configure the service" in cmd


def test_codex_stream_allows_completed_tools_and_rejects_protocol_damage():
    stream = _codex_stream(
        {
            "type": "item.started",
            "item": {"id": "tool-1", "type": "command_execution", "command": "make"},
        },
        {
            "type": "item.completed",
            "item": {"id": "tool-1", "type": "command_execution", "exit_code": 0},
        },
    )
    audit = tb.audit_codex_jsonl(stream)
    assert audit.tool_calls == 1
    assert (audit.input_tokens, audit.cached_input_tokens, audit.output_tokens) == (10, 2, 3)

    # Codex 0.141's published JSONL contract emits file_change only after the
    # patch attempt finishes, without an item.started event.
    file_change = tb.audit_codex_jsonl(
        _codex_stream(
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
    with pytest.raises(RuntimeError, match="unknown Codex event"):
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
            json.dumps({"type": "thread.started"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {"id": "tool-1", "type": "command_execution"},
                }
            ),
            json.dumps({"type": "turn.completed", "usage": {}}),
        )
    )
    with pytest.raises(RuntimeError, match="unfinished tools"):
        tb.audit_codex_jsonl(unfinished)
    with pytest.raises(RuntimeError, match="unknown Codex item type"):
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


def test_protocol_records_exact_provenance_and_contains_no_secret(tmp_path):
    wheel = tmp_path / "lha.whl"
    wheel.write_bytes(b"wheel bytes")
    codex_binary = tmp_path / "codex"
    codex_binary.write_bytes(b"codex linux binary")
    ids = [f"instance-{index}" for index in range(30)]
    protocol = tb.create_protocol(
        ids,
        model="gpt-5.4-mini-2026-06-01",
        reasoning_effort="high",
        harbor_version="0.20.0",
        codex_cli_version="codex-cli 0.141.0",
        codex_target="x86_64-unknown-linux-musl",
        codex_binary_path=codex_binary,
        task_image_digests=_image_map(ids),
        wheel_path=wheel,
    )
    assert protocol.budgets.timeout_s == 1800
    assert protocol.budgets.max_tool_calls == 20
    assert protocol.budgets.scored_runs_per_task == 1
    assert protocol.budgets.infrastructure_retries == 1
    assert protocol.budgets.task_retries == 0
    assert len(protocol.task_image_digests) == 23
    assert protocol.codex_binary_sha256 == hashlib.sha256(
        b"codex linux binary"
    ).hexdigest()
    assert protocol.wheel_sha256 == hashlib.sha256(b"wheel bytes").hexdigest()

    path = tb.write_protocol(protocol, tmp_path / "protocol.json")
    raw = path.read_text()
    assert "auth" not in raw.lower()
    assert tb.TerminalBenchProtocol.model_validate_json(raw) == protocol


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
        auth_path=inputs["auth_path"],
        jobs_dir=tmp_path / "jobs",
    )
    scored = tb.build_harbor_commands(
        protocol,
        "scored",
        protocol_path=inputs["protocol_path"],
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
        auth_path=inputs["auth_path"],
        jobs_dir=tmp_path / "jobs",
    )
    assert len(smoke) == 3
    assert len(scored) == 20
    assert [row.instance_id for row in smoke] == list(
        protocol.subset.smoke_instance_ids
    )
    for row in (*smoke, *scored):
        argv = list(row.argv)
        assert argv[argv.index("--include-task-name") + 1] == row.instance_id
        assert argv[argv.index("--n-tasks") + 1] == "1"
        assert argv[argv.index("--max-retries") + 1] == "0"
        assert tb.AGENT_IMPORT_PATH in argv
        assert inputs["auth_path"] not in argv


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
        "trusted_local_auth": True,
        "externally_sandboxed": True,
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
        "datasets": [
            {
                "name": tb.DATASET,
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
                        "task": {
                            "name": command.instance_id,
                            "digest": "sha256:" + "e" * 64,
                        },
                        "agent": agent_config,
                        "environment": environment_config,
                    }
                ],
            }
        )
    )

    trial_dir = job_dir / "trial"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    stream = _codex_stream()
    events_path = agent_dir / "codex_events.jsonl"
    events_path.write_text(stream)
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
    )
    provenance = tb.TerminalBenchAgentProvenance(
        lha_version=lha.__version__,
        run_kind=command.run_kind,
        instance_id=command.instance_id,
        model=protocol.model,
        reasoning_effort=protocol.reasoning_effort,
        harbor_version=protocol.harbor_version,
        codex_cli_version=protocol.codex_cli_version,
        codex_target=protocol.codex_target,
        codex_binary_sha256=protocol.codex_binary_sha256,
        task_image_digest=protocol.task_image_digests[command.instance_id],
        image_attestation=image,
        wheel_sha256=protocol.wheel_sha256,
        protocol_sha256=tb.sha256_file(protocol_path),
        subset=protocol.subset,
        budgets=protocol.budgets,
        infrastructure_retries_used=0,
        codex_events_sha256=tb.sha256_file(events_path),
        codex_audit=audit,
    )
    (agent_dir / "terminal_bench_agent.json").write_text(
        provenance.model_dump_json(indent=2) + "\n"
    )
    metadata = tb._agent_metadata(
        protocol=protocol,
        protocol_sha256=tb.sha256_file(protocol_path),
        instance_id=command.instance_id,
        run_kind=command.run_kind,
        audit=audit,
        codex_events_sha256=tb.sha256_file(events_path),
        image_attestation=image,
        infrastructure_retries_used=0,
    )
    trial_result = {
        "task_name": command.instance_id,
        "task_checksum": "f" * 64,
        "config": {
            "agent": agent_config,
            "environment": environment_config,
        },
        "agent_info": {
            "name": "lha",
            "version": lha.__version__,
            "model_info": {"name": protocol.model},
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
        auth_path=inputs["auth_path"],
        jobs_dir=tmp_path / "jobs",
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
        protocol.model_copy(update={"model": "different-model"}),
        protocol_path,
    )
    with pytest.raises(ValueError, match="does not contain"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )


def _prepared_results(tmp_path, run_kind):
    inputs = _agent_inputs(tmp_path)
    protocol_path = Path(inputs["protocol_path"])
    protocol = tb.TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
    commands = tb.build_harbor_commands(
        protocol,
        run_kind,
        protocol_path=protocol_path,
        wheel_path=inputs["wheel_path"],
        codex_binary_path=inputs["codex_binary_path"],
        auth_path=inputs["auth_path"],
        jobs_dir=tmp_path / "jobs",
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

    _write_harbor_job(commands[0], protocol, protocol_path)
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

    _write_harbor_job(commands[0], protocol, protocol_path)
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


def test_harbor_result_rejects_forged_provenance_and_runtime_image(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    provenance_path = (
        Path(commands[0].job_dir) / "trial" / "agent" / "terminal_bench_agent.json"
    )

    provenance = json.loads(provenance_path.read_text())
    provenance["wheel_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance does not match"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )

    _write_harbor_job(commands[0], protocol, protocol_path)
    provenance = json.loads(provenance_path.read_text())
    provenance["image_attestation"]["configured_image"] = (
        "registry.example/task@sha256:" + "9" * 64
    )
    provenance["image_attestation"]["repo_digests"] = [
        "registry.example/task@sha256:" + "9" * 64
    ]
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="runtime Docker evidence"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )


def test_harbor_result_rejects_forged_or_invalid_codex_audit(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    agent_dir = Path(commands[0].job_dir) / "trial" / "agent"
    provenance_path = agent_dir / "terminal_bench_agent.json"

    provenance = json.loads(provenance_path.read_text())
    provenance["codex_audit"]["tool_calls"] = 1
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance does not match"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )

    _write_harbor_job(commands[0], protocol, protocol_path)
    (agent_dir / "codex_events.jsonl").write_text("{forged")
    with pytest.raises(ValueError, match="JSONL failed audit"):
        tb.validate_harbor_results(
            protocol,
            "smoke",
            commands,
            protocol_path=protocol_path,
        )


def test_harbor_result_rejects_trial_model_and_metadata_drift(tmp_path):
    protocol, protocol_path, commands = _prepared_smoke_results(tmp_path)
    job_dir = Path(commands[0].job_dir)
    trial_path = job_dir / "trial" / "result.json"
    job_result_path = job_dir / "result.json"

    trial = json.loads(trial_path.read_text())
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

    _write_harbor_job(commands[0], protocol, protocol_path)
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


def test_terminal_task_record_rejects_hand_entered_pass_with_false_truth():
    with pytest.raises(ValueError, match="must agree"):
        tb.TerminalBenchTaskRecord(
            instance_id="forged",
            protocol_sha256="0" * 64,
            execution_manifest_sha256="1" * 64,
            official_result_sha256="2" * 64,
            official_status="PASS",
            independent_correct=False,
        )


def test_terminal_task_record_rejects_unmeasured_gate_and_repair_values():
    common = {
        "instance_id": "forged",
        "protocol_sha256": "0" * 64,
        "execution_manifest_sha256": "1" * 64,
        "official_result_sha256": "2" * 64,
        "official_status": "PASS",
        "independent_correct": True,
    }
    with pytest.raises(ValueError):
        tb.TerminalBenchTaskRecord(**common, gate_accepted=True)
    with pytest.raises(ValueError):
        tb.TerminalBenchTaskRecord(**common, repairs=0)


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

    manifest_path = tmp_path / "inconsistent-manifest.json"
    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    with pytest.raises(ValueError, match="binary value 0 or 1"):
        tb.derive_terminal_bench_records(
            protocol,
            commands,
            protocol_path=protocol_path,
            execution_manifest=manifest,
            manifest_path=manifest_path,
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
    manifest = tb.validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    with pytest.raises(ValueError, match="both an exception and a verifier"):
        tb.derive_terminal_bench_records(
            protocol,
            commands,
            protocol_path=protocol_path,
            execution_manifest=manifest,
            manifest_path=manifest_path,
        )

    trial["verifier_result"] = None
    trial["exception_info"] = None
    trial_path.write_text(json.dumps(trial))
    root_result["trial_results"] = [trial]
    root_result_path.write_text(json.dumps(root_result))
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

    chain = ["harbor", "harbor.agents", "harbor.agents.installed", "harbor.agents.installed.base"]
    for name in chain:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["harbor.agents.installed.base"].BaseInstalledAgent = BaseInstalledAgent


class _FakeEnv:
    """Records setup and returns a synthetic tool-enabled Codex event stream."""

    def __init__(
        self,
        event_stream: str | None,
        *,
        run_returncode: int = 0,
        fail_commands: int = 0,
    ):
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self._event_stream = event_stream
        self._run_returncode = run_returncode
        self._fail_commands = fail_commands
        self.compose_commands: list[list[str]] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append(command)
        if self._fail_commands:
            self._fail_commands -= 1
            return types.SimpleNamespace(return_code=127, stdout="", stderr="infrastructure")
        stdout = ""
        return_code = 0
        if command == "codex --version":
            stdout = "codex-cli 0.141.0\n"
        if "codex exec" in command and self._event_stream is not None:
            stdout = self._event_stream
            return_code = self._run_returncode
        return types.SimpleNamespace(return_code=return_code, stdout=stdout, stderr="")

    async def upload_file(self, source_path, target_path):
        self.uploads.append((str(source_path), str(target_path)))

    async def _run_docker_compose_command(self, command):
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
    auth = tmp_path / "auth.json"
    auth.write_text('{"tokens":"secret-test-fixture"}')
    codex_binary = tmp_path / "codex"
    codex_binary.write_bytes(b"standalone codex binary")
    protocol = tb.create_protocol(
        (ids := [f"instance-{index}" for index in range(30)]),
        model="gpt-5.4-mini-2026-06-01",
        reasoning_effort="high",
        harbor_version="0.20.0",
        codex_cli_version="codex-cli 0.141.0",
        codex_target="x86_64-unknown-linux-musl",
        codex_binary_path=codex_binary,
        task_image_digests=_image_map(ids, "c"),
        wheel_path=wheel,
    )
    protocol_path = tb.write_protocol(protocol, tmp_path / "protocol.json")
    instance_id = protocol.subset.smoke_instance_ids[0]
    return {
        "wheel_path": str(wheel),
        "codex_binary_path": str(codex_binary),
        "auth_path": str(auth),
        "model": "gpt-5.4-mini-2026-06-01",
        "reasoning_effort": "high",
        "protocol_path": str(protocol_path),
        "instance_id": instance_id,
        "run_kind": "smoke",
        "trusted_local_auth": True,
        "externally_sandboxed": True,
    }


def _make_agent(monkeypatch, logs_dir, inputs):
    async def inspect_container(container_id):
        return tb.DockerImageAttestation(
            container_id=container_id,
            image_id="sha256:" + "d" * 64,
            configured_image="registry.example/lha-task@sha256:" + "c" * 64,
            repo_digests=("registry.example/lha-task@sha256:" + "c" * 64,),
        )

    monkeypatch.setattr(tb, "_inspect_docker_container", inspect_container)
    _stub_harbor(monkeypatch)
    agent = tb.build_agent()(logs_dir=logs_dir, **inputs)
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
    assert any("install -m 755" in command for command in env.commands)
    provenance = json.loads((tmp_path / "logs" / "terminal_bench_agent.json").read_text())
    assert provenance["codex_cli_version"] == "codex-cli 0.141.0"
    assert provenance["model"] == "gpt-5.4-mini-2026-06-01"
    assert provenance["reasoning_effort"] == "high"
    assert provenance["task_image_digest"] == "sha256:" + "c" * 64
    assert provenance["image_attestation"]["image_id"] == "sha256:" + "d" * 64
    assert provenance["instance_id"] == inputs["instance_id"]
    assert len(provenance["subset"]["scored_instance_ids"]) == 20
    assert "auth" not in json.dumps(provenance).lower()


def test_agent_refuses_runtime_image_without_registered_digest(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)

    async def forged_inspection(container_id):
        return tb.DockerImageAttestation(
            container_id=container_id,
            image_id="sha256:" + "d" * 64,
            configured_image="registry.example/other:latest",
            repo_digests=("registry.example/other@sha256:" + "9" * 64,),
        )

    monkeypatch.setattr(tb, "_inspect_docker_container", forged_inspection)
    env = _FakeEnv(None)
    with pytest.raises(RuntimeError, match="task-image digest"):
        asyncio.run(agent.install(env))
    assert env.uploads == []


def test_agent_install_without_wheel_fails_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("LHA_WHEEL", raising=False)
    inputs = _agent_inputs(tmp_path / "inputs")
    inputs["wheel_path"] = None
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
            "item": {"id": "tool-1", "type": "command_execution", "command": "make"},
        },
        {
            "type": "item.completed",
            "item": {"id": "tool-1", "type": "command_execution", "exit_code": 0},
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
    assert not any("lha run" in command for command in env.commands)
    assert any("install -m 600" in command for command in env.commands)
    assert env.commands[-1] == (
        "rm -rf /tmp/lha_codex_home && rm -f /tmp/.lha_codex_auth.upload"
    )
    assert (ctx.n_input_tokens, ctx.n_cache_tokens, ctx.n_output_tokens) == (10, 2, 3)
    assert ctx.metadata["instance_id"] == inputs["instance_id"]
    assert ctx.metadata["codex_tool_calls"] == 1
    assert (tmp_path / "logs" / "codex_events.jsonl").read_text() == stream
    provenance = tb.TerminalBenchAgentProvenance.model_validate_json(
        (tmp_path / "logs" / "terminal_bench_agent.json").read_text()
    )
    assert provenance.codex_audit is not None
    assert provenance.codex_audit.tool_calls == 1
    assert provenance.codex_events_sha256 == ctx.metadata["codex_events_sha256"]
    assert provenance.image_attestation.image_id == ctx.metadata["container_image_id"]


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
    assert env.commands[-1] == (
        "rm -rf /tmp/lha_codex_home && rm -f /tmp/.lha_codex_auth.upload"
    )

    nonzero = _FakeEnv("", run_returncode=2)
    agent = _make_agent(monkeypatch, tmp_path / "nonzero", inputs)
    asyncio.run(agent.install(nonzero))
    with pytest.raises(RuntimeError, match="exited 2"):
        asyncio.run(agent.run("fix it", nonzero, types.SimpleNamespace()))
    assert sum("codex exec" in command for command in nonzero.commands) == 1
    assert nonzero.commands[-1].startswith("rm -rf /tmp/lha_codex_home")


def test_one_shared_infrastructure_retry_across_install_and_auth(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)
    agent = _make_agent(monkeypatch, tmp_path / "logs", inputs)
    env = _FakeEnv(_codex_stream(), fail_commands=1)
    asyncio.run(agent.install(env))
    # The first install failure consumes the only shared retry.
    env._fail_commands = 1
    with pytest.raises(RuntimeError, match="already exhausted"):
        asyncio.run(agent.run("fix it", env, types.SimpleNamespace()))
    assert sum("codex exec" in command for command in env.commands) == 0


def test_agent_cancellation_waits_for_credential_cleanup(monkeypatch, tmp_path):
    inputs = _agent_inputs(tmp_path)

    class BlockingEnv(_FakeEnv):
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            self.commands.append(command)
            if "codex exec" in command:
                await asyncio.Event().wait()
            if command == "codex --version":
                return types.SimpleNamespace(
                    return_code=0,
                    stdout="codex-cli 0.141.0\n",
                    stderr="",
                )
            return types.SimpleNamespace(return_code=0, stdout="", stderr="")

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
        return env

    env = asyncio.run(scenario())
    assert env.commands[-1] == (
        "rm -rf /tmp/lha_codex_home && rm -f /tmp/.lha_codex_auth.upload"
    )


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
