from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from lha.ablation import (
    AblationProvenance,
    AblationReport,
    RunRecord,
    _aggregate,
    _frozen_artifact_bytes,
    _input_snapshot_digest,
    _llm_call_receipt_bytes,
    _repo_digest,
    _report_fingerprint,
    _scorer_evidence_bytes,
    _source_file_digests,
    _source_tree_digest,
    load_ablation_report,
)
from lha.bench.terminal_public_evidence import (
    TerminalBenchPublicEvidenceValidation,
)
from lha.horizon import run_horizon
from lha.release_claims import ReleaseClaimsError, validate_release_claims

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_public_claims_validate():
    summary = validate_release_claims(REPO_ROOT)
    assert summary.status in {"legacy", "formal"}
    assert summary.tasks > 0
    assert summary.repetitions > 0
    assert summary.scheduled_cells == summary.tasks * summary.repetitions
    assert summary.cells == summary.usable_cells
    assert summary.scheduled_cells == summary.usable_cells + summary.error_cells
    assert summary.terminal_bench is not None
    assert summary.terminal_bench.evaluated_commit_sha is not None
    assert summary.terminal_bench.evaluated_tree_sha is not None
    assert summary.terminal_bench.evaluated_wheel_filename is not None
    assert summary.terminal_bench.evaluated_wheel_size_bytes is not None
    assert summary.terminal_bench.evaluated_wheel_sha256 is not None


def _terminal_validation() -> TerminalBenchPublicEvidenceValidation:
    return TerminalBenchPublicEvidenceValidation(
        evidence_tree_sha256="a" * 64,
        evaluation_id="b" * 32,
        protocol_sha256="c" * 64,
        scored_manifest_sha256="d" * 64,
        records_sha256="e" * 64,
        evaluated_commit_sha="f" * 40,
        evaluated_tree_sha="1" * 40,
        evaluated_wheel_filename="lha-0.4.2.dev0-py3-none-any.whl",
        evaluated_wheel_size_bytes=365_480,
        evaluated_wheel_sha256="2" * 64,
        model="gpt-5.5",
        reasoning_effort="xhigh",
        harbor_version="0.20.0",
        passed=12,
        failed=5,
        errors=3,
    )


def _append_terminal_claim(root: Path, claim: str) -> None:
    readme = root / "README.md"
    readme.write_text(
        readme.read_text().replace(
            "## 其他",
            f"{claim}\n\n## 其他",
        )
    )


def test_terminal_public_evidence_is_validated_and_bound_to_readme(
    tmp_path,
    monkeypatch,
):
    import lha.release_claims as claims

    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    evidence_dir = root / "benchmarks/terminal_bench_2_1"
    evidence_dir.mkdir()
    terminal = _terminal_validation()
    calls: list[Path] = []

    def validate(path):
        calls.append(Path(path))
        return terminal

    monkeypatch.setattr(
        claims,
        "validate_terminal_bench_public_evidence",
        validate,
    )
    _append_terminal_claim(
        root,
        "Terminal-Bench 2.1 固定 20 题子集：通过 12/20，"
        "failed: 5，ERROR: 3；模型 `gpt-5.5`，推理强度 `xhigh`，"
        "Harbor 版本 `0.20.0`。",
    )

    summary = validate_release_claims(root)

    assert calls == [evidence_dir]
    assert summary.terminal_bench == terminal


def test_terminal_schema_three_evidence_cannot_support_a_public_claim(
    tmp_path,
    monkeypatch,
):
    import lha.release_claims as claims

    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    (root / "benchmarks/terminal_bench_2_1").mkdir()
    legacy = _terminal_validation().model_copy(
        update={
            "evaluated_commit_sha": None,
            "evaluated_tree_sha": None,
            "evaluated_wheel_filename": None,
            "evaluated_wheel_size_bytes": None,
            "evaluated_wheel_sha256": None,
        }
    )
    monkeypatch.setattr(
        claims,
        "validate_terminal_bench_public_evidence",
        lambda _path: legacy,
    )

    with pytest.raises(ReleaseClaimsError, match="schema 4"):
        validate_release_claims(root)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("12/20", "11/20", "counts differ"),
        ("gpt-5.5", "gpt-5.4", "model differs"),
        ("xhigh", "high", "reasoning effort differs"),
        ("0.20.0", "0.19.0", "Harbor version differs"),
    ],
)
def test_terminal_readme_drift_is_rejected(
    tmp_path,
    monkeypatch,
    original,
    replacement,
    message,
):
    import lha.release_claims as claims

    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    (root / "benchmarks/terminal_bench_2_1").mkdir()
    monkeypatch.setattr(
        claims,
        "validate_terminal_bench_public_evidence",
        lambda _path: _terminal_validation(),
    )
    claim = (
        "Terminal-Bench 2.1 固定 20 题子集：通过 12/20，"
        "failed: 5，ERROR: 3；模型 `gpt-5.5`，推理强度 `xhigh`，"
        "Harbor 版本 `0.20.0`。"
    ).replace(original, replacement)
    _append_terminal_claim(root, claim)

    with pytest.raises(ReleaseClaimsError, match=message):
        validate_release_claims(root)


def test_terminal_numeric_claim_without_public_evidence_is_rejected(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    _append_terminal_claim(
        root,
        "Terminal-Bench 2.1 固定 20 题子集：通过 12/20，failed: 5，ERROR: 3。",
    )

    with pytest.raises(ReleaseClaimsError, match="without committed public evidence"):
        validate_release_claims(root)


def test_legacy_snapshot_requires_an_explicit_readme_marker(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_legacy_report(root, monkeypatch)
    readme = root / "README.md"
    readme.write_text(readme.read_text().replace("历史报告", "消融报告"))

    with pytest.raises(ReleaseClaimsError, match="历史报告"):
        validate_release_claims(root)


def test_legacy_snapshot_requires_markers_on_generated_reports(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_legacy_report(root, monkeypatch)
    report = root / "benchmarks/ablation_report.md"
    report.write_text(report.read_text().replace(" — legacy snapshot", "", 1))

    with pytest.raises(ReleaseClaimsError, match="legacy snapshot"):
        validate_release_claims(root)


def test_readme_numeric_drift_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_legacy_report(root, monkeypatch)
    readme = root / "README.md"
    readme.write_text(readme.read_text().replace("共 4 组相同首轮补丁", "共 3 组相同首轮补丁"))

    with pytest.raises(ReleaseClaimsError, match="task/repetition/cell"):
        validate_release_claims(root)


def test_ablation_markdown_numeric_drift_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_legacy_report(root, monkeypatch)
    report = root / "benchmarks/ablation_report.md"
    report.write_text(report.read_text().replace("tasks: 2", "tasks: 3", 1))

    with pytest.raises(ReleaseClaimsError, match="header differs"):
        validate_release_claims(root)


def test_horizon_json_must_be_reproducible_from_ablation_cells(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_legacy_report(root, monkeypatch)
    report = root / "benchmarks/horizon_report.json"
    raw = json.loads(report.read_text())
    raw["model"] = "different-model"
    report.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="cannot be reproduced"):
        validate_release_claims(root)


def test_horizon_markdown_drift_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_legacy_report(root, monkeypatch)
    report = root / "benchmarks/horizon_report.md"
    report.write_text(report.read_text().replace("p = 1.0000", "p = 0.4000", 1))

    with pytest.raises(ReleaseClaimsError, match="horizon Markdown"):
        validate_release_claims(root)


def _write_formal_report(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import lha.release_claims as claims

    tasks = ["task_one", "task_two"]
    reps = 2
    monkeypatch.setattr(claims, "_FORMAL_TASK_COUNT", len(tasks))
    monkeypatch.setattr(claims, "_FORMAL_REPETITIONS", reps)
    monkeypatch.setattr(claims, "_expected_formal_tasks", lambda _root: tasks)
    monkeypatch.setattr(
        claims,
        "_validate_formal_manifest_provenance",
        lambda _raw, _tasks, _root: None,
    )
    monkeypatch.setattr(
        claims,
        "_validate_docker_executable_provenance",
        lambda _provenance: None,
    )
    source_root = root / "src" / "lha"
    source_root.mkdir(parents=True)
    (source_root / "sentinel.py").write_text("VALUE = 1\n")
    source_files = _source_file_digests(source_root)
    source_tree_digest = _source_tree_digest(source_files)

    task_paths: dict[str, str] = {}
    corpus_paths: dict[str, str] = {}
    task_digests: dict[str, str] = {}
    corpus_digests: dict[str, str] = {}
    snapshot_digests: dict[str, str] = {}
    for task in tasks:
        corpus = root / "data" / "bench" / task
        corpus.mkdir(parents=True)
        (corpus / "m.py").write_text(f"TASK = {task!r}\n")
        task_path = root / "data" / "tasks" / f"{task}.yaml"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text(f"kind: issue_to_pr\ntitle: {task}\ntarget_repo: data/bench/{task}\n")
        task_paths[task] = f"data/tasks/{task}.yaml"
        corpus_paths[task] = f"data/bench/{task}"
        task_digests[task] = hashlib.sha256(task_path.read_bytes()).hexdigest()
        corpus_digests[task] = _repo_digest(corpus)
        snapshot_digests[task] = _input_snapshot_digest(
            task_digests[task],
            corpus_digests[task],
        )

    scorer_image_id = "sha256:" + "d" * 64
    evidence_payloads: dict[str, bytes] = {}
    artifact_payloads: dict[str, bytes] = {}

    def scorer_evidence(
        correct: bool,
        *,
        task: str,
        rep: int,
        artifact_sha256: str,
    ) -> str:
        nodeid = "tests/test_m.py::test_f"
        returncode = 0 if correct else 1
        nonce = hashlib.sha256(f"{task}:{rep}:{artifact_sha256}:{correct}".encode()).hexdigest()[
            :48
        ]
        receipt = {
            "schema_version": 1,
            "nonce": nonce,
            "mode": "run",
            "pytest_exit_code": returncode,
            "collected": [nodeid],
            "collection_failures": 0,
            "reports": [
                {
                    "nodeid": nodeid,
                    "when": "call",
                    "outcome": "passed" if correct else "failed",
                    "wasxfail": False,
                }
            ],
        }
        receipt_payload = _scorer_evidence_bytes(receipt)
        pytest_evidence = {
            "schema_version": 1,
            "expected_nodeids": [nodeid],
            "process_returncode": returncode,
            "receipt": receipt,
            "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
            "classification": "PASS" if correct else "TEST_FAIL",
        }
        evidence = {
            "schema_version": 2,
            "binding": {
                "task": task,
                "rep": rep,
                "artifact_sha256": artifact_sha256,
                "input_snapshot_sha256": snapshot_digests[task],
                "scorer_backend": "docker",
                "scorer_image_id": scorer_image_id,
            },
            "pytest_evidence": pytest_evidence,
        }
        payload = _scorer_evidence_bytes(evidence)
        evidence_digest = hashlib.sha256(payload).hexdigest()
        evidence_payloads[evidence_digest] = payload
        return evidence_digest

    records: list[RunRecord] = []
    for task in tasks:
        for rep in range(reps):
            artifact_payload = _frozen_artifact_bytes({"m.py": f"TASK = {task!r}\nREP = {rep}\n"})
            digest = hashlib.sha256(artifact_payload).hexdigest()
            artifact_payloads[digest] = artifact_payload
            first_attempt_correct = not (task == "task_two" and rep == 0)
            gate_accepts = first_attempt_correct and not (task == "task_one" and rep == 0)
            first_evidence = scorer_evidence(
                first_attempt_correct,
                task=task,
                rep=rep,
                artifact_sha256=digest,
            )
            pass_evidence_digest = scorer_evidence(
                True,
                task=task,
                rep=rep,
                artifact_sha256=digest,
            )
            records.extend(
                [
                    RunRecord(
                        task=task,
                        condition="trust",
                        rep=rep,
                        status="DONE",
                        claimed_success=True,
                        artifact_correct=first_attempt_correct,
                        true_success=first_attempt_correct,
                        false_success=not first_attempt_correct,
                        repairs=0,
                        artifact_sha256=digest,
                        scorer_outcome="PASS" if first_attempt_correct else "TEST_FAIL",
                        scorer_evidence_sha256=first_evidence,
                        scorer_expected_tests=1,
                        scorer_passed_tests=1 if first_attempt_correct else 0,
                    ),
                    RunRecord(
                        task=task,
                        condition="gate",
                        rep=rep,
                        status="DONE" if gate_accepts else "FAILED",
                        claimed_success=gate_accepts,
                        artifact_correct=first_attempt_correct,
                        true_success=gate_accepts and first_attempt_correct,
                        false_success=False,
                        repairs=0,
                        gate_prediction=gate_accepts,
                        artifact_sha256=digest,
                        scorer_outcome="PASS" if first_attempt_correct else "TEST_FAIL",
                        scorer_evidence_sha256=first_evidence,
                        scorer_expected_tests=1,
                        scorer_passed_tests=1 if first_attempt_correct else 0,
                    ),
                    RunRecord(
                        task=task,
                        condition="verify",
                        rep=rep,
                        status="DONE",
                        claimed_success=True,
                        artifact_correct=True,
                        true_success=True,
                        false_success=False,
                        repairs=0 if first_attempt_correct else 1,
                        gate_prediction=True,
                        artifact_sha256=digest,
                        scorer_outcome="PASS",
                        scorer_evidence_sha256=pass_evidence_digest,
                        scorer_expected_tests=1,
                        scorer_passed_tests=1,
                    ),
                ]
            )

    provenance = AblationProvenance(
        generated_at="2026-07-27T00:00:00+00:00",
        harness_version="0.5.0.dev0",
        git_commit="a" * 40,
        git_dirty=False,
        source_tree_sha256=source_tree_digest,
        source_files=source_files,
        requested_llm_backend="codex_cli",
        actual_llm_backend="codex_cli",
        model="model-x",
        cli_version="codex-cli 1.0",
        reasoning_effort="low",
        agent_backend="docker",
        scorer_requested="docker",
        scorer_backend="docker",
        scorer_image="lha:release",
        scorer_image_id=scorer_image_id,
        git_executable={
            "path": "/usr/bin/git",
            "sha256": "7" * 64,
            "size_bytes": 123,
            "trusted_install": True,
        },
        docker_executable={
            "path": "/usr/bin/docker",
            "sha256": "8" * 64,
            "size_bytes": 456,
            "trusted_install": True,
        },
        platform="test-platform",
        python_version="3.11.9",
        pytest_version="9.1.1",
        task_paths=task_paths,
        corpus_paths=corpus_paths,
        task_files_sha256=task_digests,
        corpus_sha256=corpus_digests,
        input_snapshot_sha256=snapshot_digests,
        cell_fingerprints={
            task: hashlib.sha256(f"cell:{task}".encode()).hexdigest() for task in tasks
        },
        configuration={
            "repetitions": reps,
            "task_count": len(tasks),
            "conditions": ["trust", "gate", "verify"],
            "max_repairs": 3,
            "llm_retries": 3,
            "cache_schema": 8,
            "report_schema": 4,
            "frozen_artifact_schema": 1,
            "input_snapshot_schema": 1,
            "scorer_evidence_schema": 2,
            "llm_call_receipt_schema": 1,
            "cell_attempt_schema": 1,
            "codex_operation_lease_store": ".",
            "docker_operation_lease_store": ".",
            "docker_container_absence_filter": "label=lha.operation_id",
            "docker_operations_recovered_before_run": 0,
            "docker_operations_recovered_at_completion": 0,
            "run_control_executables": {
                "git": {
                    "path": "/usr/bin/git",
                    "sha256": "7" * 64,
                    "size_bytes": 123,
                    "trusted_install": True,
                },
                "docker": {
                    "path": "/usr/bin/docker",
                    "sha256": "8" * 64,
                    "size_bytes": 456,
                    "trusted_install": True,
                },
            },
            "scorer_result_source": "nonce-bound-pytest-hook-receipt",
            "docker_image_probe": {
                "schema_version": 1,
                "image_id": scorer_image_id,
                "network": "none",
                "minimal_pytest": "passed",
                "python_version": "3.11.9",
                "pytest_version": "9.1.1",
                "pytest_json_report_version": "1.5.0",
            },
            "client": {
                "no_tools": True,
                "sandbox_mode": "read-only",
                "permission_model": "profile",
                "permission_profile": "lha-read",
                "credential_barrier": "verified",
                "externally_sandboxed": False,
                "max_retries": 2,
            },
        },
    )
    report = AblationReport(
        llm="codex_cli",
        model="model-x",
        reps=reps,
        tasks=tasks,
        records=records,
        stats=_aggregate(records),
        scorer="docker",
        fingerprint="c" * 64,
        backend_version="codex-cli 1.0",
        provenance=provenance,
    )
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True)
    artifacts = benchmarks / "artifacts"
    artifacts.mkdir()
    for digest, artifact_payload in artifact_payloads.items():
        (artifacts / f"{digest}.json").write_bytes(artifact_payload)
    scorer_evidence_dir = benchmarks / "scorer_evidence"
    scorer_evidence_dir.mkdir()
    for evidence_digest, evidence_payload in evidence_payloads.items():
        (scorer_evidence_dir / f"{evidence_digest}.json").write_bytes(evidence_payload)
    event_summary = {
        "total_events": 4,
        "events": {
            "thread.started": 1,
            "turn.started": 1,
            "item.completed": 1,
            "turn.completed": 1,
        },
        "items": {"agent_message": 1},
        "invalid_json_lines": 0,
    }
    receipt_dir = benchmarks / "llm_call_receipts"
    receipt_dir.mkdir()
    llm_calls = []
    for task in tasks:
        for rep in range(reps):
            artifact_sha256 = next(
                record.artifact_sha256
                for record in records
                if record.task == task and record.rep == rep and record.condition == "trust"
            )
            base_call = {
                "status": "succeeded",
                "backend": "codex_cli",
                "cli_version": "codex-cli 1.0",
                "model": "model-x",
                "reasoning_effort": "low",
                "sandbox_mode": "read-only",
                "permission_model": "profile",
                "permission_profile": "lha-read",
                "credential_barrier": "verified",
                "cli_executable_sha256": "9" * 64,
                "cli_executable_trusted": False,
                "externally_sandboxed": False,
                "retries": 0,
                "attempt_count": 1,
                "duration_s": 0.25,
                "event_summary": event_summary,
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "succeeded",
                        "duration_s": 0.25,
                        "event_summary": event_summary,
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "output_tokens": 5,
                    "cost_usd": None,
                    "model": "model-x",
                },
            }
            repair_count = next(
                record.repairs
                for record in records
                if record.task == task and record.rep == rep and record.condition == "verify"
            )
            for ordinal, label in enumerate(["first", *(["repair"] * repair_count)]):
                receipt = {
                    "schema_version": 1,
                    "binding": {
                        "task": task,
                        "rep": rep,
                        "label": label,
                        "ordinal": ordinal,
                        "cell_fingerprint": provenance.cell_fingerprints[task],
                        "input_snapshot_sha256": snapshot_digests[task],
                        "prompt_sha256": hashlib.sha256(
                            f"prompt:{task}:{rep}:{ordinal}".encode()
                        ).hexdigest(),
                        "response_sha256": hashlib.sha256(
                            f"response:{task}:{rep}:{ordinal}".encode()
                        ).hexdigest(),
                        "patch_sha256": hashlib.sha256(
                            f"patch:{task}:{rep}:{ordinal}".encode()
                        ).hexdigest(),
                        "result_artifact_sha256": artifact_sha256,
                    },
                    "call": base_call,
                }
                payload = _llm_call_receipt_bytes(receipt)
                digest = hashlib.sha256(payload).hexdigest()
                (receipt_dir / f"{digest}.json").write_bytes(payload)
                llm_calls.append(
                    {
                        "task": task,
                        "rep": rep,
                        "ordinal": ordinal,
                        "receipt_sha256": digest,
                        "cache_hit": False,
                    }
                )
    report.llm_calls = llm_calls

    ablation_json = {
        "schema_version": report.schema_version,
        "llm": report.llm,
        "model": report.model,
        "reps": report.reps,
        "tasks": report.tasks,
        "scorer": report.scorer,
        "fingerprint": report.fingerprint,
        "backend_version": report.backend_version,
        "harness_version": "0.5.0.dev0",
        "provenance": asdict(provenance),
        "artifact_store": {
            "schema_version": 1,
            "path": "artifacts",
            "encoding": "canonical-json",
            "count": len(artifact_payloads),
        },
        "scorer_evidence_store": {
            "schema_version": 2,
            "path": "scorer_evidence",
            "encoding": "canonical-json",
            "count": len(evidence_payloads),
        },
        "llm_call_receipt_store": {
            "schema_version": 1,
            "path": "llm_call_receipts",
            "encoding": "canonical-json",
            "count": len(llm_calls),
        },
        "llm_calls": llm_calls,
        "stats": [asdict(stat) for stat in report.stats],
        "records": [asdict(record) for record in records],
    }
    report.fingerprint = _report_fingerprint(ablation_json)
    ablation_json["fingerprint"] = report.fingerprint
    (benchmarks / "ablation_report.json").write_text(json.dumps(ablation_json, indent=2))
    (benchmarks / "ablation_report.md").write_text(report.to_markdown())

    monkeypatch.chdir(root)
    run_horizon("benchmarks/ablation_report.json", "benchmarks")
    (root / "README.md").write_text(
        """# Project

## 已提交的实测结果

仓库中的消融报告是正式报告，使用 2 个预设 Python 缺陷，每个任务重复 2 次，共 4 组相同首轮补丁。
计划执行 4 组，其中 4 组结果可用，ERROR 为 0 组；下表比例以 4 组可用结果为分母。
实测模型为 `model-x`。

| 条件 | 处理方式 | 独立评分 |
|---|---|---|
| `trust` | 直接接受首轮补丁 | 3 个正确，1 个错误仍被接受 |
| `gate` | 首轮补丁必须通过测试 | 接受 2 个正确补丁，拦截 1 个错误补丁 |
| `verify` | 失败后允许修复 | 4/4 通过独立评分 |

精确 McNemar 检验为 `p = 1.00`。
Horizon 曲线是描述性组合，不增加样本。

## 其他
"""
    )


def _write_legacy_report(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Downgrade the synthetic report without making legacy look formal."""
    _write_formal_report(root, monkeypatch)
    benchmarks = root / "benchmarks"
    report_path = benchmarks / "ablation_report.json"
    raw = json.loads(report_path.read_text())
    raw.pop("schema_version")
    raw.pop("provenance")
    raw.pop("scorer_evidence_store")
    for record in raw["records"]:
        record.pop("artifact_correct")
        record.pop("scorer_evidence_sha256")
        record.pop("scorer_expected_tests")
        record.pop("scorer_passed_tests")
    report_path.write_text(json.dumps(raw, indent=2))
    historical_records = load_ablation_report(report_path).records
    raw["stats"] = [asdict(stat) for stat in _aggregate(historical_records)]
    for stat in raw["stats"]:
        stat.pop("artifact_correct_rate")
        stat.pop("artifact_ci")
        if stat["true_success_rate"] in (0.0, 1.0):
            stat["true_ci"] = [stat["true_success_rate"], stat["true_success_rate"]]
        if stat["false_success_rate"] in (0.0, 1.0):
            stat["false_ci"] = [stat["false_success_rate"], stat["false_success_rate"]]
    report_path.write_text(json.dumps(raw, indent=2))
    legacy = load_ablation_report(report_path)
    (benchmarks / "ablation_report.md").write_text(
        legacy.to_markdown().replace(
            "# Verification ablation",
            "# Verification ablation — legacy snapshot",
            1,
        )
    )
    run_horizon("benchmarks/ablation_report.json", "benchmarks")
    horizon_md = benchmarks / "horizon_report.md"
    horizon_md.write_text(
        horizon_md.read_text().replace(
            "# Error compounding over a horizon",
            "# Error compounding over a horizon — legacy snapshot",
            1,
        )
    )
    readme = root / "README.md"
    readme.write_text(
        readme.read_text()
        .replace("正式报告", "历史报告")
        .replace(
            "## 其他",
            "204 次正式复测只有在运行并提交原始记录后才会更新。\n\n## 其他",
        )
    )


def test_schema_four_reports_are_checked_as_formal_evidence(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)

    summary = validate_release_claims(root)
    assert summary.status == "formal"
    assert summary.tasks == 2
    assert summary.repetitions == 2
    assert summary.scheduled_cells == 4
    assert summary.usable_cells == 4
    assert summary.error_cells == 0
    assert summary.cells == 4
    assert summary.trust_false_successes == 1
    assert summary.gate_successes == 2
    # One additional correct artifact is rejected. It is a false negative, not
    # an interception of an incorrect artifact.
    assert summary.gate_interceptions == 1
    assert summary.verify_successes == 4


def test_formal_report_accepts_receipt_backed_terminal_error_cell(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    _write_terminal_error_cell(root)

    summary = validate_release_claims(root)

    assert summary.scheduled_cells == 4
    assert summary.usable_cells == 3
    assert summary.error_cells == 1
    assert summary.cells == 3
    assert summary.verify_successes == 3


def test_formal_report_accepts_cached_terminal_error_seal(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    _write_terminal_error_cell(root, cache_hit=True)

    summary = validate_release_claims(root)

    assert summary.error_cells == 1


def test_formal_report_rejects_error_cell_without_receipt(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _write_terminal_error_cell(root)
    raw["llm_calls"] = [
        reference
        for reference in raw["llm_calls"]
        if not (reference["task"] == "task_two" and reference["rep"] == 1)
    ]
    raw["llm_call_receipt_store"]["count"] = len(raw["llm_calls"])
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="has no LLM call receipt"):
        validate_release_claims(root)


def test_formal_report_rejects_error_cell_with_successful_terminal_call(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _write_terminal_error_cell(root)
    error_reference = next(
        reference
        for reference in raw["llm_calls"]
        if reference["task"] == "task_two" and reference["rep"] == 1
    )
    success_reference = next(
        reference
        for reference in raw["llm_calls"]
        if reference["task"] == "task_one" and reference["rep"] == 0
    )
    error_path = _receipt_path(root, error_reference)
    error_receipt = json.loads(error_path.read_text())
    success_receipt = json.loads(_receipt_path(root, success_reference).read_text())
    for field in ("response_sha256", "patch_sha256", "result_artifact_sha256"):
        error_receipt["binding"][field] = success_receipt["binding"][field]
    error_receipt["call"] = success_receipt["call"]
    payload = _llm_call_receipt_bytes(error_receipt)
    digest = hashlib.sha256(payload).hexdigest()
    error_path.unlink()
    (root / "benchmarks" / "llm_call_receipts" / f"{digest}.json").write_bytes(payload)
    error_reference["receipt_sha256"] = digest
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="only failed first-call"):
        validate_release_claims(root)


def test_formal_report_rejects_condition_local_error(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _write_terminal_error_cell(root)
    donor = next(
        record
        for record in raw["records"]
        if record["task"] == "task_one"
        and record["rep"] == 1
        and record["condition"] == "trust"
    )
    mixed = next(
        record
        for record in raw["records"]
        if record["task"] == "task_two"
        and record["rep"] == 1
        and record["condition"] == "trust"
    )
    mixed.update(
        {
            **donor,
            "task": "task_two",
            "rep": 1,
            "condition": "trust",
        }
    )
    raw["stats"] = [
        asdict(stat) for stat in _aggregate([RunRecord(**record) for record in raw["records"]])
    ]
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="cover trust, gate, and verify"):
        validate_release_claims(root)


@pytest.mark.parametrize("store_name", ["artifact_store", "scorer_evidence_store"])
def test_formal_error_cell_must_be_excluded_from_measurement_stores(
    tmp_path,
    monkeypatch,
    store_name,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _write_terminal_error_cell(root)
    raw[store_name]["count"] += 1
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="artifact_store|scorer_evidence_store"):
        validate_release_claims(root)


def test_formal_readme_rates_must_use_usable_cell_denominator(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    _write_terminal_error_cell(root)
    readme = root / "README.md"
    readme.write_text(
        readme.read_text().replace(
            "下表比例以 3 组可用结果为分母",
            "下表比例以 4 组可用结果为分母",
        )
    )

    with pytest.raises(ReleaseClaimsError, match="rate denominator"):
        validate_release_claims(root)


def test_formal_error_cell_rejects_mixed_cache_provenance(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _write_terminal_error_cell(root, failed_calls=2)
    references = [
        reference
        for reference in raw["llm_calls"]
        if reference["task"] == "task_two" and reference["rep"] == 1
    ]
    references[0]["cache_hit"] = True
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="mixes cache provenance"):
        validate_release_claims(root)


def test_formal_cache_hits_must_form_scheduled_prefix(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    for reference in raw["llm_calls"]:
        if reference["task"] == "task_two" and reference["rep"] == 0:
            reference["cache_hit"] = True
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="scheduled prefix"):
        validate_release_claims(root)


def test_formal_call_receipt_store_rejects_unreferenced_json(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    _path, raw = _formal_raw(root)
    receipt = _receipt_path(root, raw["llm_calls"][0])
    extra = root / "benchmarks" / "llm_call_receipts" / f"{'0' * 64}.json"
    extra.write_bytes(receipt.read_bytes())

    with pytest.raises(ReleaseClaimsError, match="unreferenced or missing entries"):
        validate_release_claims(root)


def _formal_raw(root: Path) -> tuple[Path, dict]:
    path = root / "benchmarks/ablation_report.json"
    return path, json.loads(path.read_text())


def _write_terminal_error_cell(
    root: Path,
    *,
    task: str = "task_two",
    rep: int = 1,
    failed_calls: int = 1,
    cache_hit: bool = False,
) -> tuple[Path, dict]:
    """Replace one measured cell with a receipt-backed terminal ERROR."""
    path, raw = _formal_raw(root)
    cell_records = [
        record for record in raw["records"] if record["task"] == task and record["rep"] == rep
    ]
    assert len(cell_records) == 3
    for record in cell_records:
        record.update(
            {
                "status": "ERROR",
                "claimed_success": False,
                "artifact_correct": False,
                "true_success": False,
                "false_success": False,
                "repairs": 0,
                "detail": "terminal Codex failure",
                "gate_prediction": None,
                "artifact_sha256": "",
                "scorer_outcome": "INFRA_ERROR",
                "scorer_evidence_sha256": "",
                "scorer_expected_tests": 0,
                "scorer_passed_tests": 0,
            }
        )

    original_references = [
        reference
        for reference in raw["llm_calls"]
        if reference["task"] == task and reference["rep"] == rep
    ]
    assert original_references
    template = json.loads(_receipt_path(root, original_references[0]).read_text())
    for reference in original_references:
        _receipt_path(root, reference).unlink()
    raw["llm_calls"] = [
        reference
        for reference in raw["llm_calls"]
        if not (reference["task"] == task and reference["rep"] == rep)
    ]
    if cache_hit:
        for reference in raw["llm_calls"]:
            reference["cache_hit"] = True
    empty_summary = {
        "total_events": 0,
        "events": {},
        "items": {},
        "invalid_json_lines": 0,
    }
    for ordinal in range(failed_calls):
        receipt = json.loads(json.dumps(template))
        receipt["binding"].update(
            {
                "label": "first",
                "ordinal": ordinal,
                "prompt_sha256": hashlib.sha256(
                    f"terminal:{task}:{rep}:{ordinal}".encode()
                ).hexdigest(),
                "response_sha256": None,
                "patch_sha256": None,
                "result_artifact_sha256": None,
            }
        )
        receipt["call"].update(
            {
                "status": "failed",
                "retries": 0,
                "attempt_count": 1,
                "duration_s": 0.1,
                "event_summary": empty_summary,
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "failed",
                        "duration_s": 0.1,
                        "error_type": "CodexProtocolError",
                        "event_summary": empty_summary,
                    }
                ],
                "usage": {
                    "input_tokens": None,
                    "cached_input_tokens": None,
                    "output_tokens": None,
                    "cost_usd": None,
                    "model": "model-x",
                },
                "error_type": "CodexProtocolError",
                "retryable": False,
            }
        )
        payload = _llm_call_receipt_bytes(receipt)
        digest = hashlib.sha256(payload).hexdigest()
        (root / "benchmarks" / "llm_call_receipts" / f"{digest}.json").write_bytes(payload)
        raw["llm_calls"].append(
            {
                "task": task,
                "rep": rep,
                "ordinal": ordinal,
                "receipt_sha256": digest,
                "cache_hit": cache_hit,
            }
        )

    records = [RunRecord(**record) for record in raw["records"]]
    raw["stats"] = [asdict(stat) for stat in _aggregate(records)]
    raw["artifact_store"]["count"] = len(
        {record.artifact_sha256 for record in records if record.status != "ERROR"}
    )
    raw["scorer_evidence_store"]["count"] = len(
        {record.scorer_evidence_sha256 for record in records if record.status != "ERROR"}
    )
    raw["llm_call_receipt_store"]["count"] = len(raw["llm_calls"])
    raw["fingerprint"] = _report_fingerprint(raw)
    path.write_text(json.dumps(raw, indent=2))
    (root / "benchmarks" / "ablation_report.md").write_text(
        load_ablation_report(path).to_markdown()
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        run_horizon("benchmarks/ablation_report.json", "benchmarks")
    finally:
        os.chdir(previous_cwd)
    readme = root / "README.md"
    readme.write_text(
        readme.read_text()
        .replace(
            "计划执行 4 组，其中 4 组结果可用，ERROR 为 0 组；"
            "下表比例以 4 组可用结果为分母。",
            "计划执行 4 组，其中 3 组结果可用，ERROR 为 1 组；"
            "下表比例以 3 组可用结果为分母。",
        )
        .replace("3 个正确，1 个错误仍被接受", "2 个正确，1 个错误仍被接受")
        .replace("接受 2 个正确补丁，拦截 1 个错误补丁", "接受 1 个正确补丁，拦截 1 个错误补丁")
        .replace("4/4 通过独立评分", "3/3 通过独立评分")
    )
    return path, raw


def _receipt_path(root: Path, reference: dict) -> Path:
    return root / "benchmarks" / "llm_call_receipts" / f"{reference['receipt_sha256']}.json"


def _install_outer_retry_chain(
    root: Path,
    *,
    failed_calls: int,
) -> tuple[Path, dict, list[dict]]:
    path, raw = _formal_raw(root)
    original_reference = raw["llm_calls"][0]
    successful = json.loads(_receipt_path(root, original_reference).read_text())
    cell_references: list[dict] = []
    empty_summary = {
        "total_events": 0,
        "events": {},
        "items": {},
        "invalid_json_lines": 0,
    }
    for ordinal in range(failed_calls):
        failed = json.loads(json.dumps(successful))
        failed["binding"].update(
            {
                "ordinal": ordinal,
                "prompt_sha256": hashlib.sha256(f"failed:{ordinal}".encode()).hexdigest(),
                "response_sha256": None,
                "patch_sha256": None,
                "result_artifact_sha256": None,
            }
        )
        failed["call"].update(
            {
                "status": "failed",
                "retries": 0,
                "attempt_count": 1,
                "event_summary": empty_summary,
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "failed",
                        "duration_s": 0.1,
                        "error_type": "CodexTransientError",
                        "event_summary": empty_summary,
                    }
                ],
                "usage": {
                    "input_tokens": None,
                    "cached_input_tokens": None,
                    "output_tokens": None,
                    "cost_usd": None,
                    "model": "model-x",
                },
                "error_type": "CodexTransientError",
                "retryable": True,
            }
        )
        payload = _llm_call_receipt_bytes(failed)
        digest = hashlib.sha256(payload).hexdigest()
        (root / "benchmarks" / "llm_call_receipts" / f"{digest}.json").write_bytes(payload)
        cell_references.append(
            {
                **original_reference,
                "ordinal": ordinal,
                "receipt_sha256": digest,
            }
        )
    successful["binding"]["ordinal"] = failed_calls
    success_payload = _llm_call_receipt_bytes(successful)
    success_digest = hashlib.sha256(success_payload).hexdigest()
    (root / "benchmarks" / "llm_call_receipts" / f"{success_digest}.json").write_bytes(
        success_payload
    )
    cell_references.append(
        {
            **original_reference,
            "ordinal": failed_calls,
            "receipt_sha256": success_digest,
        }
    )
    raw["llm_calls"] = cell_references + raw["llm_calls"][1:]
    raw["llm_call_receipt_store"]["count"] = len(raw["llm_calls"])
    path.write_text(json.dumps(raw))
    return path, raw, cell_references


def test_formal_report_rejects_call_receipts_swapped_between_cells(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    left, right = raw["llm_calls"][0], raw["llm_calls"][1]
    left["receipt_sha256"], right["receipt_sha256"] = (
        right["receipt_sha256"],
        left["receipt_sha256"],
    )
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="another cell"):
        validate_release_claims(root)


def test_formal_report_rejects_duplicate_call_receipt_reference(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    raw["llm_calls"][1]["receipt_sha256"] = raw["llm_calls"][0]["receipt_sha256"]
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="reuses one"):
        validate_release_claims(root)


def test_formal_report_rejects_tampered_call_receipt_bytes(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    _path, raw = _formal_raw(root)
    receipt = _receipt_path(root, raw["llm_calls"][0])
    receipt.write_bytes(receipt.read_bytes() + b"\n")

    with pytest.raises(ReleaseClaimsError, match="digest"):
        validate_release_claims(root)


def test_formal_report_rejects_tool_item_hidden_in_success_summary(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    reference = raw["llm_calls"][0]
    receipt = json.loads(_receipt_path(root, reference).read_text())
    for summary in (
        receipt["call"]["event_summary"],
        receipt["call"]["attempts"][-1]["event_summary"],
    ):
        summary["items"] = {"command_execution": 1}
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (root / "benchmarks" / "llm_call_receipts" / f"{digest}.json").write_bytes(payload)
    reference["receipt_sha256"] = digest
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="no-tools"):
        validate_release_claims(root)


def test_formal_report_rejects_inconsistent_event_totals(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    reference = raw["llm_calls"][0]
    receipt = json.loads(_receipt_path(root, reference).read_text())
    receipt["call"]["event_summary"]["total_events"] = 999
    receipt["call"]["attempts"][-1]["event_summary"]["total_events"] = 999
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (root / "benchmarks" / "llm_call_receipts" / f"{digest}.json").write_bytes(payload)
    reference["receipt_sha256"] = digest
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="counts are inconsistent"):
        validate_release_claims(root)


def test_formal_report_rejects_call_exceeding_inner_retry_budget(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    reference = raw["llm_calls"][0]
    receipt = json.loads(_receipt_path(root, reference).read_text())
    empty_summary = {
        "total_events": 0,
        "events": {},
        "items": {},
        "invalid_json_lines": 0,
    }
    final_attempt = receipt["call"]["attempts"][-1]
    receipt["call"]["attempts"] = [
        {
            "attempt": attempt,
            "status": "failed",
            "duration_s": 0.1,
            "error_type": "CodexTransientError",
            "event_summary": empty_summary,
        }
        for attempt in range(1, 4)
    ] + [{**final_attempt, "attempt": 4}]
    receipt["call"]["attempt_count"] = 4
    receipt["call"]["retries"] = 3
    payload = _llm_call_receipt_bytes(receipt)
    digest = hashlib.sha256(payload).hexdigest()
    (root / "benchmarks" / "llm_call_receipts" / f"{digest}.json").write_bytes(payload)
    reference["receipt_sha256"] = digest
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="inner retry budget"):
        validate_release_claims(root)


def test_formal_report_rejects_call_exceeding_outer_retry_budget(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    _install_outer_retry_chain(root, failed_calls=3)

    with pytest.raises(ReleaseClaimsError, match="first-call retry sequence"):
        validate_release_claims(root)


def test_formal_report_rejects_deleted_failed_retry_receipt(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw, cell_references = _install_outer_retry_chain(
        root,
        failed_calls=1,
    )
    raw["llm_calls"].remove(cell_references[0])
    raw["llm_call_receipt_store"]["count"] = len(raw["llm_calls"])
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="ordinals"):
        validate_release_claims(root)


def test_formal_report_fingerprint_is_recomputed(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    raw["provenance"]["generated_at"] = "2026-07-28T00:00:00+00:00"
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="fingerprint"):
        validate_release_claims(root)


def test_formal_report_rejects_operation_lease_residue(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    lease_dir = root / "benchmarks" / "active-operations"
    lease_dir.mkdir()
    (lease_dir / "residue.json").write_text("{}")

    with pytest.raises(ReleaseClaimsError, match="operation store is not empty"):
        validate_release_claims(root)


def test_formal_report_requires_codex_operation_lease_binding(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    raw["provenance"]["configuration"]["codex_operation_lease_store"] = None
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="operation-lease attestation"):
        validate_release_claims(root)


def test_formal_report_binds_docker_probe_to_pinned_image(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    raw["provenance"]["configuration"]["docker_image_probe"]["image_id"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="capability probe"):
        validate_release_claims(root)


def test_formal_report_rejects_unbound_control_executable_provenance(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path, raw = _formal_raw(root)
    raw["provenance"]["configuration"]["run_control_executables"]["docker"]["sha256"] = "0" * 64
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="not internally bound"):
        validate_release_claims(root)


def test_formal_docker_executable_accepts_operator_owned_desktop_install():
    import lha.release_claims as claims

    recorded = {
        "path": "/fixed/docker",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "trusted_install": False,
    }

    claims._validate_docker_executable_provenance({"docker_executable": recorded})


def test_formal_docker_executable_requires_explicit_ownership_provenance():
    import lha.release_claims as claims

    recorded = {
        "path": "/fixed/docker",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "trusted_install": "unknown",
    }

    with pytest.raises(ReleaseClaimsError, match="invalid docker_executable"):
        claims._validate_docker_executable_provenance({"docker_executable": recorded})


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _formal_manifest_git_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    import lha.ablation as abl
    import lha.release_claims as claims

    source = root / "src" / "lha"
    source.mkdir(parents=True)
    (source / "sentinel.py").write_text("VALUE = 1\n")
    corpus = root / "data" / "bench" / "one"
    corpus.mkdir(parents=True)
    (corpus / "m.py").write_text("VALUE = 1\n")
    task = root / "data" / "tasks" / "bench_one.yaml"
    task.parent.mkdir(parents=True)
    task.write_text("kind: issue_to_pr\ntitle: one\ntarget_repo: data/bench/one\n")
    _git(root, "init")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fix corpus")
    corpus_commit = _git(root, "rev-parse", "HEAD")

    manifest_relative = Path("data/bench/formal_manifest.json")
    manifest_path = root / manifest_relative
    manifest = {
        "schema_version": 1,
        "benchmark": "lha-verification-ablation",
        "repetitions": 1,
        "corpus_commit": corpus_commit,
        "tasks": [
            {
                "name": "bench_one",
                "task_path": "data/tasks/bench_one.yaml",
                "task_sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
                "corpus_path": "data/bench/one",
                "corpus_sha256": _repo_digest(corpus),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _git(root, "add", ".")
    _git(root, "commit", "-m", "preregister corpus")
    evaluated_commit = _git(root, "rev-parse", "HEAD")

    monkeypatch.setattr(abl, "_FORMAL_TASK_COUNT", 1)
    monkeypatch.setattr(abl, "_FORMAL_REPETITIONS", 1)
    monkeypatch.setattr(
        abl,
        "_FORMAL_CORPUS_MANIFEST_PATH",
        manifest_relative,
    )
    monkeypatch.setattr(
        claims,
        "_FORMAL_CORPUS_MANIFEST_PATH",
        manifest_relative,
    )
    return {
        "provenance": {
            "formal_corpus_manifest_path": manifest_relative.as_posix(),
            "formal_corpus_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "preregistration_commit": evaluated_commit,
            "git_commit": evaluated_commit,
            "git_executable": claims._trusted_control_executable("git"),
        }
    }


def test_formal_manifest_git_binding_accepts_preregistered_clean_tree(
    tmp_path,
    monkeypatch,
):
    import lha.release_claims as claims

    root = tmp_path / "repo"
    root.mkdir()
    raw = _formal_manifest_git_fixture(root, monkeypatch)

    claims._validate_formal_manifest_provenance(
        raw,
        ["bench_one"],
        root,
    )


def test_formal_manifest_git_binding_rejects_bogus_commit(
    tmp_path,
    monkeypatch,
):
    import lha.release_claims as claims

    root = tmp_path / "repo"
    root.mkdir()
    raw = _formal_manifest_git_fixture(root, monkeypatch)
    raw["provenance"]["git_commit"] = "f" * 40

    with pytest.raises(ReleaseClaimsError, match="evaluated commit"):
        claims._validate_formal_manifest_provenance(
            raw,
            ["bench_one"],
            root,
        )


def test_formal_manifest_git_binding_rejects_git_digest_drift(
    tmp_path,
    monkeypatch,
):
    import lha.release_claims as claims

    root = tmp_path / "repo"
    root.mkdir()
    raw = _formal_manifest_git_fixture(root, monkeypatch)
    raw["provenance"]["git_executable"]["sha256"] = "invalid"

    with pytest.raises(ReleaseClaimsError, match="invalid git_executable"):
        claims._validate_formal_manifest_provenance(
            raw,
            ["bench_one"],
            root,
        )


def test_formal_manifest_git_binding_rejects_claimed_clean_dirty_tree(
    tmp_path,
    monkeypatch,
):
    import lha.release_claims as claims

    root = tmp_path / "repo"
    root.mkdir()
    raw = _formal_manifest_git_fixture(root, monkeypatch)
    (root / "untracked.txt").write_text("dirty\n")

    with pytest.raises(ReleaseClaimsError, match="clean Git worktree"):
        claims._validate_formal_manifest_provenance(
            raw,
            ["bench_one"],
            root,
        )


def test_formal_manifest_git_binding_rejects_replaced_task_same_count(
    tmp_path,
    monkeypatch,
):
    import lha.release_claims as claims

    root = tmp_path / "repo"
    root.mkdir()
    raw = _formal_manifest_git_fixture(root, monkeypatch)
    task = root / "data" / "tasks" / "bench_one.yaml"
    task.write_text(task.read_text() + "description: changed after output\n")

    with pytest.raises(ReleaseClaimsError, match="manifest is invalid"):
        claims._validate_formal_manifest_provenance(
            raw,
            ["bench_one"],
            root,
        )


def test_formal_report_rejects_degenerate_boundary_interval(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path = root / "benchmarks/ablation_report.json"
    raw = json.loads(path.read_text())
    verify = next(stat for stat in raw["stats"] if stat["condition"] == "verify")
    verify["true_ci"] = [1.0, 1.0]
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="Wilson"):
        validate_release_claims(root)


def test_formal_report_requires_complete_provenance(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path = root / "benchmarks/ablation_report.json"
    raw = json.loads(path.read_text())
    raw["provenance"]["cli_version"] = None
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="cli_version"):
        validate_release_claims(root)


def test_formal_report_requires_hardened_codex_backend(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path = root / "benchmarks/ablation_report.json"
    raw = json.loads(path.read_text())
    raw["llm"] = "claude_cli"
    raw["provenance"]["requested_llm_backend"] = "claude_cli"
    raw["provenance"]["actual_llm_backend"] = "claude_cli"
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="Codex CLI protocol validation"):
        validate_release_claims(root)


def test_formal_report_requires_sha256_artifact_ids(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path = root / "benchmarks/ablation_report.json"
    raw = json.loads(path.read_text())
    raw["records"][0]["artifact_sha256"] = "sha256:not-a-digest"
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="invalid artifact_sha256"):
        validate_release_claims(root)


def test_formal_report_binds_trust_and_gate_to_same_first_attempt(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path = root / "benchmarks/ablation_report.json"
    raw = json.loads(path.read_text())
    gate = next(
        record
        for record in raw["records"]
        if record["task"] == "task_one" and record["rep"] == 0 and record["condition"] == "gate"
    )
    gate["artifact_sha256"] = "d" * 64
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="same first-attempt artifact"):
        validate_release_claims(root)


def test_schema_four_artifact_bytes_are_content_addressed(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path = root / "benchmarks/ablation_report.json"
    raw = json.loads(path.read_text())
    digest = raw["records"][0]["artifact_sha256"]
    artifact = root / "benchmarks/artifacts" / f"{digest}.json"
    artifact.write_bytes(artifact.read_bytes() + b"\n")

    with pytest.raises(ReleaseClaimsError, match="digest does not match"):
        validate_release_claims(root)


def test_formal_report_is_invalid_after_scorer_source_changes(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    (root / "src/lha/sentinel.py").write_text("VALUE = 2\n")

    with pytest.raises(ReleaseClaimsError, match="source_files"):
        validate_release_claims(root)


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("data/tasks/task_one.yaml", "task digest"),
        ("data/bench/task_one/m.py", "corpus digest"),
    ],
)
def test_formal_report_binds_committed_task_and_corpus_bytes(
    tmp_path,
    monkeypatch,
    relative_path,
    message,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path = root / relative_path
    path.write_text(path.read_text() + "# changed\n")

    with pytest.raises(ReleaseClaimsError, match=message):
        validate_release_claims(root)


def test_formal_report_rejects_damaged_scorer_evidence(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    raw = json.loads((root / "benchmarks/ablation_report.json").read_text())
    digest = raw["records"][0]["scorer_evidence_sha256"]
    evidence = root / "benchmarks/scorer_evidence" / f"{digest}.json"
    evidence.write_bytes(evidence.read_bytes() + b"\n")

    with pytest.raises(ReleaseClaimsError, match="digest does not match"):
        validate_release_claims(root)


@pytest.mark.parametrize("target", ["report", "artifact", "scorer_evidence"])
def test_formal_report_rejects_oversized_evidence_file(
    tmp_path,
    monkeypatch,
    target,
):
    import lha.release_claims as claims

    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    report_path = root / "benchmarks/ablation_report.json"
    raw = json.loads(report_path.read_text())
    if target == "report":
        monkeypatch.setattr(claims, "_MAX_REPORT_BYTES", 128)
        path = report_path
    elif target == "artifact":
        monkeypatch.setattr(claims, "_MAX_ARTIFACT_BYTES", 128)
        digest = raw["records"][0]["artifact_sha256"]
        path = root / "benchmarks/artifacts" / f"{digest}.json"
    else:
        monkeypatch.setattr(claims, "_MAX_SCORER_EVIDENCE_BYTES", 128)
        digest = raw["records"][0]["scorer_evidence_sha256"]
        path = root / "benchmarks/scorer_evidence" / f"{digest}.json"
    path.write_bytes(b"x" * 129)

    with pytest.raises(ReleaseClaimsError, match="exceeds the 128-byte limit"):
        validate_release_claims(root)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_formal_report_rejects_linked_scorer_evidence(
    tmp_path,
    monkeypatch,
    link_kind,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    raw = json.loads((root / "benchmarks/ablation_report.json").read_text())
    digest = raw["records"][0]["scorer_evidence_sha256"]
    evidence = root / "benchmarks/scorer_evidence" / f"{digest}.json"
    backing = root / "benchmarks/scorer_evidence/backing.json"
    backing.write_bytes(evidence.read_bytes())
    evidence.unlink()
    if link_kind == "symlink":
        evidence.symlink_to(backing)
    else:
        os.link(backing, evidence)

    with pytest.raises(
        ReleaseClaimsError,
        match="regular file|hard links",
    ):
        validate_release_claims(root)


def test_formal_report_rejects_receipt_swapped_to_another_artifact(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path = root / "benchmarks/ablation_report.json"
    raw = json.loads(path.read_text())
    target = [
        record
        for record in raw["records"]
        if record["task"] == "task_one"
        and record["rep"] == 0
        and record["condition"] in {"trust", "gate"}
    ]
    donor = next(
        record
        for record in raw["records"]
        if record["task"] == "task_two" and record["rep"] == 1 and record["condition"] == "trust"
    )
    assert all(record["scorer_outcome"] == donor["scorer_outcome"] for record in target)
    assert all(
        record["scorer_expected_tests"] == donor["scorer_expected_tests"] for record in target
    )
    assert all(record["artifact_sha256"] != donor["artifact_sha256"] for record in target)
    for record in target:
        record["scorer_evidence_sha256"] = donor["scorer_evidence_sha256"]
    path.write_text(json.dumps(raw))

    with pytest.raises(ReleaseClaimsError, match="binding disagrees"):
        validate_release_claims(root)


def test_schema_three_cannot_be_published_as_current_formal_evidence(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    _write_legacy_report(root, monkeypatch)
    path = root / "benchmarks/ablation_report.json"
    raw = json.loads(path.read_text())
    raw["schema_version"] = 3
    path.write_text(json.dumps(raw))
    historical = load_ablation_report(path)
    (root / "benchmarks/ablation_report.md").write_text(
        historical.to_markdown().replace(
            "# Verification ablation",
            "# Verification ablation — legacy snapshot",
            1,
        )
    )
    run_horizon("benchmarks/ablation_report.json", "benchmarks")
    horizon_md = root / "benchmarks/horizon_report.md"
    horizon_md.write_text(
        horizon_md.read_text().replace(
            "# Error compounding over a horizon",
            "# Error compounding over a horizon — legacy snapshot",
            1,
        )
    )
    readme = root / "README.md"
    readme.write_text(readme.read_text().replace("历史报告", "正式报告"))

    with pytest.raises(ReleaseClaimsError, match="历史报告"):
        validate_release_claims(root)
