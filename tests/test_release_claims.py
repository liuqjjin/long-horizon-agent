from __future__ import annotations

import hashlib
import json
import os
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
    _repo_digest,
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
    assert summary.cells == summary.tasks * summary.repetitions
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
    tasks = ["task_one", "task_two"]
    reps = 2
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
        task_path.write_text(
            "kind: issue_to_pr\n"
            f"title: {task}\n"
            f"target_repo: data/bench/{task}\n"
        )
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
        nonce = hashlib.sha256(
            f"{task}:{rep}:{artifact_sha256}:{correct}".encode()
        ).hexdigest()[:48]
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
            artifact_payload = _frozen_artifact_bytes(
                {"m.py": f"TASK = {task!r}\nREP = {rep}\n"}
            )
            digest = hashlib.sha256(artifact_payload).hexdigest()
            artifact_payloads[digest] = artifact_payload
            first_attempt_correct = not (task == "task_two" and rep == 0)
            gate_accepts = first_attempt_correct and not (
                task == "task_one" and rep == 0
            )
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
        agent_backend="docker",
        scorer_requested="docker",
        scorer_backend="docker",
        scorer_image="lha:release",
        scorer_image_id=scorer_image_id,
        platform="test-platform",
        python_version="3.11.9",
        pytest_version="9.1.1",
        task_paths=task_paths,
        corpus_paths=corpus_paths,
        task_files_sha256=task_digests,
        corpus_sha256=corpus_digests,
        input_snapshot_sha256=snapshot_digests,
        configuration={
            "repetitions": reps,
            "task_count": len(tasks),
            "conditions": ["trust", "gate", "verify"],
            "cache_schema": 7,
            "report_schema": 4,
            "frozen_artifact_schema": 1,
            "input_snapshot_schema": 1,
            "scorer_evidence_schema": 2,
            "scorer_result_source": "nonce-bound-pytest-hook-receipt",
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
        (scorer_evidence_dir / f"{evidence_digest}.json").write_bytes(
            evidence_payload
        )
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
        "llm_calls": [],
        "stats": [asdict(stat) for stat in report.stats],
        "records": [asdict(record) for record in records],
    }
    (benchmarks / "ablation_report.json").write_text(json.dumps(ablation_json, indent=2))
    (benchmarks / "ablation_report.md").write_text(report.to_markdown())

    monkeypatch.chdir(root)
    run_horizon("benchmarks/ablation_report.json", "benchmarks")
    (root / "README.md").write_text(
        """# Project

## 已提交的实测结果

仓库中的消融报告是正式报告，使用 2 个预设 Python 缺陷，每个任务重复 2 次，共 4 组相同首轮补丁。
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
    assert summary.cells == 4
    assert summary.trust_false_successes == 1
    assert summary.gate_successes == 2
    # One additional correct artifact is rejected. It is a false negative, not
    # an interception of an incorrect artifact.
    assert summary.gate_interceptions == 1
    assert summary.verify_successes == 4


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

    with pytest.raises(ReleaseClaimsError, match="hardened Codex CLI"):
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


def test_formal_report_binds_trust_and_gate_to_same_first_attempt(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)
    path = root / "benchmarks/ablation_report.json"
    raw = json.loads(path.read_text())
    gate = next(
        record
        for record in raw["records"]
        if record["task"] == "task_one"
        and record["rep"] == 0
        and record["condition"] == "gate"
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
        if record["task"] == "task_two"
        and record["rep"] == 1
        and record["condition"] == "trust"
    )
    assert all(record["scorer_outcome"] == donor["scorer_outcome"] for record in target)
    assert all(
        record["scorer_expected_tests"] == donor["scorer_expected_tests"]
        for record in target
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
    readme.write_text(
        readme.read_text().replace("历史报告", "正式报告")
    )

    with pytest.raises(ReleaseClaimsError, match="历史报告"):
        validate_release_claims(root)
