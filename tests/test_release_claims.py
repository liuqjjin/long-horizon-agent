from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from lha.ablation import (
    AblationProvenance,
    AblationReport,
    RunRecord,
    _aggregate,
    load_ablation_report,
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


def test_legacy_snapshot_requires_an_explicit_readme_marker(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_legacy_report(root, monkeypatch)
    readme = root / "README.md"
    readme.write_text(readme.read_text().replace("历史快照（legacy）", "消融报告"))

    with pytest.raises(ReleaseClaimsError, match="历史快照"):
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
    records: list[RunRecord] = []
    for task in tasks:
        for rep in range(reps):
            first_attempt_correct = not (task == "task_two" and rep == 0)
            records.extend(
                [
                    RunRecord(
                        task=task,
                        condition="trust",
                        rep=rep,
                        status="DONE",
                        claimed_success=True,
                        true_success=first_attempt_correct,
                        false_success=not first_attempt_correct,
                        repairs=0,
                    ),
                    RunRecord(
                        task=task,
                        condition="gate",
                        rep=rep,
                        status="DONE" if first_attempt_correct else "FAILED",
                        claimed_success=first_attempt_correct,
                        true_success=first_attempt_correct,
                        false_success=False,
                        repairs=0,
                        gate_prediction=first_attempt_correct,
                    ),
                    RunRecord(
                        task=task,
                        condition="verify",
                        rep=rep,
                        status="DONE",
                        claimed_success=True,
                        true_success=True,
                        false_success=False,
                        repairs=0 if first_attempt_correct else 1,
                        gate_prediction=True,
                    ),
                ]
            )

    digest = "b" * 64
    provenance = AblationProvenance(
        generated_at="2026-07-27T00:00:00+00:00",
        harness_version="0.5.0.dev0",
        git_commit="a" * 40,
        git_dirty=False,
        source_tree_sha256=digest,
        requested_llm_backend="codex_cli",
        actual_llm_backend="codex_cli",
        model="model-x",
        cli_version="codex-cli 1.0",
        agent_backend="trusted-local",
        scorer_requested="trusted-local",
        scorer_backend="trusted-local",
        platform="test-platform",
        python_version="3.11.9",
        pytest_version="9.1.1",
        task_paths={task: f"data/tasks/{task}.yaml" for task in tasks},
        corpus_paths={task: f"data/bench/{task}" for task in tasks},
        task_files_sha256={task: digest for task in tasks},
        corpus_sha256={task: digest for task in tasks},
        configuration={
            "repetitions": reps,
            "task_count": len(tasks),
            "conditions": ["trust", "gate", "verify"],
        },
    )
    report = AblationReport(
        llm="codex_cli",
        model="model-x",
        reps=reps,
        tasks=tasks,
        records=records,
        stats=_aggregate(records),
        scorer="trusted-local",
        fingerprint="c" * 64,
        backend_version="codex-cli 1.0",
        provenance=provenance,
    )
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True)
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

仓库中的消融报告是正式报告（formal），使用 2 个预设 Python 缺陷，每个任务重复 2 次，共 4 组相同首轮补丁。
实测模型为 `model-x`。

| 条件 | 处理方式 | 独立评分 |
|---|---|---|
| `trust` | 直接接受首轮补丁 | 3 个正确，1 个错误仍被接受 |
| `gate` | 首轮补丁必须通过测试 | 接受 3 个正确补丁，拦截 1 个错误补丁 |
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
    for stat in raw["stats"]:
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
        .replace("正式报告（formal）", "历史快照（legacy）")
        .replace(
            "## 其他",
            "204 次正式复测只有在运行并提交原始记录后才会更新。\n\n## 其他",
        )
    )


def test_schema_two_reports_are_checked_as_formal_evidence(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_formal_report(root, monkeypatch)

    summary = validate_release_claims(root)
    assert summary.status == "formal"
    assert summary.tasks == 2
    assert summary.repetitions == 2
    assert summary.cells == 4
    assert summary.trust_false_successes == 1
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
