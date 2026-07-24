"""Benchmark adapters: format contracts and calibration, with no paid runs.

The SWE-bench tests pin the exact official predictions format, invocation, and
report parsing. The Terminal-Bench tests exercise the Harbor agent against a
stubbed harbor package (the real one needs Python >= 3.12). Nothing here
downloads a dataset, builds an image, or calls a model.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest
import yaml

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


# --- Terminal-Bench agent (harbor stubbed) -----------------------------------
def test_task_yaml_maps_the_instruction():
    spec = yaml.safe_load(tb.task_yaml("Fix the broken cron entry\nDetails here."))
    assert spec["kind"] == "issue_to_pr"
    assert spec["description"].startswith("Fix the broken cron entry")
    assert spec["target_repo"] == "."
    assert spec["context_requirement"] == "optional"  # no search backend in-container


def test_parse_result_line():
    out = "noise\n__LHA_RESULT__ " + json.dumps({"status": "DONE", "run_id": "r1"}) + "\n"
    assert tb.parse_result_line(out) == {"status": "DONE", "run_id": "r1"}
    assert tb.parse_result_line("no marker here") is None
    assert tb.parse_result_line("__LHA_RESULT__ {broken") is None


def test_run_command_pins_model_and_isolates_state():
    cmd = tb.run_command("claude-haiku-4-5-20251001")
    assert "LHA_CLAUDE_MODEL=claude-haiku-4-5-20251001" in cmd
    assert "LHA_RUNS_DIR=/tmp/lha_runs" in cmd  # graded filesystem stays clean
    assert "--auto-approve" in cmd and "--json" in cmd


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
    """Records exec/upload calls; answers the lha invocation with a result line."""

    def __init__(self, result: dict | None):
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self._result = result

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append(command)
        stdout = ""
        if " lha run " in command and self._result is not None:
            stdout = "__LHA_RESULT__ " + json.dumps(self._result)
        return types.SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    async def upload_file(self, source_path, target_path):
        self.uploads.append((str(source_path), str(target_path)))


def test_agent_install_uploads_wheel_and_installs(monkeypatch, tmp_path):
    _stub_harbor(monkeypatch)
    agent = tb.build_agent()(logs_dir=tmp_path, wheel_path="dist/lha-0.3.0-py3-none-any.whl")
    assert type(agent).name() == "lha"
    assert agent.version() == lha.__version__

    env = _FakeEnv(result=None)
    asyncio.run(agent.install(env))
    assert env.uploads == [("dist/lha-0.3.0-py3-none-any.whl", "/tmp/lha.whl")]
    assert any("pip install" in c for c in env.commands)
    assert any("claude" in c for c in env.commands)


def test_agent_install_without_wheel_fails_closed(monkeypatch, tmp_path):
    _stub_harbor(monkeypatch)
    monkeypatch.delenv("LHA_WHEEL", raising=False)
    agent = tb.build_agent()(logs_dir=tmp_path)
    with pytest.raises(RuntimeError, match="wheel"):
        asyncio.run(agent.install(_FakeEnv(result=None)))


def test_agent_run_copies_workdir_only_when_done(monkeypatch, tmp_path):
    _stub_harbor(monkeypatch)
    make = tb.build_agent()

    done = _FakeEnv({"status": "DONE", "run_id": "r42", "llm_usage": {"input_tokens": 10}})
    ctx = types.SimpleNamespace(n_input_tokens=None, n_output_tokens=None, cost_usd=None)
    asyncio.run(make(logs_dir=tmp_path, wheel_path="w.whl").run("fix it", done, ctx))
    assert any("cp -a" in c and "r42/workdir" in c for c in done.commands)
    assert ctx.n_input_tokens == 10

    failed = _FakeEnv({"status": "FAILED", "run_id": "r43"})
    asyncio.run(make(logs_dir=tmp_path, wheel_path="w.whl").run("fix it", failed, ctx))
    assert not any("cp -a" in c for c in failed.commands)  # unverified edits stay put


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
