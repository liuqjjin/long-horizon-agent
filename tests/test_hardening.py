"""Pinning tests for the verification-first hardening pass.

Each test pins a fix whose absence would let an unverified result pass, a budget
escape across resume, a silent LLM truncation, or a facade-isolation leak. Without
the corresponding fix the test fails — that is the point.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.agents.verifier_agent import VerifierAgent
from lha.artifacts import Patch, Step
from lha.config import Config
from lha.harness import Harness
from lha.harness.approval import ApprovalDecision, HumanApprovalGate
from lha.llm.anthropic_client import AnthropicClient, AnthropicResponseError
from lha.tools.shell import ProcResult
from lha.verifiers import VerifyContext, registry
from lha.verifiers.base import Verifier
from lha.verifiers.code import RuffVerifier
from lha.verifiers.verdict import Check


def _step(*verifiers: str) -> Step:
    return Step(step_id="s", kind="code", action="edit_code", goal="g", verifiers=list(verifiers))


def _cfg(tmp_path: Path, **over) -> Config:
    base = dict(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
    )
    base.update(over)
    return Config(**base)


# --- ruff verifier must fail when ruff could not run ------------------------
class _CannedExec:
    """Execution-backend stub returning a scripted result (verifier tests)."""

    def __init__(self, result: ProcResult):
        self._result = result

    def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
        return self._result

    def python(self):
        return "python"

    def tool(self, name):
        return name


def test_ruff_fails_when_it_cannot_run(tmp_path):
    # rc=2 (config/internal error) with empty stdout: ruff produced no lint pass.
    exec_stub = _CannedExec(ProcResult(2, "", "ruff: config error", 0.0))
    check = RuffVerifier().verify(
        Patch(step_id="s"), VerifyContext(workdir=tmp_path, step=_step("ruff"), exec=exec_stub)
    )
    assert not check.passed  # "couldn't verify" must not read as "verified"
    assert "failed to run" in check.detail["summary"]


def test_ruff_fails_on_inconsistent_exit_codes(tmp_path):
    # rc=1 means "violations found" but stdout is empty -> inconsistent -> fail.
    c1 = RuffVerifier().verify(
        Patch(step_id="s"),
        VerifyContext(
            workdir=tmp_path, step=_step("ruff"), exec=_CannedExec(ProcResult(1, "", "", 0.0))
        ),
    )
    assert not c1.passed

    # non-JSON on the success channel -> ruff did not run cleanly -> fail.
    c2 = RuffVerifier().verify(
        Patch(step_id="s"),
        VerifyContext(
            workdir=tmp_path,
            step=_step("ruff"),
            exec=_CannedExec(ProcResult(0, "not json", "", 0.0)),
        ),
    )
    assert not c2.passed


# --- a step that verified nothing must not pass vacuously --------------------
def test_empty_verifier_list_fails_closed(tmp_path):
    step = _step()  # no verifiers declared
    verdict = VerifierAgent(parallel=False).verify(
        step, Patch(step_id="s"), VerifyContext(workdir=tmp_path, step=step)
    )
    assert not verdict.passed
    assert any(c.name == "no-verifier" for c in verdict.checks)


def test_unregistered_verifier_fails_closed(tmp_path):
    step = _step("does-not-exist")
    verdict = VerifierAgent(parallel=False).verify(
        step, Patch(step_id="s"), VerifyContext(workdir=tmp_path, step=step)
    )
    assert not verdict.passed


# --- budget bounds the whole run, not a single process ----------------------
def test_budget_is_per_run_across_resume(tmp_path):
    task = hermetic_task("data/tasks/fix_average.yaml")

    paused = Harness(_cfg(tmp_path, max_steps=1)).run(task)
    assert paused.status == "PAUSED"
    # exactly one step executed (the failing tick is not counted — no off-by-one).
    assert paused.state.steps_used == 1

    # resuming with the SAME exhausted budget must re-pause without advancing —
    # proving the budget carries across processes rather than resetting.
    again = Harness(_cfg(tmp_path, max_steps=1)).resume(paused.state.run_id)
    assert again.status == "PAUSED"
    assert again.state.cursor == 1

    # a larger budget lets the same run finish.
    done = Harness(_cfg(tmp_path, max_steps=20)).resume(paused.state.run_id)
    assert done.status == "DONE"


# --- deadline_s reaches the loop through the documented env path ------------
def test_deadline_s_is_read_from_env(monkeypatch):
    monkeypatch.setenv("LHA_DEADLINE_S", "1.5")
    assert Config.from_env().deadline_s == 1.5
    monkeypatch.delenv("LHA_DEADLINE_S", raising=False)
    assert Config.from_env().deadline_s is None


def test_deadline_s_rejects_non_finite_or_negative(monkeypatch):
    for bad in ("nan", "inf", "-1", "abc"):
        monkeypatch.setenv("LHA_DEADLINE_S", bad)
        with pytest.raises(ValueError):
            Config.from_env()
    monkeypatch.delenv("LHA_DEADLINE_S", raising=False)


# --- Anthropic backend never returns a truncated / refused completion -------
class _Block:
    type = "text"
    text = "ok"


class _Msg:
    def __init__(self, stop_reason: str):
        self.stop_reason = stop_reason
        self.stop_details = type("SD", (), {"category": "cyber"})()
        self.content = [_Block()]


class _FakeClient:
    def __init__(self, stop_reason: str):
        self.messages = type("M", (), {"create": lambda _self, **kw: _Msg(stop_reason)})()


def test_anthropic_raises_on_truncation_and_refusal():
    trunc = AnthropicClient(max_tokens=100)
    trunc._client = _FakeClient("max_tokens")
    with pytest.raises(AnthropicResponseError):
        trunc.complete("sys", "prompt")

    refused = AnthropicClient(max_tokens=100)
    refused._client = _FakeClient("refusal")
    with pytest.raises(AnthropicResponseError):
        refused.complete("sys", "prompt")

    ok = AnthropicClient(max_tokens=100)
    ok._client = _FakeClient("end_turn")
    assert ok.complete("sys", "prompt") == "ok"


class _FakeStream:
    def __init__(self, msg):
        self._msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self._msg


class _FakeStreamingClient:
    def __init__(self, stop_reason: str):
        msg = _Msg(stop_reason)
        self.messages = type("M", (), {"stream": lambda _self, **kw: _FakeStream(msg)})()


def test_anthropic_default_max_tokens_and_streaming_branch():
    assert AnthropicClient().max_tokens == 16000  # raised from the old silent-truncating 4096
    # a large cap must route through streaming, not a non-streaming create.
    streamed = AnthropicClient(max_tokens=20000)
    streamed._client = _FakeStreamingClient("end_turn")
    assert streamed.complete("sys", "prompt") == "ok"


# --- the facade-isolation rule, enforced by AST (not just a grep) -----------
def test_no_cocoindex_imports_outside_live_context():
    root = Path(__file__).resolve().parents[1] / "src" / "lha"
    banned = {"cocoindex", "cocoindex_code"}
    offenders: list[str] = []
    for py in root.rglob("*.py"):
        if "live_context" in py.parts:
            continue  # the facade is the one place allowed to import the indexer
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [
                    f"{py}: import {n.name}" for n in node.names if n.name.split(".")[0] in banned
                ]
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in banned:
                    offenders.append(f"{py}: from {node.module}")
    assert not offenders, offenders


# --- a step that exhausts its repairs is reverted in BOTH runtimes ----------
class _AlwaysFail(Verifier):
    name = "pytest"  # shadow the real pytest verifier for the test
    family = "code"

    def verify(self, artifact, ctx) -> Check:
        return Check(name="pytest", family="code", passed=False, detail={"summary": "forced fail"})


def _force_failing_pytest(monkeypatch):
    monkeypatch.setitem(registry._REGISTRY, "pytest", _AlwaysFail())


def test_loop_reverts_unverified_patch(tmp_path, monkeypatch):
    _force_failing_pytest(monkeypatch)
    result = Harness(_cfg(tmp_path, max_repairs=0)).run(
        hermetic_task("data/tasks/fix_average.yaml")
    )
    assert result.status == "FAILED"
    # the stub applied its fix, verification failed, so the change must be reverted:
    # the original bug is back in the sandbox.
    restored = (Path(result.state.run_dir) / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" in restored


def test_langgraph_reverts_unverified_patch(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")
    _force_failing_pytest(monkeypatch)
    from lha.runtime.langgraph_runner import LangGraphHarness

    result = LangGraphHarness(_cfg(tmp_path, max_repairs=0)).run(
        hermetic_task("data/tasks/fix_average.yaml")
    )
    assert result.status == "FAILED"
    # without the revert fix, langgraph would leave the unverified patch in place.
    restored = (Path(result.state.run_dir) / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" in restored


def test_langgraph_budget_caps_steps(tmp_path):
    pytest.importorskip("langgraph")
    from lha.runtime.langgraph_runner import LangGraphHarness

    paused = LangGraphHarness(_cfg(tmp_path, max_steps=1)).run(
        hermetic_task("data/tasks/fix_average.yaml")
    )
    assert paused.status == "PAUSED"  # max_steps bounds the durable runtime too
    assert paused.state.steps_used == 1
    assert paused.state.cursor == 1  # ran step 1, paused before step 2


def test_langgraph_deadline_pauses(tmp_path):
    pytest.importorskip("langgraph")
    from lha.runtime.langgraph_runner import LangGraphHarness

    paused = LangGraphHarness(_cfg(tmp_path, deadline_s=0.0)).run(
        hermetic_task("data/tasks/fix_average.yaml")
    )
    assert paused.status == "PAUSED"  # deadline_s now bounds the durable runtime too


def test_loop_reverts_on_approval_rejection(tmp_path):
    cfg = _cfg(tmp_path)  # auto_approve defaults to False -> non-interactive pause
    paused = Harness(cfg).run(hermetic_task("data/tasks/fix_average_approval.yaml"))
    assert paused.status == "AWAITING_APPROVAL"
    mathutils = Path(paused.state.run_dir) / "workdir" / "mathutils.py"
    assert "len(values) - 1" not in mathutils.read_text()  # patch applied, pending approval

    HumanApprovalGate(paused.state.run_dir).resolve(approved=False, note="rejected")
    rejected = Harness(cfg).resume(paused.state.run_id)
    assert rejected.status == "FAILED"
    assert "len(values) - 1" in mathutils.read_text()  # rejected change reverted


def test_approval_reuses_the_reviewed_patch_on_resume(tmp_path, monkeypatch):
    from lha.agents import implementer

    cfg = _cfg(tmp_path)
    paused = Harness(cfg).run(hermetic_task("data/tasks/fix_average_approval.yaml"))
    assert paused.status == "AWAITING_APPROVAL"
    HumanApprovalGate(paused.state.run_dir).resolve(approved=True, note="ok")

    # The approved patch must be reused, not regenerated: if the harness called the
    # implementer again on resume this would raise and the run would FAIL instead.
    def boom(self, *a, **k):
        raise RuntimeError("must not regenerate an approved patch")

    monkeypatch.setattr(implementer.Implementer, "implement", boom)
    done = Harness(cfg).resume(paused.state.run_id)
    assert done.status == "DONE"
    fixed = (Path(done.state.run_dir) / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" not in fixed  # the reviewed fix was applied


def test_stale_approval_decision_is_ignored(tmp_path):
    cfg = _cfg(tmp_path)
    paused = Harness(cfg).run(hermetic_task("data/tasks/fix_average_approval.yaml"))
    assert paused.status == "AWAITING_APPROVAL"
    # an approval left for a DIFFERENT step must not resume the current one.
    (Path(paused.state.run_dir) / "approval.json").write_text(
        ApprovalDecision(approved=True, step_id="some-other-step").model_dump_json()
    )
    resumed = Harness(cfg).resume(paused.state.run_id)
    assert resumed.status == "AWAITING_APPROVAL"  # stale decision discarded, not applied
