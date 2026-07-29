"""Regressions for the correctness-review findings (v0.3 hardening pass).

Each test pins one reviewed defect: approval binding for non-patch steps,
LangGraph mid-node faults, ledger append after a torn tail, case-variant and
quoted-path bypasses of the oracle policy, stale LLM usage on error, and a
partially dark context backend masquerading as "ok".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import hermetic_task

from lha.agents.supervisor import Supervisor
from lha.agents.verifier_agent import VerifierAgent
from lha.artifacts import Step
from lha.clock import now
from lha.config import Config
from lha.harness import Harness
from lha.harness.approval import ApprovalDecision, HumanApprovalGate
from lha.harness.checkpoint import append_ledger, load_state, read_ledger
from lha.harness.state import StepRecord
from lha.live_context.models import ContextBundle, Freshness
from lha.tools.policy import diff_paths, is_protected
from lha.verifiers import VerifyContext
from lha.verifiers.context import FreshnessVerifier


def _cfg(tmp_path, **kw) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
        **kw,
    )


# --- approvals for non-patch steps bind by step id (no livelock) -------------
def test_binds_step_only_for_actions_without_a_patch():
    request_sha = "a" * 64
    request_ref = "attempts/s1-r0/approval_request.json"
    step_only = ApprovalDecision(
        approved=True,
        step_id="s1",
        attempt_id="s1-r0",
        request_ref=request_ref,
        request_sha256=request_sha,
        artifact_sha256=None,
    )
    binding = {
        "step_id": "s1",
        "attempt_id": "s1-r0",
        "request_ref": request_ref,
        "request_sha256": request_sha,
        "artifact_sha256": None,
    }
    assert step_only.binds(**binding)
    assert not step_only.binds(**(binding | {"step_id": "s2"}))
    assert not step_only.binds(
        **(binding | {"artifact_sha256": "b" * 64})
    )

    with_hash = step_only.model_copy(
        update={"artifact_sha256": "b" * 64}
    )
    assert with_hash.binds(
        **(binding | {"artifact_sha256": "b" * 64})
    )
    assert not with_hash.binds(**binding)


def test_gated_context_step_approval_takes_effect_on_resume(tmp_path, monkeypatch):
    task = hermetic_task("data/tasks/fix_average.yaml")
    real_plan = Supervisor.plan

    def plan_with_gated_context(self, t):
        plan = real_plan(self, t)
        assert plan.steps[0].action == "gather_context"
        plan.steps[0].requires_approval = True
        return plan

    monkeypatch.setattr(Supervisor, "plan", plan_with_gated_context)
    r1 = Harness(_cfg(tmp_path)).run(task, run_id="gated-ctx")
    assert r1.status == "AWAITING_APPROVAL"

    HumanApprovalGate(tmp_path / "runs" / "gated-ctx").resolve(approved=True)
    monkeypatch.undo()
    r2 = Harness(_cfg(tmp_path)).resume("gated-ctx")
    assert r2.status == "DONE"  # the approval bound and the run finished


# --- LangGraph: a mid-node fault fails closed, it does not wedge -------------
def test_langgraph_mid_node_fault_fails_closed(tmp_path, monkeypatch):
    from lha.runtime.langgraph_runner import LangGraphHarness

    task = hermetic_task("data/tasks/fix_average.yaml")
    real_verify = VerifierAgent.verify

    def boom(self, step, artifact, ctx):
        if step.action == "edit_code":
            raise RuntimeError("verifier crashed mid-node")
        return real_verify(self, step, artifact, ctx)

    monkeypatch.setattr(VerifierAgent, "verify", boom)
    r = LangGraphHarness(_cfg(tmp_path)).run(task, run_id="lg-crash")
    assert r.status == "FAILED"
    assert "RuntimeError" in r.message

    state = load_state(tmp_path / "runs" / "lg-crash")
    assert state.status == "FAILED"  # checkpointed terminal, not wedged RUNNING
    # the applied-but-unverified patch did not survive in the sandbox
    content = (Path(state.workdir) / "mathutils.py").read_text()
    assert "len(values) - 1" in content  # the planted bug is back (reverted)


# --- ledger: appending after a torn tail must not merge records --------------
def test_ledger_append_after_torn_tail_drops_only_the_fragment(tmp_path):
    state = SimpleNamespace(run_dir=str(tmp_path))
    append_ledger(state, StepRecord(seq=1, step_id="a", phase="plan"))
    with open(tmp_path / "ledger.jsonl", "a") as f:
        f.write('{"seq": 2, "step_')  # crash mid-append

    append_ledger(state, StepRecord(seq=3, step_id="b", phase="context"))
    assert [r.seq for r in read_ledger(tmp_path)] == [1, 3]

    # and the file stays healthy for further appends (no mid-file corruption)
    append_ledger(state, StepRecord(seq=4, step_id="c", phase="execute"))
    assert [r.seq for r in read_ledger(tmp_path)] == [1, 3, 4]


# --- oracle policy: case variants and quoted headers -------------------------
def test_protected_paths_are_case_insensitive():
    assert is_protected("Conftest.py")
    assert is_protected("PYPROJECT.TOML")
    assert is_protected("Tests/test_anything.py")
    assert not is_protected("src/contest.py")  # not a near-miss false positive


def test_quoted_diff_headers_are_unquoted_before_the_check():
    diff = '--- "a/conftest.py"\n+++ "b/conftest.py"\n@@ -1 +1 @@\n-x\n+y\n'
    paths = diff_paths(diff)
    assert any(is_protected(p) for p in paths)
    assert not any('"' in p for p in paths)


# --- claude CLI: a failed call must not leave stale usage --------------------
def test_claude_cli_clears_stale_usage_on_failure(monkeypatch):
    from lha.llm import claude_cli as mod

    client = mod.ClaudeCLIClient()
    client.last_usage = {"input_tokens": 5, "output_tokens": 2, "cost_usd": 0.1}

    def failing_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=1, stderr="boom", stdout="")

    monkeypatch.setattr(client, "_cli_version", lambda **_kwargs: "test-version")
    monkeypatch.setattr(mod, "_run_isolated_process", failing_run)
    with pytest.raises(RuntimeError, match="return code 1"):
        client.complete("system", "prompt")
    assert client.last_usage is None  # the tracer must not re-count old tokens


# --- context: one dark backend must not hide behind another's hits -----------
def _bundle_with_dark_kind(dark: list[str]) -> ContextBundle:
    return ContextBundle(
        query="q",
        items=[],
        freshness=Freshness(index_version="v", indexed_at=now()),
        status="ok",  # another kind contributed, so the bundle is not empty
        unavailable_kinds=dark,
    )


def _step(requirement: str) -> Step:
    return Step(
        step_id="s",
        kind="code",
        action="edit_code",
        goal="g",
        verifiers=["freshness"],
        context_requirement=requirement,
    )


def test_partially_dark_backend_fails_a_required_step(tmp_path):
    bundle = _bundle_with_dark_kind(["code"])
    check = FreshnessVerifier().verify(
        None, VerifyContext(workdir=tmp_path, step=_step("required"), bundle=bundle)
    )
    assert not check.passed
    assert "code" in check.detail["summary"]


def test_partially_dark_backend_passes_when_optional_or_skill_only(tmp_path):
    optional = FreshnessVerifier().verify(
        None,
        VerifyContext(workdir=tmp_path, step=_step("optional"), bundle=_bundle_with_dark_kind(["code"])),
    )
    assert optional.passed

    skill_only = FreshnessVerifier().verify(
        None,
        VerifyContext(workdir=tmp_path, step=_step("required"), bundle=_bundle_with_dark_kind(["skill"])),
    )
    assert skill_only.passed  # skill memory is an optional augmentation
