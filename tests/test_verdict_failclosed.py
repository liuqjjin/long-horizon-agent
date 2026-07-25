"""The aggregate obeys the same rule as the parts: nothing verified, nothing passed.

`Verdict.from_checks` used to return ``passed=True`` for an empty check list.
Every call site guarded it, so no run ever passed vacuously — but the guard was
a convention, and the harness's first rule ("a check that cannot run fails") was
not enforced by the type. These tests pin it at the primitive.
"""

from __future__ import annotations

from lha.artifacts import Patch, Step
from lha.verifiers import VerifyContext
from lha.verifiers.verdict import Check, Verdict


def test_empty_verdict_fails_closed():
    verdict = Verdict.from_checks("s", [])
    assert verdict.passed is False
    assert verdict.failures == []  # nothing failed; nothing verified either


def test_non_empty_verdicts_are_unchanged():
    ok = Verdict.from_checks("s", [Check(name="pytest", family="code", passed=True)])
    assert ok.passed is True

    mixed = Verdict.from_checks(
        "s",
        [
            Check(name="pytest", family="code", passed=True),
            Check(name="ruff", family="code", passed=False, detail={"summary": "1 violations"}),
        ],
    )
    assert mixed.passed is False
    assert any("ruff" in f for f in mixed.failures)


def test_verifier_agent_still_names_the_reason(tmp_path):
    """Failing closed at the primitive must not cost the repair loop its diagnosis."""
    from lha.agents.verifier_agent import VerifierAgent

    step = Step(step_id="s", kind="code", action="edit_code", goal="g", verifiers=[])
    verdict = VerifierAgent(parallel=False).verify(
        step, Patch(step_id="s"), VerifyContext(workdir=tmp_path, step=step)
    )
    assert verdict.passed is False
    assert any(c.name == "no-verifier" for c in verdict.checks)
    assert verdict.failures  # the reason reaches the repair loop, not just a bare False
