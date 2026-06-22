"""Dynamic LLM planning: opt-in, validated, and template-fallback on any failure.

The default path (and `lha eval`) stays on the deterministic template — these tests
drive a fake LLM so no network/real backend is involved.
"""

from __future__ import annotations

from lha.agents.supervisor import Supervisor
from lha.config import Config
from lha.llm.base import LLMClient
from lha.tasks.spec import TaskSpec

_TASK = "data/tasks/fix_average.yaml"
_TEMPLATE_ACTIONS = ["gather_context", "edit_code"]


class _FakeLLM(LLMClient):
    name = "fake"

    def __init__(self, response: str):
        self._response = response

    def complete(self, system: str, prompt: str) -> str:
        return self._response


def test_planning_uses_template_by_default():
    # dynamic_planning is off by default -> template plan regardless of the backend
    plan = Supervisor(Config(), _FakeLLM("anything")).plan(TaskSpec.from_file(_TASK))
    assert [s.action for s in plan.steps] == _TEMPLATE_ACTIONS


def test_planning_uses_a_valid_llm_plan_when_enabled():
    resp = (
        "```json\n"
        '{"summary": "fix it", "steps": ['
        '{"step_id": "s1", "kind": "code", "action": "edit_code", '
        '"goal": "fix the bug", "verifiers": ["pytest", "ruff"]}]}\n```'
    )
    plan = Supervisor(Config(dynamic_planning=True), _FakeLLM(resp)).plan(TaskSpec.from_file(_TASK))
    assert len(plan.steps) == 1 and plan.steps[0].action == "edit_code"
    assert plan.steps[0].verifiers == ["pytest", "ruff"]


def test_planning_falls_back_on_unregistered_verifier():
    # a plan that names a verifier the harness can't run is rejected -> template
    resp = (
        "```json\n"
        '{"summary": "x", "steps": ['
        '{"step_id": "s1", "kind": "code", "action": "edit_code", '
        '"goal": "g", "verifiers": ["not-a-verifier"]}]}\n```'
    )
    plan = Supervisor(Config(dynamic_planning=True), _FakeLLM(resp)).plan(TaskSpec.from_file(_TASK))
    assert [s.action for s in plan.steps] == _TEMPLATE_ACTIONS


def test_planning_falls_back_on_unparseable_response():
    plan = Supervisor(Config(dynamic_planning=True), _FakeLLM("not json at all")).plan(
        TaskSpec.from_file(_TASK)
    )
    assert [s.action for s in plan.steps] == _TEMPLATE_ACTIONS


def test_planning_rejects_unsafe_step_id():
    # an LLM plan whose step_id would escape the run dir must be rejected -> template
    resp = (
        "```json\n"
        '{"summary": "x", "steps": ['
        '{"step_id": "../../escape", "kind": "code", "action": "edit_code", '
        '"goal": "g", "verifiers": ["pytest"]}]}\n```'
    )
    plan = Supervisor(Config(dynamic_planning=True), _FakeLLM(resp)).plan(TaskSpec.from_file(_TASK))
    assert [s.action for s in plan.steps] == _TEMPLATE_ACTIONS


def test_planning_falls_back_when_no_llm():
    # dynamic_planning on but no backend -> still the template
    plan = Supervisor(Config(dynamic_planning=True), None).plan(TaskSpec.from_file(_TASK))
    assert [s.action for s in plan.steps] == _TEMPLATE_ACTIONS
