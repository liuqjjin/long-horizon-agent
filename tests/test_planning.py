"""Dynamic LLM planning: opt-in, validated, and fallback only for invalid candidates.

The default path (and `lha eval`) stays on the deterministic template — these tests
drive a fake LLM so no network/real backend is involved. Backend failures remain
failures; they are not converted into a template plan.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from lha.agents.supervisor import Supervisor
from lha.artifacts import Plan, Step
from lha.config import Config
from lha.harness.checkpoint import load_state, save_state
from lha.harness.errors import CheckpointCorrupt
from lha.harness.loop import _dump
from lha.harness.state import RunState
from lha.llm.base import LLMClient
from lha.step_ids import validate_plan_step_ids
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
        '{"step_id": "s1", "kind": "code", "action": "gather_context", '
        '"goal": "read", "verifiers": ["freshness", "citation"]},'
        '{"step_id": "s2", "kind": "code", "action": "edit_code", '
        '"goal": "fix the bug", "verifiers": ["pytest", "ruff"]}]}\n```'
    )
    plan = Supervisor(Config(dynamic_planning=True), _FakeLLM(resp)).plan(TaskSpec.from_file(_TASK))
    assert [step.step_id for step in plan.steps] == ["s1", "s2"]
    assert plan.steps[1].verifiers == ["pytest", "ruff"]


def test_planning_cannot_omit_the_edit_or_pytest_gate():
    candidates = [
        (
            '{"summary":"x","steps":[{"step_id":"s1","kind":"code",'
            '"action":"gather_context","goal":"read",'
            '"verifiers":["freshness","citation"]}]}'
        ),
        (
            '{"summary":"x","steps":['
            '{"step_id":"s1","kind":"code","action":"gather_context",'
            '"goal":"read","verifiers":["freshness","citation"]},'
            '{"step_id":"s2","kind":"code","action":"edit_code",'
            '"goal":"fix","verifiers":["ruff"]}]}'
        ),
    ]
    for response in candidates:
        plan = Supervisor(
            Config(dynamic_planning=True), _FakeLLM(response)
        ).plan(TaskSpec.from_file(_TASK))
        assert [step.action for step in plan.steps] == _TEMPLATE_ACTIONS
        assert plan.steps[-1].verifiers == ["pytest", "ruff"]


def test_planning_cannot_weaken_approval_or_required_context():
    task = TaskSpec.from_file("data/tasks/fix_average_approval.yaml")
    response = (
        '{"summary":"x","steps":['
        '{"step_id":"s1","kind":"code","action":"gather_context",'
        '"goal":"read","verifiers":["freshness","citation"],'
        '"context_requirement":"optional"},'
        '{"step_id":"s2","kind":"code","action":"edit_code",'
        '"goal":"fix","verifiers":["pytest","ruff"],'
        '"requires_approval":false,"context_requirement":"optional"}]}'
    )
    plan = Supervisor(Config(dynamic_planning=True), _FakeLLM(response)).plan(task)
    assert [step.action for step in plan.steps] == _TEMPLATE_ACTIONS
    assert plan.steps[-1].requires_approval
    assert all(step.context_requirement == "required" for step in plan.steps)


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


def test_planning_does_not_swallow_backend_failure():
    class _BrokenBackend(LLMClient):
        name = "broken"

        def complete(self, system: str, prompt: str) -> str:
            raise CheckpointCorrupt("malformed backend response")

    with pytest.raises(CheckpointCorrupt, match="malformed backend response"):
        Supervisor(Config(dynamic_planning=True), _BrokenBackend()).plan(
            TaskSpec.from_file(_TASK)
        )


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


def test_planning_rejects_duplicate_step_ids():
    resp = (
        '{"summary":"x","steps":['
        '{"step_id":"same","kind":"code","action":"edit_code","goal":"a",'
        '"verifiers":["pytest"]},'
        '{"step_id":"same","kind":"code","action":"edit_code","goal":"b",'
        '"verifiers":["ruff"]}]}'
    )
    plan = Supervisor(Config(dynamic_planning=True), _FakeLLM(resp)).plan(
        TaskSpec.from_file(_TASK)
    )
    assert [step.action for step in plan.steps] == _TEMPLATE_ACTIONS


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("...", "step"),
        ("A", "a"),
        ("é", unicodedata.normalize("NFD", "é")),
    ],
)
def test_dynamic_planning_rejects_step_ids_that_could_alias_artifacts(
    first: str,
    second: str,
):
    response = json.dumps(
        {
            "summary": "x",
            "steps": [
                {
                    "step_id": first,
                    "kind": "code",
                    "action": "gather_context",
                    "goal": "read",
                    "verifiers": ["freshness", "citation"],
                },
                {
                    "step_id": second,
                    "kind": "code",
                    "action": "edit_code",
                    "goal": "fix",
                    "verifiers": ["pytest", "ruff"],
                },
            ],
        }
    )
    plan = Supervisor(Config(dynamic_planning=True), _FakeLLM(response)).plan(
        TaskSpec.from_file(_TASK)
    )
    assert [step.step_id for step in plan.steps] == ["s1-context", "s2-fix"]


def test_plan_rejects_casefold_aliases():
    first = Step(
        step_id="A",
        kind="code",
        action="edit_code",
        goal="first",
        verifiers=["pytest"],
    )
    second = first.model_copy(update={"step_id": "a", "goal": "second"})
    with pytest.raises(ValueError, match="alias"):
        Plan(task_id="task", summary="summary", steps=[first, second])


def test_plan_validator_detects_unicode_normalization_aliases_before_grammar():
    composed = "é"
    decomposed = unicodedata.normalize("NFD", composed)
    with pytest.raises(ValueError, match="alias"):
        validate_plan_step_ids([composed, decomposed])


def test_artifact_writer_rejects_invalid_step_id_before_any_write(tmp_path: Path):
    with pytest.raises(ValueError, match="artifact identity"):
        _dump(tmp_path, "...", "patch.json", "{}")

    assert not (tmp_path / "patch.json").exists()
    assert not (tmp_path / "steps").exists()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("...", "step"),
        ("A", "a"),
        ("é", unicodedata.normalize("NFD", "é")),
    ],
)
def test_checkpoint_load_rejects_noncanonical_or_aliasing_step_ids(
    tmp_path: Path,
    first: str,
    second: str,
):
    task = TaskSpec.from_file(_TASK)
    state = RunState.new(
        task,
        run_id="checkpoint-step-ids",
        run_dir=str(tmp_path),
        workdir=str(tmp_path / "workdir"),
        config=Config(),
    )
    state.plan = Supervisor(Config(), None).plan(task)
    save_state(state)

    checkpoint = tmp_path / "state.json"
    envelope = json.loads(checkpoint.read_text())
    envelope["payload"]["plan"]["steps"][0]["step_id"] = first
    envelope["payload"]["plan"]["steps"][1]["step_id"] = second
    canonical = json.dumps(
        envelope["payload"], sort_keys=True, separators=(",", ":")
    ).encode()
    envelope["sha256"] = hashlib.sha256(canonical).hexdigest()
    checkpoint.write_text(json.dumps(envelope))

    with pytest.raises(CheckpointCorrupt, match="does not validate"):
        load_state(tmp_path)


def test_planning_rejects_verifier_family_mismatch():
    for action, kind, verifier in (
        ("edit_code", "code", "citation"),
        ("run_experiment", "experiment", "pytest"),
    ):
        response = (
            '{"summary":"x","steps":[{'
            f'"step_id":"s","kind":"{kind}","action":"{action}",'
            f'"goal":"g","verifiers":["{verifier}"]'
            "}]}"
        )
        plan = Supervisor(Config(dynamic_planning=True), _FakeLLM(response)).plan(
            TaskSpec.from_file(_TASK)
        )
        assert [step.action for step in plan.steps] == _TEMPLATE_ACTIONS
