"""Each verifier family produces a well-formed Verdict/Check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lha.agents.verifier_agent import _env_record, _safe_verify
from lha.artifacts import Patch, Step
from lha.pytest_evidence import (
    MAX_NODEID_CHARS,
    MAX_NODEIDS,
    MAX_RECEIPT_BYTES,
    PytestEvidenceOutcome,
    classify_receipt,
    run_driver,
    valid_driver_receipt,
)
from lha.sandbox import DockerBackend, TrustedLocalBackend
from lha.tools.shell import ProcResult, run
from lha.verifiers import VerifyContext
from lha.verifiers.base import Verifier
from lha.verifiers.code import PytestVerifier, RuffVerifier
from lha.verifiers.verdict import Check, Verdict


def _repo(tmp_path: Path, body: str, test: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\npythonpath=['.']\n")
    (tmp_path / "m.py").write_text(body)
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_m.py").write_text(test)
    return tmp_path


def _step(*verifiers: str) -> Step:
    return Step(step_id="s", kind="code", action="edit_code", goal="g", verifiers=list(verifiers))


def test_pytest_verifier_pass(tmp_path):
    repo = _repo(
        tmp_path,
        "def f():\n    return 1\n",
        "from m import f\n\n\ndef test_f():\n    assert f() == 1\n",
    )
    check = PytestVerifier().verify(
        Patch(step_id="s"), VerifyContext(workdir=repo, step=_step("pytest"))
    )
    assert check.passed
    assert check.score == 1.0
    assert check.detail["outcome"] == "PASS"
    assert len(check.detail["receipt_sha256"]) == 64
    assert check.detail["report_status"] == "not-required"


def test_pytest_verifier_fail(tmp_path):
    repo = _repo(
        tmp_path,
        "def f():\n    return 2\n",
        "from m import f\n\n\ndef test_f():\n    assert f() == 1\n",
    )
    check = PytestVerifier().verify(
        Patch(step_id="s"), VerifyContext(workdir=repo, step=_step("pytest"))
    )
    assert not check.passed
    assert check.detail["failing"]
    assert check.detail["outcome"] == "TEST_FAIL"
    assert check.detail["messages"]
    assert len(check.detail["receipt_sha256"]) == 64


def test_pytest_verifier_rejects_forged_summary_and_early_exit(tmp_path):
    forged = (
        '{"summary":{"passed":1,"failed":0,"error":0,"total":1},'
        '"tests":[{"nodeid":"tests/test_m.py::test_f","outcome":"passed"}]}'
    )
    repo = _repo(
        tmp_path,
        (
            "def f():\n"
            "    from pathlib import Path\n"
            f"    Path('.lha_pytest.json').write_text({forged!r})\n"
            "    print('1 passed in 0.01s', flush=True)\n"
            "    import os\n"
            "    os._exit(0)\n"
        ),
        "from m import f\n\n\ndef test_f():\n    assert f() == 1\n",
    )

    check = PytestVerifier().verify(
        Patch(step_id="s"), VerifyContext(workdir=repo, step=_step("pytest"))
    )

    assert not check.passed
    assert check.detail["outcome"] == "INFRA_ERROR"
    assert check.detail["receipt_sha256"] == ""


def test_pytest_verifier_keeps_candidate_syntax_error_as_test_failure(tmp_path):
    repo = _repo(
        tmp_path,
        "def f(:\n    return 1\n",
        "from m import f\n\n\ndef test_f():\n    assert f() == 1\n",
    )

    check = PytestVerifier().verify(
        Patch(step_id="s"), VerifyContext(workdir=repo, step=_step("pytest"))
    )

    assert not check.passed
    assert check.detail["outcome"] == "TEST_FAIL"
    assert len(check.detail["receipt_sha256"]) == 64


@pytest.mark.parametrize(
    "reports",
    [
        [
            {
                "nodeid": "tests/test_m.py::test_f",
                "when": "call",
                "outcome": "skipped",
                "wasxfail": False,
            }
        ],
        [
            {
                "nodeid": "tests/test_m.py::test_f",
                "when": "call",
                "outcome": "passed",
                "wasxfail": True,
            }
        ],
        [
            {
                "nodeid": "tests/test_m.py::test_f",
                "when": "call",
                "outcome": "passed",
                "wasxfail": False,
            },
            {
                "nodeid": "tests/test_m.py::test_f",
                "when": "call",
                "outcome": "passed",
                "wasxfail": False,
            },
        ],
    ],
)
def test_pytest_evidence_requires_exactly_one_plain_pass_call(reports):
    nodeid = "tests/test_m.py::test_f"
    receipt = {
        "schema_version": 1,
        "nonce": "a" * 48,
        "mode": "run",
        "pytest_exit_code": 0,
        "collected": [nodeid],
        "collection_failures": 0,
        "reports": reports,
    }

    outcome, _passed = classify_receipt(
        process_returncode=0,
        receipt=receipt,
        expected_nodeids=(nodeid,),
    )

    assert outcome is PytestEvidenceOutcome.INFRA_ERROR


def test_pytest_receipt_rejects_nodeid_count_and_field_length_limits():
    nonce = "b" * 48
    base = {
        "schema_version": 1,
        "nonce": nonce,
        "mode": "collect",
        "pytest_exit_code": 0,
        "collection_failures": 0,
        "reports": [],
    }
    too_many = {
        **base,
        "collected": [f"tests/test_bulk.py::test_{index}" for index in range(MAX_NODEIDS + 1)],
    }
    too_long = {
        **base,
        "collected": ["x" * (MAX_NODEID_CHARS + 1)],
    }

    assert not valid_driver_receipt(too_many, nonce=nonce, mode="collect")
    assert not valid_driver_receipt(too_long, nonce=nonce, mode="collect")


def test_pytest_driver_rejects_oversized_receipt_before_json_decode(tmp_path):
    class OversizedReceiptBackend(TrustedLocalBackend):
        def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
            del cmd, timeout, limits
            config = json.loads(input)
            (Path(cwd) / config["report_name"]).write_bytes(
                b"x" * (MAX_RECEIPT_BYTES + 1)
            )
            return ProcResult(0, "", "", 0.01)

    result = run_driver(tmp_path, OversizedReceiptBackend(), mode="run")

    assert result.receipt is None
    assert result.receipt_sha256 == ""
    assert result.detail == "missing or inconsistent control-plane receipt"


def test_ruff_verifier(tmp_path):
    (tmp_path / "clean.py").write_text("x = 1\nprint(x)\n")
    clean = RuffVerifier().verify(
        Patch(step_id="s"), VerifyContext(workdir=tmp_path, step=_step("ruff"))
    )
    assert clean.passed

    (tmp_path / "dirty.py").write_text("import os\n")  # F401 unused import
    dirty = RuffVerifier().verify(
        Patch(step_id="s"), VerifyContext(workdir=tmp_path, step=_step("ruff"))
    )
    assert not dirty.passed
    assert dirty.score >= 1


def test_verifier_side_effects_stay_in_disposable_copy(tmp_path):
    original = tmp_path / "tests" / "test_oracle.py"
    original.parent.mkdir()
    original.write_text("ORACLE = 'canonical'\n")

    class _MutatingVerifier(Verifier):
        name = "mutating"

        def verify(self, artifact, ctx):
            (ctx.workdir / "tests" / "test_oracle.py").write_text("ORACLE = 'forged'\n")
            (ctx.workdir / "runtime-side-effect.txt").write_text("created\n")
            return Check(name=self.name, family="code", passed=True)

    check = _safe_verify(
        _MutatingVerifier(),
        Patch(step_id="s"),
        VerifyContext(workdir=tmp_path, step=_step("mutating")),
    )

    assert check.passed
    assert original.read_text() == "ORACLE = 'canonical'\n"
    assert not (tmp_path / "runtime-side-effect.txt").exists()


def test_environment_record_does_not_walk_into_parent_repository(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    assert run(["git", "init", "-q"], cwd=parent).returncode == 0
    (parent / "tracked.txt").write_text("baseline\n")
    assert run(["git", "add", "tracked.txt"], cwd=parent).returncode == 0
    assert (
        run(
            [
                "git",
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "baseline",
            ],
            cwd=parent,
        ).returncode
        == 0
    )

    nested = parent / "runs" / "workdir"
    nested.mkdir(parents=True)
    local = _env_record(nested, TrustedLocalBackend())
    docker_provenance = {
        "backend": "docker",
        "image": "python:test",
        "image_id": f"sha256:{'a' * 64}",
        "image_id_status": "available",
        "image_id_reason": None,
        "versions": {"python": "3.12.8", "pytest": "8.3.4", "ruff": "0.9.1"},
        "versions_image": f"sha256:{'a' * 64}",
        "versions_bound_to_image_id": True,
        "versions_status": "available",
        "versions_reason": None,
    }
    monkeypatch.setattr(DockerBackend, "provenance", lambda self: docker_provenance)
    docker = _env_record(parent, DockerBackend(image="python:test"))

    assert local["target_git_commit"] is None
    assert local["execution_backend"] == {
        "name": "trusted-local",
        "image": None,
    }
    assert docker["target_git_commit"] == run(
        ["git", "rev-parse", "HEAD"],
        cwd=parent,
    ).stdout.strip()
    assert docker["execution_backend"] == {
        "name": "docker",
        "image": "python:test",
        "provenance": docker_provenance,
    }
    assert "python" in local["control_plane"]


def test_environment_record_keeps_provenance_failure_non_sensitive(tmp_path):
    class BrokenProvenanceBackend(TrustedLocalBackend):
        def provenance(self):
            raise RuntimeError("/Users/example/.docker/config.json: secret-token")

    record = _env_record(tmp_path, BrokenProvenanceBackend())

    assert record["execution_backend"]["provenance"] == {
        "status": "unavailable",
        "reason": "probe_failed",
    }
    assert "secret-token" not in str(record)


def test_verdict_aggregation():
    verdict = Verdict.from_checks(
        "s",
        [
            Check(name="pytest", family="code", passed=True),
            Check(name="ruff", family="code", passed=False, detail={"summary": "1 violations"}),
        ],
    )
    assert verdict.passed is False
    assert any("ruff" in f for f in verdict.failures)
