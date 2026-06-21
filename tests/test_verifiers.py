"""Each verifier family produces a well-formed Verdict/Check."""

from __future__ import annotations

from pathlib import Path

from lha.artifacts import Patch, Step
from lha.verifiers import VerifyContext
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
    repo = _repo(tmp_path, "def f():\n    return 1\n", "from m import f\n\n\ndef test_f():\n    assert f() == 1\n")
    check = PytestVerifier().verify(Patch(step_id="s"), VerifyContext(workdir=repo, step=_step("pytest")))
    assert check.passed
    assert check.score == 1.0


def test_pytest_verifier_fail(tmp_path):
    repo = _repo(tmp_path, "def f():\n    return 2\n", "from m import f\n\n\ndef test_f():\n    assert f() == 1\n")
    check = PytestVerifier().verify(Patch(step_id="s"), VerifyContext(workdir=repo, step=_step("pytest")))
    assert not check.passed
    assert check.detail["failing"]


def test_ruff_verifier(tmp_path):
    (tmp_path / "clean.py").write_text("x = 1\nprint(x)\n")
    clean = RuffVerifier().verify(Patch(step_id="s"), VerifyContext(workdir=tmp_path, step=_step("ruff")))
    assert clean.passed

    (tmp_path / "dirty.py").write_text("import os\n")  # F401 unused import
    dirty = RuffVerifier().verify(Patch(step_id="s"), VerifyContext(workdir=tmp_path, step=_step("ruff")))
    assert not dirty.passed
    assert dirty.score >= 1


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
