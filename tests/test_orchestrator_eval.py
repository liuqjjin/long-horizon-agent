"""Hermetic tests for orchestrator result parsing and the eval report."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import lha.orchestrator as orchestrator
from lha.config import Config
from lha.eval import EvalReport, EvalResult, _case_resume, _eval_data_root
from lha.orchestrator import _parse, _worker_env
from lha.sandbox.base import ProcessCleanupResult


@dataclass
class _Proc:
    stdout: str
    stderr: str = ""
    returncode: int = 0


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)


def test_orchestrator_parses_result_line():
    out = (
        "some noise\n"
        "[Stats] cocoindex chatter\n"
        '__LHA_RESULT__ {"run_id": "r1", "status": "DONE", "verified": true}\n'
    )
    outcome = _parse("t.yaml", _Proc(out))
    assert outcome.status == "DONE"
    assert outcome.verified is True
    assert outcome.run_id == "r1"


def test_orchestrator_handles_missing_result():
    outcome = _parse("t.yaml", _Proc("just noise, no result line", "stderr tail"))
    assert outcome.status == "ERROR"
    assert "stderr tail" in outcome.detail


def test_orchestrator_flags_crash_after_clean_status():
    out = '__LHA_RESULT__ {"run_id": "r1", "status": "DONE", "verified": true}\n'
    outcome = _parse("t.yaml", _Proc(out, returncode=1))  # crashed after emitting
    assert outcome.status == "ERROR"
    assert outcome.run_id == "r1"


def test_batch_worker_environment_uses_a_backend_scoped_allowlist(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LHA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("LHA_SECRET_PROBE", "do-not-forward")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-forward")
    monkeypatch.setenv("GITHUB_TOKEN", "do-not-forward")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/do-not-forward")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-only")

    stub_env = _worker_env("stub")
    anthropic_env = _worker_env("anthropic")

    assert stub_env["LHA_RUNS_DIR"] == str(tmp_path / "runs")
    assert "PATH" in stub_env
    assert not {
        "LHA_SECRET_PROBE",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "SSH_AUTH_SOCK",
        "ANTHROPIC_API_KEY",
    } & stub_env.keys()
    assert anthropic_env["ANTHROPIC_API_KEY"] == "anthropic-only"


def test_batch_worker_receives_allowed_config_but_not_host_secrets(
    tmp_path, monkeypatch
):
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """
import json
import os

verified = (
    os.environ.get("LHA_RUNS_DIR") == "allowed-runs"
    and "LHA_SECRET_PROBE" not in os.environ
    and "AWS_SECRET_ACCESS_KEY" not in os.environ
)
print("__LHA_RESULT__ " + json.dumps({
    "run_id": "env-check",
    "status": "DONE",
    "verified": verified,
}), flush=True)
""".lstrip(),
    )
    monkeypatch.setattr(orchestrator.sys, "executable", str(fake_python))
    monkeypatch.setenv("LHA_RUNS_DIR", "allowed-runs")
    monkeypatch.setenv("LHA_SECRET_PROBE", "do-not-forward")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-forward")

    [outcome] = orchestrator.run_tasks(["ignored.yaml"], max_workers=1)

    assert outcome.status == "DONE"
    assert outcome.verified is True
    assert outcome.run_id == "env-check"


def _make_process_tree_worker(
    tmp_path: Path,
    *,
    emit_result: bool,
) -> tuple[Path, Path]:
    real_python = sys.executable
    marker = tmp_path / "grandchild-marker"
    ready = tmp_path / "grandchild-ready"
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "time.sleep(1.0)\n"
        f"Path({str(marker)!r}).write_text('survived')\n"
    )
    child = tmp_path / "child.py"
    child.write_text(
        "import subprocess\n"
        "import time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([{real_python!r}, {str(grandchild)!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL)\n"
        f"Path({str(ready)!r}).write_text('ready')\n"
        "time.sleep(30)\n"
    )
    worker = tmp_path / "fake-python"
    final_action = (
        'print(\'__LHA_RESULT__ {"run_id":"tree","status":"DONE",'
        '"verified":true}\', flush=True)\n'
        if emit_result
        else "time.sleep(30)\n"
    )
    _write_executable(
        worker,
        (
            "import subprocess\n"
            "import time\n"
            "from pathlib import Path\n"
            f"subprocess.Popen([{real_python!r}, {str(child)!r}], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
            "stderr=subprocess.DEVNULL)\n"
            f"ready = Path({str(ready)!r})\n"
            "deadline = time.monotonic() + 5\n"
            "while not ready.exists() and time.monotonic() < deadline:\n"
            "    time.sleep(0.01)\n"
            "if not ready.exists():\n"
            "    raise RuntimeError('grandchild did not start')\n"
            f"{final_action}"
        ),
    )
    return worker, marker


def test_batch_normal_exit_removes_worker_children_and_grandchildren(
    tmp_path, monkeypatch
):
    worker, marker = _make_process_tree_worker(tmp_path, emit_result=True)
    monkeypatch.setattr(orchestrator.sys, "executable", str(worker))

    [outcome] = orchestrator.run_tasks(["ignored.yaml"], max_workers=1, timeout=10)

    assert outcome.status == "DONE"
    time.sleep(1.5)
    assert not marker.exists(), "grandchild survived a clean worker exit"


def test_batch_timeout_removes_worker_children_and_grandchildren(
    tmp_path, monkeypatch
):
    worker, marker = _make_process_tree_worker(tmp_path, emit_result=False)
    monkeypatch.setattr(orchestrator.sys, "executable", str(worker))

    [outcome] = orchestrator.run_tasks(["ignored.yaml"], max_workers=1, timeout=0.5)

    assert outcome.status == "TIMEOUT"
    assert "timeout after 0.5s" in outcome.detail
    time.sleep(1.5)
    assert not marker.exists(), "grandchild survived a timed-out worker"


def test_batch_worker_output_is_bounded_and_fails_closed(tmp_path, monkeypatch):
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """
import os

os.write(1, b"x" * 20_000)
os.write(2, b"y" * 20_000)
print('__LHA_RESULT__ {"run_id":"noisy","status":"DONE","verified":true}')
""".lstrip(),
    )
    monkeypatch.setattr(orchestrator.sys, "executable", str(fake_python))
    monkeypatch.setattr(orchestrator, "_WORKER_OUTPUT_BYTES", 1024)

    [outcome] = orchestrator.run_tasks(["ignored.yaml"], max_workers=1)

    assert outcome.status == "ERROR"
    assert "capture limit" in outcome.detail
    assert len(outcome.detail.encode()) <= 300


def test_batch_fails_when_worker_group_cleanup_cannot_be_confirmed(
    tmp_path, monkeypatch
):
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """
print('__LHA_RESULT__ {"run_id":"unclean","status":"DONE","verified":true}')
""".lstrip(),
    )
    monkeypatch.setattr(orchestrator.sys, "executable", str(fake_python))
    monkeypatch.setattr(
        orchestrator,
        "terminate_process_group",
        lambda _process: ProcessCleanupResult(False, "group still visible"),
    )

    [outcome] = orchestrator.run_tasks(["ignored.yaml"], max_workers=1)

    assert outcome.status == "ERROR"
    assert outcome.run_id == "unclean"
    assert "cleanup could not be confirmed" in outcome.detail
    assert "group still visible" in outcome.detail


def test_batch_rejects_unsupported_process_groups_before_spawn(
    tmp_path, monkeypatch
):
    task = tmp_path / "task.yaml"
    task.write_text("task_id: must-not-run\n")

    def unexpected_spawn(*_args, **_kwargs):
        raise AssertionError("unsupported hosts must fail before spawning a worker")

    monkeypatch.setattr(
        orchestrator,
        "process_group_cleanup_supported",
        lambda: False,
    )
    monkeypatch.setattr(orchestrator, "run_bounded_process", unexpected_spawn)

    [outcome] = orchestrator.run_tasks([str(task)], max_workers=1)

    assert outcome.status == "ERROR"
    assert "requires POSIX process-group cleanup" in outcome.detail


def test_eval_report_score_and_markdown():
    report = EvalReport(
        results=[
            EvalResult("a", "issue-to-PR", True, "ok"),
            EvalResult("b", "freshness", False, "nope"),
        ]
    )
    assert report.score == "1/2"
    assert report.all_passed is False
    md = report.to_markdown()
    assert "PASS" in md and "FAIL" in md


def test_resume_eval_uses_one_persisted_budget_contract(tmp_path, monkeypatch):
    class _TTY:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr("sys.stdin", _TTY())

    def unexpected_prompt(*args, **kwargs):
        raise AssertionError("self-eval must not depend on an interactive terminal")

    monkeypatch.setattr("builtins.input", unexpected_prompt)
    result = _case_resume(
        Config(
            runs_dir=tmp_path / "runs",
            data_dir=Path.cwd() / "data",
            code_backend="null",
            use_skill_memory=False,
        )
    )

    assert result.passed, result.detail
    assert "first=AWAITING_APPROVAL" in result.detail
    assert "resumed=DONE" in result.detail


def test_quick_eval_materializes_packaged_fixtures_outside_checkout(tmp_path, monkeypatch):
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    root = _eval_data_root(Config(runs_dir=tmp_path / "runs"), quick=True)

    assert root == (tmp_path / "runs" / "eval" / "_fixtures").resolve()
    assert (root / "tasks" / "fix_average.yaml").is_file()
    assert (root / "sample_repo" / "tests" / "test_mathutils.py").is_file()
    assert (root / "papers" / "note_srgan.md").is_file()
    assert not list(root.rglob("__pycache__"))


def test_full_eval_outside_checkout_fails_with_an_explicit_boundary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        _eval_data_root(Config(runs_dir=tmp_path / "runs"), quick=False)
    except FileNotFoundError as error:
        assert "--quick" in str(error)
    else:
        raise AssertionError("full installed-package eval unexpectedly found checkout fixtures")


def test_checkout_eval_keeps_using_checkout_data():
    root = _eval_data_root(Config(), quick=True)
    assert root == (Path.cwd() / "data").resolve()


def test_packaged_quick_fixtures_match_the_checkout_corpus():
    packaged = resources.files("lha.resources").joinpath("eval")
    for relative in (
        "tasks/fix_average.yaml",
        "sample_repo/mathutils.py",
        "sample_repo/pyproject.toml",
        "sample_repo/tests/test_mathutils.py",
        "papers/note_srgan.md",
    ):
        assert packaged.joinpath(relative).read_bytes() == (Path("data") / relative).read_bytes()
