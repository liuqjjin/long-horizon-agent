"""Deterministic tests for the paired verification ablation — no network.

The real ablation runs a live LLM, but every mechanism is pinned here with injected
fake backends: the paired trust/gate contrast, tamper-proof source-only patching,
the repair lift, transient-error handling, aggregation, and resumability.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from lha.ablation import (
    CONDITIONS,
    AblationProvenance,
    AblationReport,
    ConditionStats,
    PytestResult,
    RunRecord,
    ScoreOutcome,
    _aggregate,
    _classify_scorer_receipt,
    _sanitize,
    _score,
    run_ablation,
)
from lha.artifacts import Patch
from lha.config import Config
from lha.llm.base import LLMClient
from lha.llm.claude_cli import ClaudeCLIClient
from lha.llm.trace import TracedLLM
from lha.sandbox import TrustedLocalBackend
from lha.tasks.spec import TaskSpec

_PYPROJECT = (
    '[project]\nname = "buggy"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n\n'
    '[tool.pytest.ini_options]\npythonpath = ["."]\n'
)
_TEST = "from m import f\n\n\ndef test_f():\n    assert f() == 2\n"
_FORMAL_TEST_REMOTES: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _translate_formal_test_remotes(monkeypatch, request, tmp_path):
    """Keep production URLs public while using local bare remotes in tests."""
    import lha.ablation as abl

    original_anonymous = abl._formal_anonymous_git_output
    original_run = abl.run
    original_helper = abl._formal_git_credential_helper

    def translated(values):
        return [
            _FORMAL_TEST_REMOTES.get(value, value)
            for value in values
        ]

    def anonymous(git_path, arguments, **kwargs):
        return original_anonymous(
            git_path,
            translated(arguments),
            **kwargs,
        )

    def run(command, **kwargs):
        return original_run(translated(command), **kwargs)

    def credential_helper(host, *, expected=None):
        if expected is not None:
            return expected
        return original_helper(host)

    @contextmanager
    def authenticated_env(helper):
        home = tmp_path / "formal-auth-home"
        home.mkdir(exist_ok=True)
        env = abl._git_control_env()
        env.update(
            {
                "HOME": str(home),
                "GH_CONFIG_DIR": str(home),
                "GH_HOST": helper.host,
                "GH_TOKEN": "test-token",
            }
        )
        try:
            yield env
        finally:
            env.pop("GH_TOKEN", None)

    monkeypatch.setattr(abl, "_formal_anonymous_git_output", anonymous)
    monkeypatch.setattr(abl, "run", run)
    if request.node.name != "test_formal_authenticated_git_env_is_disposable":
        monkeypatch.setattr(
            abl,
            "_git_authenticated_push_env",
            authenticated_env,
        )
    if not request.node.name.startswith(
        "test_formal_git_credential_preflight_"
    ):
        monkeypatch.setattr(
            abl,
            "_preflight_formal_git_credential_helper",
            lambda *_args, **_kwargs: {
                "host": "github.com",
                "fields": ("host", "password", "protocol", "username"),
            },
        )
    if request.node.name != "test_formal_git_helper_identity_uses_disposable_home":
        monkeypatch.setattr(
            abl,
            "_formal_git_credential_helper",
            credential_helper,
        )


def _hold_formal_output_lock(path: str, ready, release) -> None:
    import lha.ablation as abl

    announced = False
    try:
        with abl._formal_ablation_lock(Path(path)):
            ready.put(("locked", ""))
            announced = True
            if not release.wait(20):
                raise TimeoutError("test did not release the formal output lock")
    except BaseException as error:
        if not announced:
            ready.put(("error", f"{type(error).__name__}: {error}"))
        raise


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        [str(Path(shutil.which("git") or "git").resolve()), *arguments],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _formal_attempt_repository(
    root: Path,
    *,
    event: str,
    output_path: str,
    tracked: bool = True,
):
    import lha.ablation as abl
    from lha.ablation_attempts import (
        FormalAblationAttemptRegistry,
        FormalAblationProtocol,
        FormalCodexClientConfig,
        FormalGitCredentialHelper,
        RegisteredAttempt,
        UnregisteredRunRecorded,
        formal_ablation_attempt_registry_bytes,
        formal_ablation_protocol_sha256,
        formal_codex_client_sha256,
    )

    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "LHA Test")
    _git(root, "config", "user.email", "lha@example.invalid")
    witness_remote = (root.parent / f"{root.name}-formal-witness.git").resolve()
    witness_url = f"https://github.com/example/{root.name}-formal-witness.git"
    _FORMAL_TEST_REMOTES[witness_url] = str(witness_remote)
    _git(root, "init", "--bare", "-q", str(witness_remote))
    _git(root, "remote", "add", "formal-witness", witness_url)
    (root / ".gitignore").write_text("runs/\n")
    (root / "src" / "lha").mkdir(parents=True)
    (root / "src" / "lha" / "runtime.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = \"formal-test\"\nversion = \"0.0.0\"\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / ".python-version").write_text("3.11\n", encoding="utf-8")
    manifest_entries = []
    for index in range(17):
        name = f"task_{index:02d}"
        corpus_relative = f"data/bench/{name}"
        task_relative = f"data/tasks/{name}.yaml"
        corpus = root / corpus_relative
        corpus.mkdir(parents=True)
        (corpus / "module.py").write_text(
            f"VALUE = {index}\n",
            encoding="utf-8",
        )
        task = root / task_relative
        task.parent.mkdir(parents=True, exist_ok=True)
        task.write_text(
            "\n".join(
                (
                    "kind: issue_to_pr",
                    f"title: fix {name}",
                    f"description: repair {name}",
                    f"target_repo: {corpus_relative}",
                    "inputs: {}",
                    "success:",
                    '  - "pytest passes"',
                    "",
                )
            ),
            encoding="utf-8",
        )
        manifest_entries.append(
            {
                "name": name,
                "task_path": task_relative,
                "task_sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
                "corpus_path": corpus_relative,
                "corpus_sha256": abl._repo_digest(corpus),
            }
        )
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "corpus")
    corpus_commit = _git(root, "rev-parse", "HEAD")
    manifest = {
        "schema_version": 1,
        "benchmark": "lha-verification-ablation",
        "repetitions": 12,
        "corpus_commit": corpus_commit,
        "tasks": manifest_entries,
    }
    manifest_path = root / "benchmarks" / "formal_ablation_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    _git(root, "add", "benchmarks/formal_ablation_manifest.json")
    _git(root, "commit", "-qm", "source")
    source_commit = _git(root, "rev-parse", "HEAD")

    client_config = FormalCodexClientConfig(
        max_retries=2,
        timeout_s=300.0,
        retry_backoff_s=1.0,
    )
    helper_path = "/opt/homebrew/bin/gh"
    credential_helper = FormalGitCredentialHelper(
        host="github.com",
        executable_path=helper_path,
        executable_sha256="8" * 64,
        version="gh version 2.92.0",
        command=f"!{helper_path} auth git-credential",
    )
    protocol = FormalAblationProtocol(
        source_commit=source_commit,
        source_tree_sha256=abl._source_tree_digest(
            abl._source_file_digests(root / "src" / "lha")
        ),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        model="model-x",
        reasoning_effort="medium",
        docker_image_id="sha256:" + "c" * 64,
        codex_cli_version="codex-cli 0.141.0",
        codex_cli_executable_sha256="9" * 64,
        codex_client=client_config,
        codex_client_sha256=formal_codex_client_sha256(client_config),
        witness_credential_helper=(
            credential_helper if event == "REGISTERED" else None
        ),
    )
    event_type = {
        "REGISTERED": RegisteredAttempt,
        "UNREGISTERED_RUN_RECORDED": UnregisteredRunRecorded,
    }[event]
    common = {
        "attempt_id": "a" * 64,
        "protocol_sha256": formal_ablation_protocol_sha256(protocol),
        "source_commit": protocol.source_commit,
        "source_tree_sha256": protocol.source_tree_sha256,
        "manifest_sha256": protocol.manifest_sha256,
        "output_path": output_path,
        "model": protocol.model,
        "reasoning_effort": protocol.reasoning_effort,
        "docker_image_id": protocol.docker_image_id,
        "codex_cli_version": protocol.codex_cli_version,
        "codex_cli_executable_sha256": protocol.codex_cli_executable_sha256,
        "codex_client": protocol.codex_client,
        "codex_client_sha256": protocol.codex_client_sha256,
    }
    if event == "REGISTERED":
        attempt_event = event_type(
            **common,
            witness_credential_helper=credential_helper,
            witness_remote_name="formal-witness",
            witness_remote_url=witness_url,
            registered_at="2026-07-28T12:00:00+08:00",
        )
    else:
        attempt_event = event_type(
            **common,
            recorded_at="2026-07-28T12:00:00+08:00",
            reason="旧运行只作事后披露",
            published_report_path=(
                "benchmarks/formal_ablation_history/"
                f"{protocol.source_commit}/ablation_report.json"
            ),
            report_sha256="e" * 64,
            report_fingerprint="f" * 64,
            scheduled_cells=204,
            usable_cells=204,
            error_cells=0,
            trust_delivered_correct=190,
            trust_delivered_wrong=14,
            gate_delivered_correct=190,
            gate_delivered_wrong=0,
            gate_intercepted_wrong=14,
            gate_rejected_correct=0,
            verify_delivered_correct=204,
            verify_delivered_wrong=0,
            verify_not_delivered=0,
        )
    registry = FormalAblationAttemptRegistry(events=(attempt_event,))
    registry_path = root / "benchmarks" / "formal_ablation_attempts.json"
    registry_path.parent.mkdir(exist_ok=True)
    registry_path.write_bytes(formal_ablation_attempt_registry_bytes(registry))
    if tracked:
        _git(root, "add", "benchmarks/formal_ablation_attempts.json")
        _git(root, "commit", "-qm", "register formal attempt")
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "symbolic-ref", "--short", "HEAD")
    _git(
        root,
        "push",
        "-q",
        str(witness_remote),
        f"HEAD:refs/heads/{branch}",
    )
    binding = abl._FormalCorpusBinding(
        path="benchmarks/formal_ablation_manifest.json",
        sha256=protocol.manifest_sha256,
        preregistration_commit=head,
        git_executable={"path": str(Path(shutil.which("git") or "git").resolve())},
    )
    return binding, protocol, registry_path


class _FakeDockerIdentity:
    path = "/usr/bin/docker"

    def as_provenance(self):
        return {
            "path": self.path,
            "sha256": "d" * 64,
            "size_bytes": 1,
            "trusted_install": True,
        }


def _run_formal_until_registration(
    repo: Path,
    binding,
    out: Path,
    llm: LLMClient,
    monkeypatch,
    *,
    inject_llm: bool = False,
):
    import lha.ablation as abl
    from lha.llm.codex_cli import CodexCLIClient

    client = CodexCLIClient(
        model="model-x",
        reasoning_effort="medium",
        no_tools=True,
    )

    def preflight() -> None:
        client._version = "codex-cli 0.141.0"
        client._cli_identity = (
            "/usr/bin/codex",
            1,
            2,
            3,
            4,
            "9" * 64,
            False,
        )
        client._verified_permission_roots.add(("/tmp", "lha-read"))

    monkeypatch.setattr(
        abl,
        "_prepare_formal_corpus_binding",
        lambda *args, **kwargs: binding,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)
    monkeypatch.setattr(
        abl,
        "resolve_docker_executable",
        lambda *args, **kwargs: _FakeDockerIdentity(),
    )
    monkeypatch.setattr(
        abl,
        "_inspect_docker_image_id",
        lambda *args, **kwargs: "sha256:" + "c" * 64,
    )
    monkeypatch.setattr(client, "preflight", preflight)
    monkeypatch.setattr(
        abl,
        "make_formal_codex_client",
        lambda *args, **kwargs: client,
    )
    return run_ablation(
        _base(repo),
        [],
        llm="codex_cli",
        model="model-x",
        reps=1,
        out_dir=out,
        llm_client=llm if inject_llm else None,
        scorer_backend="docker",
    )


def _repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(_PYPROJECT)
    (root / "m.py").write_text("def f():\n    return 1\n")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_m.py").write_text(_TEST)
    return root


def _task(tmp_path: Path, src: Path) -> str:
    y = tmp_path / "task.yaml"
    y.write_text(
        "kind: issue_to_pr\n"
        'title: "f should return 2"\n'
        'description: "f() returns the wrong value"\n'
        f"target_repo: {src}\n"
        "inputs:\n  context_query: f\n"
        'success:\n  - "pytest passes"\n'
    )
    return str(y)


class _FixedLLM(LLMClient):
    """Always returns the same body for m.py (and optionally a tampered test)."""

    def __init__(self, value: int, tamper: bool = False):
        self.value = value
        self.tamper = tamper
        self.calls = 0

    def complete(self, system: str, prompt: str) -> str:
        return ""

    def propose_patch(self, step, bundle, workdir) -> Patch:
        self.calls += 1
        fc = {"m.py": f"def f():\n    return {self.value}\n"}
        if self.tamper:
            fc["tests/test_m.py"] = "def test_f():\n    assert True\n"
        return Patch(step_id=step.step_id, file_contents=fc, touched_files=list(fc))


class _RepairLLM(LLMClient):
    """Wrong first attempt, correct on the repair — exercises the repair loop."""

    def __init__(self):
        self.n = 0

    def complete(self, system: str, prompt: str) -> str:
        return ""

    def propose_patch(self, step, bundle, workdir) -> Patch:
        self.n += 1
        value = 2 if self.n >= 2 else 3
        return Patch(
            step_id=step.step_id,
            file_contents={"m.py": f"def f():\n    return {value}\n"},
            touched_files=["m.py"],
        )


class _AuditedCodexLLM(LLMClient):
    name = "codex_cli"

    def __init__(self):
        self.calls = 0
        self.model = "model-x"
        self.reasoning_effort = "low"
        self.no_tools = True
        self.sandbox_mode = "read-only"
        self.permission_model = "profile"
        self.permission_profile = "lha-read"
        self.credential_barrier = "verified"
        self.externally_sandboxed = False
        self.max_retries = 0
        self.retry_backoff_s = 0.0
        self._version = "codex-cli 1.0"
        self.last_call = None
        self.last_usage = None

    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        summary = {
            "total_events": 4,
            "events": {
                "thread.started": 1,
                "turn.started": 1,
                "item.completed": 1,
                "turn.completed": 1,
            },
            "items": {"agent_message": 1},
            "invalid_json_lines": 0,
        }
        self.last_call = {
            "status": "succeeded",
            "cli_version": self._version,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "sandbox_mode": self.sandbox_mode,
            "permission_model": self.permission_model,
            "permission_profile": self.permission_profile,
            "credential_barrier": self.credential_barrier,
            "cli_executable_sha256": "9" * 64,
            "cli_executable_trusted": False,
            "externally_sandboxed": False,
            "retries": 0,
            "attempt_count": 1,
            "duration_s": 0.1,
            "event_summary": summary,
            "attempts": [
                {
                    "attempt": 1,
                    "status": "succeeded",
                    "duration_s": 0.1,
                    "event_summary": summary,
                }
            ],
        }
        self.last_usage = {
            "input_tokens": 10,
            "cached_input_tokens": 0,
            "output_tokens": 5,
            "cost_usd": None,
            "model": self.model,
        }
        return "### m.py\n```python\ndef f():\n    return 2\n```"


class _NonRetryableCallError(RuntimeError):
    retryable = False


class _FailingAuditedCodexLLM(_AuditedCodexLLM):
    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        summary = {
            "total_events": 3,
            "events": {
                "thread.started": 1,
                "turn.started": 1,
                "turn.failed": 1,
            },
            "items": {},
            "invalid_json_lines": 0,
        }
        self.last_call = {
            "status": "failed",
            "cli_version": self._version,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "sandbox_mode": self.sandbox_mode,
            "permission_model": self.permission_model,
            "permission_profile": self.permission_profile,
            "credential_barrier": self.credential_barrier,
            "cli_executable_sha256": "9" * 64,
            "cli_executable_trusted": False,
            "externally_sandboxed": False,
            "retries": 0,
            "attempt_count": 1,
            "duration_s": 0.1,
            "event_summary": summary,
            "error_type": "_NonRetryableCallError",
            "retryable": False,
            "attempts": [
                {
                    "attempt": 1,
                    "status": "failed",
                    "duration_s": 0.1,
                    "error_type": "_NonRetryableCallError",
                    "event_summary": summary,
                }
            ],
        }
        self.last_usage = {
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "cost_usd": None,
            "model": self.model,
        }
        raise _NonRetryableCallError("protocol rejected")


class _SuccessThenFailAuditedCodexLLM(_FailingAuditedCodexLLM):
    def complete(self, system: str, prompt: str) -> str:
        if self.calls == 0:
            response = _AuditedCodexLLM.complete(self, system, prompt)
            return response.replace("return 2", "return 3")
        return _FailingAuditedCodexLLM.complete(self, system, prompt)


def _base(tmp_path: Path) -> Config:
    return Config(runs_dir=tmp_path / "runs", data_dir=tmp_path / "nodata")


def _run(tmp_path, llm, out="out"):
    src = _repo(tmp_path / "src")
    return run_ablation(
        _base(tmp_path),
        [_task(tmp_path, src)],
        llm="stub",
        reps=1,
        out_dir=tmp_path / out,
        llm_client=llm,
    )


def _by_cond(report) -> dict[str, RunRecord]:
    return {r.condition: r for r in report.records}


def test_programmatic_ablation_default_matches_cli_backend():
    assert run_ablation.__kwdefaults__["llm"] == "codex_cli"


def test_client_operation_lease_binding_walks_wrapper_chain(tmp_path):
    import lha.ablation as abl

    class _LeaseClient:
        operation_lease_dir = None

        def set_operation_lease_dir(self, path):
            self.operation_lease_dir = Path(path).resolve()

    inner = _LeaseClient()
    outer = type("Wrapper", (), {"inner": inner})()

    assert abl._bind_client_operation_lease(outer, tmp_path)
    assert inner.operation_lease_dir == tmp_path.resolve()


def test_docker_image_probe_returns_versioned_receipt(tmp_path):
    import lha.ablation as abl
    from lha.tools.shell import ProcResult

    image_id = "sha256:" + "a" * 64

    class _ProbeBackend:
        name = "docker"
        image = image_id

        @staticmethod
        def python():
            return "python"

        @staticmethod
        def run(cmd, *, cwd, timeout=300.0, input=None, limits=None):
            assert cmd[:3] == ["python", "-I", "-c"]
            assert cwd == tmp_path
            payload = {
                "python_version": "3.11.9",
                "pytest_version": "9.1.1",
                "pytest_json_report_version": "1.5.0",
            }
            return ProcResult(
                0,
                abl._DOCKER_IMAGE_PROBE_MARKER + json.dumps(payload) + "\n",
                "",
                0.01,
            )

    assert abl._probe_docker_image(
        _ProbeBackend(),
        image_id=image_id,
        workdir=tmp_path,
    ) == {
        "schema_version": 1,
        "image_id": image_id,
        "network": "none",
        "minimal_pytest": "passed",
        "python_version": "3.11.9",
        "pytest_version": "9.1.1",
        "pytest_json_report_version": "1.5.0",
    }


def test_experimental_claude_cli_cannot_produce_ablation_evidence(tmp_path):
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="experimental"):
        run_ablation(
            _base(tmp_path),
            [task],
            llm="claude_cli",
            reps=1,
            out_dir=out,
            llm_client=_FixedLLM(2),
        )

    assert not out.exists()


@pytest.mark.parametrize("wrapped", [False, True])
def test_injected_claude_client_cannot_bypass_ablation_gate(
    wrapped: bool,
    tmp_path,
):
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    out = tmp_path / "out"
    client = ClaudeCLIClient(cli_path="must-not-run")
    injected = TracedLLM(client) if wrapped else client

    with pytest.raises(ValueError, match="experimental"):
        run_ablation(
            _base(tmp_path),
            [task],
            llm="stub",
            reps=1,
            out_dir=out,
            llm_client=injected,
        )

    assert not out.exists()


# --- patch sanitization (tamper-proofing) -----------------------------------
def test_sanitize_keeps_only_source():
    p = Patch(
        step_id="s",
        file_contents={
            "m.py": "x",
            "tests/test_m.py": "y",
            "conftest.py": "z",
            "pyproject.toml": "w",
        },
    )
    s = _sanitize(p)
    assert set(s.file_contents) == {"m.py"}
    assert s.touched_files == ["m.py"]


# --- the paired trust/gate paths --------------------------------------------
def test_trust_false_gate_refuses_on_wrong_fix(tmp_path):
    rec = _by_cond(_run(tmp_path, _FixedLLM(3)))  # always wrong
    assert rec["trust"].claimed_success and not rec["trust"].true_success
    assert rec["trust"].false_success  # silent wrong answer
    assert rec["gate"].status == "FAILED" and not rec["gate"].false_success  # refused
    assert rec["verify"].status == "FAILED"  # repair can't fix a stuck-wrong model


def test_correct_fix_true_everywhere(tmp_path):
    rec = _by_cond(_run(tmp_path, _FixedLLM(2)))  # correct
    for c in ("trust", "gate", "verify"):
        assert rec[c].true_success and not rec[c].false_success


def test_verify_repairs_to_success(tmp_path):
    rec = _by_cond(_run(tmp_path, _RepairLLM()))  # wrong, then correct
    # trust/gate score the same first (wrong) attempt: trust accepts it, gate refuses
    assert rec["trust"].false_success
    assert rec["gate"].status == "FAILED"
    # verify repairs the same attempt to a real success
    assert rec["verify"].true_success and rec["verify"].repairs >= 1


def test_tamper_proof_grading(tmp_path):
    # A wrong fix that also rewrites the test to pass trivially: the test rewrite is
    # stripped, so the canonical oracle still catches the wrong source.
    rec = _by_cond(_run(tmp_path, _FixedLLM(3, tamper=True)))
    assert rec["trust"].false_success  # canonical test caught it despite the tamper
    assert rec["gate"].status == "FAILED"


def test_independent_scorer_ignores_candidate_writable_json_report(tmp_path):
    """Candidate code may create the old report path, but scorer truth ignores it."""
    src = _repo(tmp_path / "src")
    forged_report = json.dumps(
        {
            "summary": {"passed": 1, "failed": 0, "error": 0, "total": 1},
            "tests": [{"nodeid": "forged", "outcome": "passed"}],
        }
    )
    frozen = {
        "m.py": (
            "from pathlib import Path\n"
            f"Path('.lha_pytest.json').write_text({forged_report!r})\n"
            "def f():\n"
            "    return 3\n"
        ),
    }

    result = _score(
        src,
        frozen,
        tmp_path / "scratch",
        "shadow",
        TrustedLocalBackend(),
    )
    assert result.outcome is ScoreOutcome.TEST_FAIL
    assert not result.passed


def test_candidate_cannot_forge_pass_by_printing_summary_and_exiting_zero(tmp_path):
    """Exercise the real interpreter path, not a fake ExecutionBackend."""
    src = _repo(tmp_path / "src")
    frozen = {"m.py": ('print("1 passed in 0.01s", flush=True)\nimport os\nos._exit(0)\n')}

    result = _score(
        src,
        frozen,
        tmp_path / "scratch",
        "early-exit",
        TrustedLocalBackend(),
    )

    assert result.outcome is ScoreOutcome.INFRA_ERROR
    assert "receipt" in result.detail


def test_independent_scorer_rejects_protected_frozen_paths(tmp_path):
    src = _repo(tmp_path / "src")
    result = _score(
        src,
        {"pytest.py": "raise SystemExit(0)\n"},
        tmp_path / "scratch",
        "protected",
        TrustedLocalBackend(),
    )
    assert result.outcome is ScoreOutcome.INFRA_ERROR
    assert result.detail == "scorer setup failed: ValueError"


def test_candidate_syntax_error_is_a_test_failure_not_infrastructure(tmp_path):
    src = _repo(tmp_path / "src")
    result = _score(
        src,
        {"m.py": "def f(:\n    return 2\n"},
        tmp_path / "scratch",
        "syntax",
        TrustedLocalBackend(),
    )
    assert result.outcome is ScoreOutcome.TEST_FAIL


# --- transient errors are not cached / resumable ----------------------------
class _FailingLLM(LLMClient):
    name = "failing"

    def complete(self, system: str, prompt: str) -> str:
        raise RuntimeError("backend down")

    def propose_patch(self, step, bundle, workdir) -> Patch:
        raise RuntimeError("backend down")


def test_transient_errors_excluded_and_not_cached(tmp_path, monkeypatch):
    import lha.ablation as abl

    monkeypatch.setattr(abl, "_LLM_RETRIES", 1)
    monkeypatch.setattr(abl.time, "sleep", lambda *a: None)
    src = _repo(tmp_path / "src")
    out = tmp_path / "out"
    rep = run_ablation(
        _base(tmp_path), [_task(tmp_path, src)], reps=1, out_dir=out, llm_client=_FailingLLM()
    )
    assert all(r.status == "ERROR" for r in rep.records)
    # ERROR cells are NOT cached, so a later good run recomputes them.
    assert not (out / "results" / "task__r0.json").exists()


def test_failed_codex_receipt_is_referenced_and_error_seal_is_resumable(tmp_path):
    import lha.ablation as abl

    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    llm = _FailingAuditedCodexLLM()

    first = run_ablation(
        _base(tmp_path),
        [task],
        llm="codex_cli",
        model=llm.model,
        reps=1,
        out_dir=out,
        llm_client=llm,
    )

    assert llm.calls == 1
    assert all(
        record.status == "ERROR"
        and record.scorer_outcome == "INFRA_ERROR"
        and not record.artifact_sha256
        and not record.scorer_evidence_sha256
        for record in first.records
    )
    cache = json.loads((out / "results" / "task__r0.json").read_text())
    assert cache["schema_version"] == abl._CACHE_SCHEMA
    assert cache["terminal_error"] is True
    assert len(cache["llm_call_receipts"]) == 1
    raw = json.loads((out / "ablation_report.json").read_text())
    assert raw["llm_call_receipt_store"]["count"] == 1
    assert len(raw["llm_calls"]) == 1
    reference = raw["llm_calls"][0]
    assert reference["cache_hit"] is False
    assert reference["receipt_sha256"] == cache["llm_call_receipts"][0]
    receipt = json.loads(
        (out / "llm_call_receipts" / f"{reference['receipt_sha256']}.json").read_text()
    )
    assert receipt["call"]["status"] == "failed"
    assert receipt["binding"]["response_sha256"] is None
    assert receipt["binding"]["patch_sha256"] is None
    assert receipt["binding"]["result_artifact_sha256"] is None

    second = run_ablation(
        _base(tmp_path),
        [task],
        llm="codex_cli",
        model=llm.model,
        reps=1,
        out_dir=out,
        llm_client=llm,
    )

    assert llm.calls == 1
    assert all(record.status == "ERROR" for record in second.records)
    assert second.llm_calls == [{**reference, "cache_hit": True}]


def test_formal_error_seal_cannot_be_resumed_or_reused(tmp_path):
    import lha.ablation as abl

    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = TaskSpec.from_file(_task(tmp_path, src))
    task.inputs["_name"] = "task"
    llm = _FailingAuditedCodexLLM()

    def run_cell():
        return abl._run_cell(
            abl._PromptAuditClient(llm),
            src,
            task,
            0,
            out,
            "f" * 64,
            abl._repo_digest(src),
            "a" * 64,
            TrustedLocalBackend(),
            TrustedLocalBackend(),
            "trusted-local",
            None,
            [],
            True,
            1,
            True,
        )

    first = run_cell()
    assert llm.calls == 1
    assert all(record.status == "ERROR" for record in first)
    marker = out / "results" / "task__r0.started.json"
    cache = out / "results" / "task__r0.json"
    assert marker.exists() and cache.exists()

    with pytest.raises(RuntimeError, match="does not resume or reuse"):
        run_cell()
    assert llm.calls == 1

    cache.unlink()
    with pytest.raises(RuntimeError, match="does not resume or reuse"):
        run_cell()
    assert llm.calls == 1


def test_formal_cache_without_start_marker_is_not_recomputed(tmp_path):
    import lha.ablation as abl

    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = TaskSpec.from_file(_task(tmp_path, src))
    task.inputs["_name"] = "task"
    llm = _FailingAuditedCodexLLM()
    (out / "results").mkdir(parents=True)
    (out / "results" / "task__r0.json").write_text("{}")

    with pytest.raises(RuntimeError, match="does not resume or reuse"):
        abl._run_cell(
            abl._PromptAuditClient(llm),
            src,
            task,
            0,
            out,
            "f" * 64,
            abl._repo_digest(src),
            "a" * 64,
            TrustedLocalBackend(),
            TrustedLocalBackend(),
            "trusted-local",
            None,
            [],
            True,
            1,
            True,
        )
    assert llm.calls == 0


def test_formal_output_lock_is_cross_process_and_precedes_run_setup(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    out = tmp_path / "out"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    release = context.Event()
    holder = context.Process(
        target=_hold_formal_output_lock,
        args=(str(out), ready, release),
    )
    holder.start()
    try:
        assert ready.get(timeout=10) == ("locked", "")
        run_setup_calls = 0

        def forbidden_run_setup(*args, **kwargs):
            nonlocal run_setup_calls
            run_setup_calls += 1
            raise AssertionError("formal setup ran before the output lock")

        monkeypatch.setattr(
            abl,
            "_prepare_formal_corpus_binding",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(abl, "_run_ablation_with_binding", forbidden_run_setup)

        with pytest.raises(RuntimeError, match="already active"):
            run_ablation(
                _base(tmp_path),
                [],
                llm="codex_cli",
                model="model-x",
                reps=1,
                out_dir=out,
                llm_client=None,
                scorer_backend="docker",
            )
        assert run_setup_calls == 0
    finally:
        release.set()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert holder.exitcode == 0


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "fifo"])
def test_formal_output_lock_rejects_link_and_nonregular_inodes(tmp_path, attack):
    import lha.ablation as abl

    out = tmp_path / "out"
    out.mkdir()
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("unchanged")
    lock = out / abl._FORMAL_OUTPUT_LOCK_NAME
    if attack == "symlink":
        lock.symlink_to(sentinel)
    elif attack == "hardlink":
        os.link(sentinel, lock)
    else:
        os.mkfifo(lock)

    with pytest.raises(RuntimeError, match="output lock is unsafe"):
        with abl._formal_ablation_lock(out):
            raise AssertionError("unsafe lock inode was accepted")
    assert sentinel.read_text() == "unchanged"


def test_formal_output_path_rejects_symbolic_link_component(tmp_path):
    import lha.ablation as abl

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(RuntimeError, match="output directory is unsafe"):
        with abl._formal_ablation_lock(alias):
            raise AssertionError("symbolic output directory was accepted")


def test_formal_output_path_rejects_symbolic_link_parent(tmp_path):
    import lha.ablation as abl

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(RuntimeError, match="output directory is unsafe"):
        with abl._formal_ablation_lock(alias_parent / "out"):
            raise AssertionError("symbolic parent directory was accepted")


def test_formal_evidence_path_rejects_intermediate_symlink(tmp_path):
    import lha.ablation as abl

    root = tmp_path / "repo"
    real = root / "real"
    real.mkdir(parents=True)
    (real / "task.yaml").write_text("kind: issue_to_pr\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "alias").symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain a symlink"):
        abl._repo_relative_evidence_path(
            root,
            "data/alias/task.yaml",
            kind="task",
        )


def test_formal_attempt_binding_accepts_committed_matching_registration(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl
    from lha.ablation_attempts import formal_ablation_protocol_sha256

    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    corpus_binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt_binding = abl._bind_formal_attempt(
            formal_corpus=corpus_binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
            codex_client=protocol.codex_client,
        )

    assert attempt_binding.attempt_id == attempt_id
    assert attempt_binding.registration_commit == corpus_binding.preregistration_commit
    assert attempt_binding.protocol_sha256 == formal_ablation_protocol_sha256(
        protocol
    )


@pytest.mark.parametrize(
    "index_flag",
    ["--assume-unchanged", "--skip-worktree"],
)
def test_formal_head_binding_rejects_hidden_source_drift(
    tmp_path,
    monkeypatch,
    index_flag,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    source = repo / "src" / "lha" / "runtime.py"
    _git(repo, "update-index", index_flag, "--", "src/lha/runtime.py")
    source.write_text("VALUE = 99\n", encoding="utf-8")
    assert _git(repo, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with pytest.raises(RuntimeError, match="source bytes differ"):
        abl._revalidate_formal_checkout(binding)


def test_formal_head_binding_rejects_hidden_executable_bit_drift(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    relative = "src/lha/runtime.py"
    source = repo / relative
    _git(repo, "update-index", "--assume-unchanged", "--", relative)
    source.chmod(source.stat().st_mode | 0o111)
    assert _git(repo, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with pytest.raises(RuntimeError, match="file mode differs"):
        abl._revalidate_formal_checkout(binding)


def test_formal_completion_rejects_hidden_source_drift(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl
    import lha.formal_attempt_cli as commands
    from lha.ablation_attempts import parse_formal_ablation_attempt_registry

    repo = tmp_path / "repo"
    binding, _protocol, registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    registry = parse_formal_ablation_attempt_registry(registry_path.read_bytes())
    registration = registry.open_registration()
    assert registration is not None
    source = repo / "src" / "lha" / "runtime.py"
    _git(
        repo,
        "update-index",
        "--skip-worktree",
        "--",
        "src/lha/runtime.py",
    )
    source.write_text("VALUE = 99\n", encoding="utf-8")
    assert _git(repo, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with pytest.raises(
        commands.FormalAttemptCommandError,
        match="changed before completion",
    ):
        commands._validate_registration_checkout(
            repo,
            registration_head=binding.preregistration_commit,
            registration=registration,
        )


def test_formal_head_binding_rejects_excluded_untracked_source(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text("src/lha/hidden.py\n", encoding="utf-8")
    (repo / "src" / "lha" / "hidden.py").write_text(
        "HIDDEN = True\n",
        encoding="utf-8",
    )
    assert _git(repo, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with pytest.raises(RuntimeError, match="source bytes differ"):
        abl._revalidate_formal_checkout(binding)


def test_formal_head_binding_rejects_excluded_untracked_manifest(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    manifest = repo / "benchmarks" / "formal_ablation_manifest.json"
    payload = manifest.read_bytes()
    _git(repo, "rm", "-q", "benchmarks/formal_ablation_manifest.json")
    _git(repo, "commit", "-qm", "remove manifest from head")
    (repo / ".git" / "info" / "exclude").write_text(
        "benchmarks/formal_ablation_manifest.json\n",
        encoding="utf-8",
    )
    manifest.write_bytes(payload)
    assert _git(repo, "status", "--porcelain=v1") == ""
    hidden_binding = abl._FormalCorpusBinding(
        path=binding.path,
        sha256=binding.sha256,
        preregistration_commit=_git(repo, "rev-parse", "HEAD"),
        git_executable=binding.git_executable,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with pytest.raises(RuntimeError, match="trusted HEAD blob"):
        abl._revalidate_formal_checkout(hidden_binding)


@pytest.mark.parametrize(
    ("relative", "replacement", "message"),
    [
        (
            "benchmarks/formal_ablation_manifest.json",
            lambda current: current + "\n",
            "manifest changed",
        ),
        (
            "data/tasks/task_00.yaml",
            lambda current: current + "# hidden task drift\n",
            "formal corpus bytes disagree",
        ),
        (
            "pyproject.toml",
            lambda current: current + "\n# hidden control drift\n",
            "control file",
        ),
    ],
)
def test_formal_head_binding_rejects_hidden_tracked_input_drift(
    tmp_path,
    monkeypatch,
    relative,
    replacement,
    message,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    target = repo / relative
    _git(repo, "update-index", "--assume-unchanged", "--", relative)
    target.write_text(
        replacement(target.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    assert _git(repo, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with pytest.raises((RuntimeError, ValueError), match=message):
        abl._revalidate_formal_checkout(binding)


def test_formal_head_binding_rejects_excluded_untracked_corpus_file(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text(
        "data/bench/task_00/hidden.py\n",
        encoding="utf-8",
    )
    (repo / "data" / "bench" / "task_00" / "hidden.py").write_text(
        "HIDDEN = True\n",
        encoding="utf-8",
    )
    assert _git(repo, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with pytest.raises(ValueError, match="formal corpus bytes disagree"):
        abl._revalidate_formal_checkout(binding)


def test_formal_head_binding_rejects_excluded_corpus_directory_symlink(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    external = tmp_path / "external-corpus"
    external.mkdir()
    (external / "payload.py").write_text("COPIED = True\n", encoding="utf-8")
    relative = "data/bench/task_00/hidden_dir"
    (repo / ".git" / "info" / "exclude").write_text(
        f"{relative}\n",
        encoding="utf-8",
    )
    (repo / relative).symlink_to(external, target_is_directory=True)
    assert _git(repo, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with pytest.raises(RuntimeError, match="link or special node"):
        abl._revalidate_formal_checkout(binding)


@pytest.mark.parametrize(
    "relative",
    [
        "src/lha/runtime.py",
        "data/bench/task_00/module.py",
    ],
)
def test_formal_head_binding_rejects_hidden_hardlink(
    tmp_path,
    monkeypatch,
    relative,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    target = repo / relative
    payload = target.read_bytes()
    external = tmp_path / f"external-{target.name}"
    external.write_bytes(payload)
    _git(repo, "update-index", "--assume-unchanged", "--", relative)
    target.unlink()
    os.link(external, target)
    assert _git(repo, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with pytest.raises(RuntimeError, match="link or special node"):
        abl._revalidate_formal_checkout(binding)


def test_formal_git_plumbing_ignores_local_replace_objects(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    manifest = json.loads(
        (repo / "benchmarks" / "formal_ablation_manifest.json").read_text()
    )
    _git(
        repo,
        "replace",
        binding.preregistration_commit,
        manifest["corpus_commit"],
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    assert abl._git_control_env()["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert abl._revalidate_formal_checkout(binding)


def test_formal_git_remote_resolution_ignores_includes_and_rewrites(
    tmp_path,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    included = tmp_path / "included.gitconfig"
    included.write_text(
        '[remote "formal-witness"]\n'
        "    pushurl = https://github.com/attacker/other.git\n",
        encoding="utf-8",
    )
    expected_url = _git(repo, "remote", "get-url", "formal-witness")
    _git(repo, "config", "--local", "include.path", str(included))
    _git(
        repo,
        "config",
        "--local",
        "url.file:///tmp/attacker.insteadOf",
        "https://github.com/",
    )
    git_path = str(binding.git_executable["path"])

    assert abl._git_control_env()["GIT_CONFIG"] == os.devnull
    assert (
        abl._formal_witness_remote_url(
            git_path,
            repository_root=repo,
            remote_name="formal-witness",
        )
        == expected_url
    )
    _git(
        repo,
        "config",
        "--local",
        "--add",
        "remote.formal-witness.pushurl",
        "https://github.com/example/one.git",
    )
    _git(
        repo,
        "config",
        "--local",
        "--add",
        "remote.formal-witness.pushurl",
        "https://github.com/example/two.git",
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        abl._formal_witness_remote_url(
            git_path,
            repository_root=repo,
            remote_name="formal-witness",
        )


def test_formal_git_helper_identity_uses_disposable_home(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    executable = tmp_path / "bin" / "gh"
    executable.parent.mkdir()
    executable.write_text(
        "#!/bin/sh\n"
        'mkdir -p "$XDG_STATE_HOME/gh"\n'
        'printf "device" > "$XDG_STATE_HOME/gh/device-id"\n'
        'printf "gh version test\\n"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        abl,
        "trusted_executable",
        lambda *_args, **_kwargs: str(executable),
    )

    helper = abl._formal_git_credential_helper("github.com")

    assert helper.executable_path == str(executable)
    assert helper.version == "gh version test"
    assert not (workspace / ".local").exists()
    assert not (workspace / "state").exists()
    executable.write_text(
        "#!/bin/sh\nprintf 'gh version changed\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    with pytest.raises(RuntimeError, match="differs from its registration"):
        abl._formal_git_credential_helper(
            "github.com",
            expected=helper,
        )
    assert not (workspace / ".local").exists()


def test_formal_authenticated_git_env_is_disposable(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl
    from lha.ablation_attempts import FormalGitCredentialHelper

    config = (tmp_path / "gh-config").resolve()
    config.mkdir(mode=0o700)
    hosts = config / "hosts.yml"
    hosts.write_text("github.com:\n  user: test\n", encoding="utf-8")
    hosts.chmod(0o600)
    executable = (tmp_path / "bin" / "gh").resolve()
    executable.parent.mkdir()
    executable.write_text(
        "#!/bin/sh\n"
        'mkdir -p "$XDG_STATE_HOME/gh" '
        '"$XDG_CACHE_HOME/gh" "$XDG_DATA_HOME/gh"\n'
        'printf "device" > "$XDG_STATE_HOME/gh/device-id"\n'
        'printf "test-token\\n"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    helper = FormalGitCredentialHelper(
        host="github.com",
        executable_path=str(executable),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        version="gh version test",
        command=f"!{executable} auth git-credential",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("GH_CONFIG_DIR", str(config))

    with abl._git_authenticated_push_env(helper) as environment:
        home = Path(environment["HOME"])
        temporary_root = home.parent
        assert environment["GH_TOKEN"] == "test-token"
        for key in (
            "HOME",
            "GH_CONFIG_DIR",
            "XDG_CONFIG_HOME",
            "XDG_STATE_HOME",
            "XDG_CACHE_HOME",
            "XDG_DATA_HOME",
            "XDG_RUNTIME_DIR",
            "TMPDIR",
        ):
            assert Path(environment[key]).is_relative_to(temporary_root)
        (home / "probe").mkdir()
        assert (Path(environment["XDG_STATE_HOME"]) / "gh" / "device-id").is_file()

    assert not temporary_root.exists()
    assert hosts.read_text(encoding="utf-8") == "github.com:\n  user: test\n"
    assert not (workspace / ".local").exists()


def test_formal_git_credential_preflight_returns_only_field_names(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl
    from lha.ablation_attempts import FormalGitCredentialHelper
    from lha.tools.shell import ProcResult

    helper = FormalGitCredentialHelper(
        host="github.com",
        executable_path="/opt/homebrew/bin/gh",
        executable_sha256="8" * 64,
        version="gh version test",
        command="!/opt/homebrew/bin/gh auth git-credential",
    )
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        observed["input"] = kwargs["input"]
        return ProcResult(
            0,
            (
                "protocol=https\n"
                "host=github.com\n"
                "username=x-access-token\n"
                "password=test-secret\n"
            ),
            "",
            0.01,
        )

    monkeypatch.setattr(abl, "run", fake_run)
    summary = abl._preflight_formal_git_credential_helper(
        str((tmp_path / "git").resolve()),
        helper,
    )

    assert summary == {
        "host": "github.com",
        "fields": ("host", "password", "protocol", "username"),
    }
    assert "test-secret" not in json.dumps(summary)
    assert "push" not in observed["command"]
    assert "refs/" not in str(observed["input"])


def test_formal_git_credential_preflight_rejects_unknown_fields_without_values(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl
    from lha.ablation_attempts import FormalGitCredentialHelper
    from lha.tools.shell import ProcResult

    helper = FormalGitCredentialHelper(
        host="github.com",
        executable_path="/opt/homebrew/bin/gh",
        executable_sha256="8" * 64,
        version="gh version test",
        command="!/opt/homebrew/bin/gh auth git-credential",
    )
    monkeypatch.setattr(
        abl,
        "run",
        lambda *_args, **_kwargs: ProcResult(
            0,
            (
                "protocol=https\n"
                "host=github.com\n"
                "username=x-access-token\n"
                "password=test-secret\n"
                "unexpected=test-secret\n"
            ),
            "",
            0.01,
        ),
    )

    with pytest.raises(RuntimeError) as captured:
        abl._preflight_formal_git_credential_helper(
            str((tmp_path / "git").resolve()),
            helper,
        )
    assert "test-secret" not in str(captured.value)


def test_formal_runner_rejects_source_only_remote_branch(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    output_path = f"runs/formal_ablation/{'a' * 64}"
    binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    witness_url = _git(repo, "remote", "get-url", "formal-witness")
    _git(
        repo,
        "push",
        "-q",
        "--force",
        _FORMAL_TEST_REMOTES[witness_url],
        f"{protocol.source_commit}:refs/heads/{branch}",
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        with pytest.raises(RuntimeError, match="not published"):
            abl._bind_formal_attempt(
                formal_corpus=binding,
                formal_output_lease=output_lease,
                model=protocol.model,
                reasoning_effort=protocol.reasoning_effort,
                docker_image_id=protocol.docker_image_id,
                source_tree_sha256=protocol.source_tree_sha256,
                codex_cli_version=protocol.codex_cli_version,
                codex_cli_executable_sha256=(
                    protocol.codex_cli_executable_sha256
                ),
                codex_client=protocol.codex_client,
            )


def test_registration_rejects_hidden_source_before_external_preflights(
    tmp_path,
    monkeypatch,
):
    import lha.formal_attempt_cli as commands
    from lha.ablation_attempts import (
        FormalAblationAttemptRegistry,
        formal_ablation_attempt_registry_bytes,
    )

    repo = tmp_path / "repo"
    _binding, protocol, registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=f"runs/formal_ablation/{'a' * 64}",
    )
    registry_path.write_bytes(
        formal_ablation_attempt_registry_bytes(
            FormalAblationAttemptRegistry(events=())
        )
    )
    _git(repo, "add", "benchmarks/formal_ablation_attempts.json")
    _git(repo, "commit", "-qm", "close test registry")
    _git(
        repo,
        "update-index",
        "--assume-unchanged",
        "--",
        "src/lha/runtime.py",
    )
    (repo / "src" / "lha" / "runtime.py").write_text(
        "VALUE = 99\n",
        encoding="utf-8",
    )
    assert _git(repo, "status", "--porcelain=v1") == ""
    external_calls: list[str] = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            external_calls.append(name)
            raise AssertionError(f"{name} ran before trusted HEAD validation")

        return fail

    monkeypatch.setattr(commands, "_https_witness_remote", forbidden("remote"))
    monkeypatch.setattr(commands, "_resolve_docker_image_id", forbidden("docker"))
    monkeypatch.setattr(commands, "_probe_docker_image", forbidden("probe"))
    monkeypatch.setattr(commands, "_codex_protocol", forbidden("codex"))

    with pytest.raises(
        commands.FormalAttemptCommandError,
        match="trusted HEAD",
    ):
        commands.register_formal_attempt(
            repo_root=repo,
            config=Config(),
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            witness_remote_name="formal-witness",
        )

    assert external_calls == []


def test_formal_run_rejects_an_injected_llm_before_writing_header(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    llm = _AuditedCodexLLM()

    with pytest.raises(ValueError, match="does not accept an injected LLM"):
        _run_formal_until_registration(
            repo,
            binding,
            repo / output_path,
            llm,
            monkeypatch,
            inject_llm=True,
        )

    assert llm.calls == 0
    assert not (repo / output_path / "formal_run.json").exists()


def test_formal_run_rejects_copied_cell_evidence_before_model_call(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    copied = repo / output_path / "results"
    copied.mkdir(parents=True)
    (copied / "task__r0.started.json").write_text("{}")
    (copied / "task__r0.json").write_text("{}")
    llm = _AuditedCodexLLM()

    with pytest.raises(RuntimeError, match="mark the registered attempt ABANDONED"):
        _run_formal_until_registration(
            repo,
            binding,
            repo / output_path,
            llm,
            monkeypatch,
        )

    assert llm.calls == 0


def test_formal_run_rejects_witness_network_failure_before_model_call(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    witness_url = _git(repo, "remote", "get-url", "formal-witness")
    remote_path = Path(_FORMAL_TEST_REMOTES[witness_url])
    shutil.rmtree(remote_path)
    llm = _AuditedCodexLLM()

    with pytest.raises(RuntimeError, match="registration branch confirmation"):
        _run_formal_until_registration(
            repo,
            binding,
            repo / output_path,
            llm,
            monkeypatch,
        )

    assert llm.calls == 0
    assert not (repo / output_path / "formal_run.json").exists()


def test_formal_header_is_durable_before_witness_push(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    output_path = f"runs/formal_ablation/{'a' * 64}"
    binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)
    original = abl._create_formal_start_witness
    observed_header: list[bytes] = []

    def create_witness(attempt, *, outcome_key, run_header_sha256):
        header = (repo / output_path / "formal_run.json").read_bytes()
        assert hashlib.sha256(header).hexdigest() == run_header_sha256
        observed_header.append(header)
        return original(
            attempt,
            outcome_key=outcome_key,
            run_header_sha256=run_header_sha256,
        )

    monkeypatch.setattr(abl, "_create_formal_start_witness", create_witness)
    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt = abl._bind_formal_attempt(
            formal_corpus=binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=(
                protocol.codex_cli_executable_sha256
            ),
            codex_client=protocol.codex_client,
        )
        run_binding = abl._initialize_formal_run(attempt, output_lease)

    assert observed_header
    assert run_binding.header_sha256 == hashlib.sha256(
        observed_header[0]
    ).hexdigest()


def test_formal_witness_push_uses_only_the_registered_helper(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    output_path = f"runs/formal_ablation/{'a' * 64}"
    binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)
    original_run = abl.run
    observed: list[tuple[list[str], dict[str, str]]] = []

    def inspect_run(command, **kwargs):
        if "push" in command:
            observed.append((list(command), dict(kwargs["env"])))
        return original_run(command, **kwargs)

    monkeypatch.setattr(abl, "run", inspect_run)
    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt = abl._bind_formal_attempt(
            formal_corpus=binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=(
                protocol.codex_cli_executable_sha256
            ),
            codex_client=protocol.codex_client,
        )
        abl._initialize_formal_run(attempt, output_lease)

    assert len(observed) == 1
    command, environment = observed[0]
    helper = protocol.witness_credential_helper
    assert helper is not None
    assert command[1:5] == [
        "-c",
        "credential.helper=",
        "-c",
        f"credential.https://github.com.helper={helper.command}",
    ]
    assert attempt.witness_remote_url in command
    assert attempt.witness_remote_name not in command
    assert environment["GIT_CONFIG"] == os.devnull
    assert Path(environment["HOME"]).is_absolute()


def test_witness_push_failure_leaves_complete_header_for_abandonment(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    output_path = f"runs/formal_ablation/{'a' * 64}"
    binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    def fail_push(_attempt, *, outcome_key, run_header_sha256):
        header = (repo / output_path / "formal_run.json").read_bytes()
        parsed = json.loads(header)
        assert parsed["outcome_key"] == outcome_key
        assert hashlib.sha256(header).hexdigest() == run_header_sha256
        raise RuntimeError("injected witness push failure")

    monkeypatch.setattr(abl, "_create_formal_start_witness", fail_push)
    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt = abl._bind_formal_attempt(
            formal_corpus=binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=(
                protocol.codex_cli_executable_sha256
            ),
            codex_client=protocol.codex_client,
        )
        with pytest.raises(RuntimeError, match="push failure"):
            abl._initialize_formal_run(attempt, output_lease)

    header = repo / output_path / "formal_run.json"
    assert header.exists()
    assert json.loads(header.read_bytes())["formal_attempt_id"] == "a" * 64


def test_formal_run_header_prevents_retry_after_cell_files_are_deleted(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    corpus_binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt_binding = abl._bind_formal_attempt(
            formal_corpus=corpus_binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
            codex_client=protocol.codex_client,
        )
        run_binding = abl._initialize_formal_run(
            attempt_binding,
            output_lease,
        )
        results = repo / output_path / "results"
        results.mkdir()
        (results / "task__r0.started.json").write_text("{}")
        (results / "task__r0.json").write_text("{}")
        for child in results.iterdir():
            child.unlink()
        results.rmdir()

        with pytest.raises(RuntimeError, match="mark the registered attempt ABANDONED"):
            abl._initialize_formal_run(attempt_binding, output_lease)

    assert (repo / output_path / run_binding.header_path).exists()


def test_formal_witness_prevents_retry_after_output_directory_is_deleted(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    corpus_binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)

    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt = abl._bind_formal_attempt(
            formal_corpus=corpus_binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
            codex_client=protocol.codex_client,
        )
        first = abl._initialize_formal_run(attempt, output_lease)

    shutil.rmtree(repo / output_path)
    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt = abl._bind_formal_attempt(
            formal_corpus=corpus_binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
            codex_client=protocol.codex_client,
        )
        with pytest.raises(RuntimeError, match="start witness was not created"):
            abl._initialize_formal_run(attempt, output_lease)

    remote_ref = _git(
        repo,
        "ls-remote",
        "--refs",
        _FORMAL_TEST_REMOTES[first.witness_remote_url],
        first.witness_ref,
    )
    assert remote_ref == f"{first.witness_commit}\t{first.witness_ref}"


def test_formal_witness_up_to_date_push_is_not_a_new_start(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    corpus_binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)
    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt = abl._bind_formal_attempt(
            formal_corpus=corpus_binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
            codex_client=protocol.codex_client,
        )
        outcome_key = "1" * 64
        header_sha256 = "2" * 64
        abl._create_formal_start_witness(
            attempt,
            outcome_key=outcome_key,
            run_header_sha256=header_sha256,
        )
        with pytest.raises(RuntimeError, match="start witness was not created"):
            abl._create_formal_start_witness(
                attempt,
                outcome_key=outcome_key,
                run_header_sha256=header_sha256,
            )


def test_concurrent_formal_witness_creation_has_one_winner(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    corpus_binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)
    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt = abl._bind_formal_attempt(
            formal_corpus=corpus_binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
            codex_client=protocol.codex_client,
        )

        def create(index: int) -> str:
            return abl._create_formal_start_witness(
                attempt,
                outcome_key=f"{index + 1:064x}",
                run_header_sha256="3" * 64,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(create, index) for index in range(2)]
            outcomes: list[str] = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except RuntimeError:
                    outcomes.append("rejected")

    assert len([value for value in outcomes if value != "rejected"]) == 1
    assert outcomes.count("rejected") == 1


def test_header_failure_happens_before_witness_creation(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    corpus_binding, protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    monkeypatch.setattr(abl, "_project_root", lambda: repo)
    original_safety_check = abl._safe_named_regular_file

    def fail_header_safety_check(*_args, **_kwargs):
        raise OSError("injected")

    with abl._formal_ablation_lock(repo / output_path) as output_lease:
        attempt = abl._bind_formal_attempt(
            formal_corpus=corpus_binding,
            formal_output_lease=output_lease,
            model=protocol.model,
            reasoning_effort=protocol.reasoning_effort,
            docker_image_id=protocol.docker_image_id,
            source_tree_sha256=protocol.source_tree_sha256,
            codex_cli_version=protocol.codex_cli_version,
            codex_cli_executable_sha256=protocol.codex_cli_executable_sha256,
            codex_client=protocol.codex_client,
        )
        monkeypatch.setattr(
            abl,
            "_safe_named_regular_file",
            fail_header_safety_check,
        )
        try:
            with pytest.raises(RuntimeError, match="header could not be sealed"):
                abl._initialize_formal_run(attempt, output_lease)
        finally:
            monkeypatch.setattr(
                abl,
                "_safe_named_regular_file",
                original_safety_check,
            )

    assert (repo / output_path / "formal_run.json").exists()
    assert (
        _git(
                repo,
                "ls-remote",
                "--refs",
                _FORMAL_TEST_REMOTES[attempt.witness_remote_url],
            attempt.witness_ref,
        )
        == ""
    )


def test_unregistered_run_record_cannot_authorize_a_formal_run(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="UNREGISTERED_RUN_RECORDED",
        output_path=output_path,
    )
    llm = _AuditedCodexLLM()

    with pytest.raises(RuntimeError, match="open REGISTERED"):
        _run_formal_until_registration(
            repo,
            binding,
            repo / output_path,
            llm,
            monkeypatch,
        )
    assert llm.calls == 0


def test_formal_registration_rejects_a_different_output_before_model_call(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    registered_output = f"runs/formal_ablation/{attempt_id}"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=registered_output,
    )
    llm = _AuditedCodexLLM()
    other_output = repo / "runs" / "formal_ablation" / ("b" * 64)

    with pytest.raises(RuntimeError, match="does not match this run"):
        _run_formal_until_registration(
            repo,
            binding,
            other_output,
            llm,
            monkeypatch,
        )
    assert llm.calls == 0


def test_formal_registration_file_must_be_tracked_before_model_call(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    binding, _protocol, _registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
        tracked=False,
    )
    llm = _AuditedCodexLLM()

    with pytest.raises(RuntimeError, match="tracked-file check"):
        _run_formal_until_registration(
            repo,
            binding,
            repo / output_path,
            llm,
            monkeypatch,
        )
    assert llm.calls == 0


def test_formal_registration_worktree_rewrite_fails_before_model_call(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    attempt_id = "a" * 64
    output_path = f"runs/formal_ablation/{attempt_id}"
    binding, _protocol, registry_path = _formal_attempt_repository(
        repo,
        event="REGISTERED",
        output_path=output_path,
    )
    registry_path.write_text(registry_path.read_text() + "\n")
    llm = _AuditedCodexLLM()

    with pytest.raises(RuntimeError, match="clean committed worktree"):
        _run_formal_until_registration(
            repo,
            binding,
            repo / output_path,
            llm,
            monkeypatch,
        )
    assert llm.calls == 0


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_formal_cell_start_link_attack_does_not_touch_target(tmp_path, attack):
    import lha.ablation as abl

    out = tmp_path / "out"
    results = out / "results"
    results.mkdir(parents=True)
    marker = results / "task__r0.started.json"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"unchanged")
    if attack == "symlink":
        marker.symlink_to(sentinel)
    else:
        os.link(sentinel, marker)

    with pytest.raises(RuntimeError, match="already exists"):
        abl._write_formal_cell_start(
            marker,
            b'{"started":true}',
            out_dir=out,
        )
    assert sentinel.read_bytes() == b"unchanged"


def test_formal_cell_start_is_exclusive_and_syncs_file_before_directory(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    out = tmp_path / "out"
    marker = out / "results" / "task__r0.started.json"
    payload = b'{"started":true}'
    with abl._formal_ablation_lock(out) as lease:
        results_descriptor = abl._open_lease_subdirectory(lease, ("results",))
        os.close(results_descriptor)
        real_fsync = os.fsync
        sync_kinds = []

        def recording_fsync(descriptor):
            metadata = os.fstat(descriptor)
            sync_kinds.append("directory" if stat.S_ISDIR(metadata.st_mode) else "file")
            real_fsync(descriptor)

        monkeypatch.setattr(abl.os, "fsync", recording_fsync)
        abl._write_formal_cell_start(
            marker,
            payload,
            out_dir=out,
            lease=lease,
        )
        assert sync_kinds == ["file", "directory"]
        with pytest.raises(RuntimeError, match="already exists"):
            abl._write_formal_cell_start(
                marker,
                b"different",
                out_dir=out,
                lease=lease,
            )
    assert marker.read_bytes() == payload


def test_formal_cell_start_directory_sync_failure_leaves_orphan_marker(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    out = tmp_path / "out"
    marker = out / "results" / "task__r0.started.json"
    payload = b'{"started":true}'
    with abl._formal_ablation_lock(out) as lease:
        results_descriptor = abl._open_lease_subdirectory(lease, ("results",))
        os.close(results_descriptor)
        real_fsync = os.fsync

        def fail_directory_sync(descriptor):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError("simulated crash boundary")
            real_fsync(descriptor)

        monkeypatch.setattr(abl.os, "fsync", fail_directory_sync)
        with pytest.raises(RuntimeError, match="was not durable"):
            abl._write_formal_cell_start(
                marker,
                payload,
                out_dir=out,
                lease=lease,
            )
        monkeypatch.setattr(abl.os, "fsync", real_fsync)
        with pytest.raises(RuntimeError, match="already exists"):
            abl._write_formal_cell_start(
                marker,
                payload,
                out_dir=out,
                lease=lease,
            )
    assert marker.read_bytes() == payload


def test_formal_cell_start_zero_length_write_fails_closed(tmp_path, monkeypatch):
    import lha.ablation as abl

    out = tmp_path / "out"
    marker = out / "results" / "task__r0.started.json"
    with abl._formal_ablation_lock(out) as lease:
        results_descriptor = abl._open_lease_subdirectory(lease, ("results",))
        os.close(results_descriptor)
        monkeypatch.setattr(abl.os, "write", lambda descriptor, payload: 0)

        with pytest.raises(RuntimeError, match="was not durable"):
            abl._write_formal_cell_start(
                marker,
                b'{"started":true}',
                out_dir=out,
                lease=lease,
            )
    assert marker.exists()
    assert marker.read_bytes() == b""


def test_ablation_atomic_replace_syncs_parent_directory(tmp_path, monkeypatch):
    import lha.ablation as abl

    target = tmp_path / "result.json"
    real_fsync = os.fsync
    sync_kinds = []

    def recording_fsync(descriptor):
        metadata = os.fstat(descriptor)
        sync_kinds.append("directory" if stat.S_ISDIR(metadata.st_mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(abl.os, "fsync", recording_fsync)
    abl._atomic_write_bytes(target, b"complete")

    assert sync_kinds == ["file", "directory", "file"]
    assert target.read_bytes() == b"complete"


def test_formal_cell_failure_before_llm_call_fails_closed(tmp_path, monkeypatch):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = TaskSpec.from_file(_task(tmp_path, src))
    task.inputs["_name"] = "task"

    def fail_before_call(*args, **kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(abl, "_first_attempt", fail_before_call)

    with pytest.raises(RuntimeError, match="before an auditable LLM call"):
        abl._run_cell(
            abl._PromptAuditClient(_AuditedCodexLLM()),
            src,
            task,
            0,
            tmp_path / "out",
            "f" * 64,
            abl._repo_digest(src),
            "a" * 64,
            TrustedLocalBackend(),
            TrustedLocalBackend(),
            "trusted-local",
            None,
            [],
            True,
            1,
            True,
        )


def test_formal_cell_rejects_initial_snapshot_mismatch(tmp_path):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = TaskSpec.from_file(_task(tmp_path, src))
    task.inputs["_name"] = "task"

    with pytest.raises(RuntimeError, match="input snapshot failed validation"):
        abl._run_cell(
            abl._PromptAuditClient(_AuditedCodexLLM()),
            src,
            task,
            0,
            tmp_path / "out",
            "f" * 64,
            "0" * 64,
            "a" * 64,
            TrustedLocalBackend(),
            TrustedLocalBackend(),
            "trusted-local",
            None,
            [],
            True,
            1,
            True,
        )


def test_formal_generic_error_after_successful_call_fails_closed(tmp_path, monkeypatch):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = TaskSpec.from_file(_task(tmp_path, src))
    task.inputs["_name"] = "task"

    def fail_after_success(*args, **kwargs):
        audits = args[-1]
        audits[-1]["result_artifact_sha256"] = "b" * 64
        raise RuntimeError("scorer setup failed")

    monkeypatch.setattr(abl, "_evaluate", fail_after_success)

    with pytest.raises(RuntimeError, match="did not end in a failed LLM call"):
        abl._run_cell(
            abl._PromptAuditClient(_AuditedCodexLLM()),
            src,
            task,
            0,
            tmp_path / "out",
            "f" * 64,
            abl._repo_digest(src),
            "a" * 64,
            TrustedLocalBackend(),
            TrustedLocalBackend(),
            "trusted-local",
            None,
            [],
            True,
            1,
            True,
        )


def test_formal_schema4_rejects_terminal_repair_failure(tmp_path):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = TaskSpec.from_file(_task(tmp_path, src))
    task.inputs["_name"] = "task"
    llm = _SuccessThenFailAuditedCodexLLM()

    with pytest.raises(
        RuntimeError,
        match="schema-4 ERROR cannot contain a successful or repair call",
    ):
        abl._run_cell(
            abl._PromptAuditClient(llm),
            src,
            task,
            0,
            tmp_path / "out",
            "f" * 64,
            abl._repo_digest(src),
            "a" * 64,
            TrustedLocalBackend(),
            TrustedLocalBackend(),
            "trusted-local",
            None,
            [],
            True,
            1,
            True,
        )
    assert llm.calls == 2
    assert not (tmp_path / "out" / "results" / "task__r0.json").exists()


def test_formal_evaluator_infra_error_fails_closed(tmp_path, monkeypatch):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = TaskSpec.from_file(_task(tmp_path, src))
    task.inputs["_name"] = "task"

    def return_infra_error(*args, **kwargs):
        audits = args[-1]
        audits[-1]["result_artifact_sha256"] = "b" * 64
        return [
            RunRecord(
                "task",
                condition,
                0,
                "ERROR",
                False,
                False,
                False,
                False,
                0,
                scorer_outcome="INFRA_ERROR",
            )
            for condition, _ in CONDITIONS
        ]

    monkeypatch.setattr(abl, "_evaluate", return_infra_error)

    with pytest.raises(RuntimeError, match="evaluator returned infrastructure ERROR"):
        abl._run_cell(
            abl._PromptAuditClient(_AuditedCodexLLM()),
            src,
            task,
            0,
            tmp_path / "out",
            "f" * 64,
            abl._repo_digest(src),
            "a" * 64,
            TrustedLocalBackend(),
            TrustedLocalBackend(),
            "trusted-local",
            None,
            [],
            True,
            1,
            True,
        )


def test_formal_completed_cell_cache_write_failure_aborts(tmp_path, monkeypatch):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = TaskSpec.from_file(_task(tmp_path, src))
    task.inputs["_name"] = "task"

    def return_success(*args, **kwargs):
        audits = args[-1]
        audits[-1]["result_artifact_sha256"] = "b" * 64
        return [
            RunRecord(
                "task",
                condition,
                0,
                "DONE",
                True,
                True,
                True,
                False,
                0,
                artifact_sha256="b" * 64,
                scorer_outcome="PASS",
                scorer_evidence_sha256="c" * 64,
            )
            for condition, _ in CONDITIONS
        ]

    real_formal_write = abl._write_formal_cell_start

    def fail_result_cache(path, payload, **kwargs):
        if path.parent.name == "results" and not path.name.endswith(
            ".started.json"
        ):
            raise OSError("disk full")
        real_formal_write(path, payload, **kwargs)

    monkeypatch.setattr(abl, "_evaluate", return_success)
    monkeypatch.setattr(abl, "_write_formal_cell_start", fail_result_cache)

    audit_log = []
    with pytest.raises(RuntimeError, match="could not seal a completed cell"):
        abl._run_cell(
            abl._PromptAuditClient(_AuditedCodexLLM()),
            src,
            task,
            0,
            tmp_path / "out",
            "f" * 64,
            abl._repo_digest(src),
            "a" * 64,
            TrustedLocalBackend(),
            TrustedLocalBackend(),
            "trusted-local",
            None,
            audit_log,
            True,
            1,
            True,
        )
    assert audit_log == []


def test_formal_post_cell_snapshot_drift_aborts(tmp_path, monkeypatch):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = TaskSpec.from_file(_task(tmp_path, src))
    task.inputs["_name"] = "task"
    source_sha256 = abl._repo_digest(src)
    digest_calls = 0

    def drifting_digest(path):
        nonlocal digest_calls
        digest_calls += 1
        return source_sha256 if digest_calls == 1 else "0" * 64

    def return_success(*args, **kwargs):
        audits = args[-1]
        audits[-1]["result_artifact_sha256"] = "b" * 64
        return [
            RunRecord(
                "task",
                condition,
                0,
                "DONE",
                True,
                True,
                True,
                False,
                0,
                artifact_sha256="b" * 64,
                scorer_outcome="PASS",
                scorer_evidence_sha256="c" * 64,
            )
            for condition, _ in CONDITIONS
        ]

    monkeypatch.setattr(abl, "_repo_digest", drifting_digest)
    monkeypatch.setattr(abl, "_evaluate", return_success)

    with pytest.raises(RuntimeError, match="input changed during the cell"):
        abl._run_cell(
            abl._PromptAuditClient(_AuditedCodexLLM()),
            src,
            task,
            0,
            tmp_path / "out",
            "f" * 64,
            source_sha256,
            "a" * 64,
            TrustedLocalBackend(),
            TrustedLocalBackend(),
            "trusted-local",
            None,
            [],
            True,
            1,
            True,
        )


def test_error_records_are_never_reused_from_cache(tmp_path):
    import lha.ablation as abl

    cache = tmp_path / "cell.json"
    fingerprint = "f" * 64
    error = RunRecord("task", "trust", 0, "ERROR", False, False, False, False, 0)
    cache.write_text(
        json.dumps(
            {
                "schema_version": abl._CACHE_SCHEMA,
                "fingerprint": fingerprint,
                "records": [error.__dict__],
                "llm_calls": [],
            }
        )
    )
    assert abl._load_cached(cache, fingerprint) is None


def test_committed_formal_corpus_manifest_matches_fixed_inputs():
    import lha.ablation as abl

    root = Path(__file__).resolve().parents[1]
    manifest, digest = abl._load_formal_corpus_manifest(
        root / abl._FORMAL_CORPUS_MANIFEST_PATH,
        root,
    )

    assert len(manifest["tasks"]) == 17
    assert manifest["repetitions"] == 12
    assert len(digest) == 64


def test_cache_reader_rejects_oversized_file(tmp_path, monkeypatch):
    import lha.ablation as abl

    monkeypatch.setattr(abl, "_MAX_CACHE_BYTES", 128)
    cache = tmp_path / "cell.json"
    cache.write_bytes(b"x" * 129)

    assert abl._read_cache(cache) is None


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_bounded_evidence_reader_rejects_links(tmp_path, link_kind):
    import lha.ablation as abl

    backing = tmp_path / "backing.json"
    backing.write_text("{}")
    evidence = tmp_path / "evidence.json"
    if link_kind == "symlink":
        evidence.symlink_to(backing)
    else:
        os.link(backing, evidence)

    with pytest.raises(ValueError, match="regular file|hard links"):
        abl._read_bounded_bytes(evidence, max_bytes=1024)


def test_bounded_evidence_reader_rejects_file_changed_during_read(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(b"{}")
    real_read = abl.os.read
    changed = False

    def mutate_after_read(descriptor, size):
        nonlocal changed
        chunk = real_read(descriptor, size)
        if chunk and not changed:
            changed = True
            with evidence.open("ab") as stream:
                stream.write(b" ")
        return chunk

    monkeypatch.setattr(abl.os, "read", mutate_after_read)

    with pytest.raises(ValueError, match="changed while"):
        abl._read_bounded_bytes(evidence, max_bytes=1024)


def test_resumable_caches_real_outcomes(tmp_path):
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    base = _base(tmp_path)
    llm = _FixedLLM(2)
    rep1 = run_ablation(base, [task], reps=1, out_dir=out, llm_client=llm)
    assert (out / "results" / "task__r0.json").exists()
    assert json.loads((out / "results" / "task__r0.json").read_text())["terminal_error"] is False
    assert rep1.llm_calls[0]["cache_hit"] is False
    calls_after_first_run = llm.calls

    # A cached cell must NOT re-invoke the LLM.
    rep2 = run_ablation(base, [task], reps=1, out_dir=out, llm_client=llm)
    assert _by_cond(rep2)["trust"].true_success  # served from cache, no LLM call
    assert llm.calls == calls_after_first_run
    assert rep2.llm_calls[0]["cache_hit"] is True


def test_formal_cache_without_call_receipts_is_a_cache_miss(tmp_path):
    import lha.ablation as abl

    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    base = _base(tmp_path)
    run_ablation(
        base,
        [task],
        reps=1,
        out_dir=out,
        llm_client=_FixedLLM(2),
    )
    cache = out / "results" / "task__r0.json"
    raw = json.loads(cache.read_text())
    report = json.loads((out / "ablation_report.json").read_text())

    assert (
        abl._load_cached_cell(
            cache,
            raw["fingerprint"],
            input_snapshot_sha256=report["provenance"]["input_snapshot_sha256"]["task"],
            scorer_backend="trusted-local",
            scorer_image_id=None,
            require_call_receipts=True,
            expected_task="task",
            expected_rep=0,
            receipt_dir=out / "llm_call_receipts",
            max_outer_attempts=3,
            max_inner_attempts=3,
        )
        is None
    )


def test_codex_cache_with_missing_receipt_reference_is_recomputed(tmp_path):
    import lha.ablation as abl

    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    llm = _AuditedCodexLLM()
    run_ablation(
        _base(tmp_path),
        [task],
        reps=1,
        out_dir=out,
        llm_client=llm,
        model=llm.model,
    )
    calls_after_first = llm.calls
    cache = out / "results" / "task__r0.json"
    raw = json.loads(cache.read_text())
    raw.pop("llm_call_receipts")
    cache.write_text(json.dumps(raw))

    second = run_ablation(
        _base(tmp_path),
        [task],
        reps=1,
        out_dir=out,
        llm_client=llm,
        model=llm.model,
    )

    assert llm.calls > calls_after_first
    assert second.llm_calls[0]["cache_hit"] is False
    assert json.loads(cache.read_text())["llm_call_receipts"]
    report = json.loads((out / "ablation_report.json").read_text())
    reference = report["llm_calls"][0]
    receipt = json.loads(
        (out / "llm_call_receipts" / f"{reference['receipt_sha256']}.json").read_text()
    )
    assert receipt["binding"]["task"] == "task"
    assert receipt["binding"]["rep"] == 0
    assert len(receipt["binding"]["prompt_sha256"]) == 64
    assert len(receipt["binding"]["response_sha256"]) == 64
    assert len(receipt["binding"]["patch_sha256"]) == 64
    assert report["fingerprint"] == abl._report_fingerprint(report)


def test_cache_is_rejected_when_frozen_artifact_bytes_are_damaged(tmp_path):
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    base = _base(tmp_path)
    llm = _FixedLLM(2)
    first = run_ablation(base, [task], reps=1, out_dir=out, llm_client=llm)
    digest = _by_cond(first)["trust"].artifact_sha256
    artifact = out / "artifacts" / f"{digest}.json"
    artifact.write_text("{}")
    calls = llm.calls
    llm.value = 3

    second = run_ablation(base, [task], reps=1, out_dir=out, llm_client=llm)

    assert llm.calls > calls
    assert _by_cond(second)["trust"].false_success


def test_cache_rejects_valid_receipt_swapped_from_another_artifact(tmp_path):
    class DifferentPassingLLM(_FixedLLM):
        def propose_patch(self, step, bundle, workdir):
            self.calls += 1
            expression = "2" if self.calls % 2 else "1 + 1"
            return Patch(
                step_id=step.step_id,
                file_contents={"m.py": f"def f():\n    return {expression}\n"},
                touched_files=["m.py"],
            )

    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    llm = DifferentPassingLLM(2)
    first = run_ablation(
        _base(tmp_path),
        [task],
        reps=2,
        out_dir=out,
        llm_client=llm,
    )
    rep_artifacts = {
        rep: next(
            record.artifact_sha256
            for record in first.records
            if record.rep == rep and record.condition == "trust"
        )
        for rep in (0, 1)
    }
    assert rep_artifacts[0] != rep_artifacts[1]

    first_cache = out / "results" / "task__r0.json"
    second_cache = out / "results" / "task__r1.json"
    first_raw = json.loads(first_cache.read_text())
    second_raw = json.loads(second_cache.read_text())
    donor_digest = second_raw["records"][0]["scorer_evidence_sha256"]
    for record in first_raw["records"]:
        record["scorer_evidence_sha256"] = donor_digest
    first_cache.write_text(json.dumps(first_raw))
    calls = llm.calls

    run_ablation(
        _base(tmp_path),
        [task],
        reps=2,
        out_dir=out,
        llm_client=llm,
    )

    assert llm.calls == calls + 1


def test_live_corpus_change_then_restore_cannot_change_frozen_run_inputs(tmp_path):
    src = _repo(tmp_path / "src")
    original_test = (src / "tests" / "test_m.py").read_text()

    class _MutatingLLM(_FixedLLM):
        def propose_patch(self, step, bundle, workdir):
            if self.calls == 0:
                (src / "tests" / "test_m.py").write_text(original_test.replace("== 2", "== 3"))
            else:
                (src / "tests" / "test_m.py").write_text(original_test)
            return super().propose_patch(step, bundle, workdir)

    report = run_ablation(
        _base(tmp_path),
        [_task(tmp_path, src)],
        llm="stub",
        reps=2,
        out_dir=tmp_path / "out",
        llm_client=_MutatingLLM(2),
    )

    assert all(record.status != "ERROR" for record in report.records)
    assert all(record.artifact_correct for record in report.records)
    assert (src / "tests" / "test_m.py").read_text() == original_test


def test_cell_infrastructure_error_does_not_stop_later_cells(tmp_path, monkeypatch):
    import lha.ablation as abl

    original_evaluate = abl._evaluate
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated artifact store failure")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(abl, "_evaluate", fail_once)
    src = _repo(tmp_path / "src")
    out = tmp_path / "out"
    report = run_ablation(
        _base(tmp_path),
        [_task(tmp_path, src)],
        llm="stub",
        reps=2,
        out_dir=out,
        llm_client=_FixedLLM(2),
    )

    first = [record for record in report.records if record.rep == 0]
    second = [record for record in report.records if record.rep == 1]
    assert all(record.status == "ERROR" for record in first)
    assert all("infrastructure failure" in record.detail for record in first)
    assert all(record.status != "ERROR" for record in second)
    assert not (out / "results" / "task__r0.json").exists()
    assert (out / "results" / "task__r1.json").exists()


# --- aggregation + report ---------------------------------------------------
def test_aggregate_and_markdown():
    records = [
        RunRecord("t1", "trust", 0, "DONE", True, False, False, True, 0),
        RunRecord("t1", "gate", 0, "FAILED", False, False, False, False, 0),
        RunRecord("t1", "verify", 0, "DONE", True, True, True, False, 1),
        RunRecord("t2", "trust", 0, "DONE", True, True, True, False, 0),
        RunRecord("t2", "gate", 0, "DONE", True, True, True, False, 0),
        RunRecord("t2", "verify", 0, "DONE", True, True, True, False, 0),
    ]
    stats = {s.condition: s for s in _aggregate(records)}
    assert stats["trust"].false_success_rate == 0.5
    assert stats["gate"].false_success_rate == 0.0
    assert stats["verify"].true_success_rate == 1.0
    report = AblationReport("stub", "", 1, ["t1", "t2"], records, list(stats.values()))
    md = report.to_markdown()
    assert "Verification ablation" in md and "false success" in md and "false-pass" in md


def test_formal_markdown_does_not_round_nonzero_mcnemar_p_to_zero():
    records: list[RunRecord] = []
    for rep in range(12):
        records.extend(
            [
                RunRecord("t", "trust", rep, "DONE", True, False, False, True, 0),
                RunRecord("t", "gate", rep, "FAILED", False, False, False, False, 0),
                RunRecord("t", "verify", rep, "DONE", True, True, True, False, 1),
            ]
        )
    report = AblationReport(
        "codex_cli",
        "model-x",
        12,
        ["t"],
        records,
        _aggregate(records),
        schema_version=4,
    )

    markdown = report.to_markdown()

    assert "exact McNemar p = 0.0005" in markdown
    assert "exact McNemar p = 0.00\n" not in markdown


def test_formal_markdown_names_attempt_registration():
    attempt_id = "1" * 64
    registration_commit = "2" * 40
    registry_path = "benchmarks/formal_ablation_attempts.json"
    report = AblationReport(
        "codex_cli",
        "model-x",
        12,
        [],
        schema_version=4,
        provenance=AblationProvenance(
            formal_attempt_id=attempt_id,
            formal_attempt_registration_commit=registration_commit,
            formal_attempt_registry_path=registry_path,
        ),
    )

    markdown = report.to_markdown()

    assert f"- formal attempt: `{attempt_id}`" in markdown
    assert registration_commit in markdown
    assert registry_path in markdown


def test_errored_runs_excluded_from_rates():
    records = [
        RunRecord("t1", "trust", 0, "ERROR", False, False, False, False, 0),
        RunRecord("t2", "trust", 0, "DONE", True, False, False, True, 0),
    ]
    stats = {s.condition: s for s in _aggregate(records)}
    assert stats["trust"].n == 1 and stats["trust"].false_success_rate == 1.0
    assert isinstance(stats["trust"], ConditionStats)
    assert CONDITIONS[0][0] == "trust"


def test_boundary_rate_intervals_use_wilson_instead_of_collapsing():
    records = [
        RunRecord(f"t{i}", "verify", 0, "DONE", True, True, True, False, 0) for i in range(4)
    ]
    verify = {s.condition: s for s in _aggregate(records)}["verify"]
    assert verify.true_ci is not None and verify.true_ci[0] < 1.0
    assert verify.true_ci[1] == 1.0
    assert verify.false_ci is not None and verify.false_ci[0] == 0.0
    assert verify.false_ci[1] > 0.0


# --- P0-D: independent truth (prediction vs scorer) ---------------------------
def test_gate_rejected_correct_fix_is_scored_as_false_negative(tmp_path, monkeypatch):
    """The internal gate is a prediction, not truth: a correct fix the gate
    wrongly refuses must be graded by the scorer and counted as a false
    negative (measurable recall), not vanish."""
    import lha.ablation as abl
    from lha.tools.shell import ProcResult

    class _BrokenGateExec:
        """Agent-side backend whose pytest always 'fails' (e.g. broken local env)."""

        name = "broken-gate"

        def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
            if input is not None:
                config = json.loads(input)
                nodeid = "tests/test_m.py::test_f"
                is_collect = config["mode"] == "collect"
                receipt = {
                    "schema_version": 1,
                    "nonce": config["nonce"],
                    "mode": config["mode"],
                    "pytest_exit_code": 0 if is_collect else 1,
                    "collected": [nodeid],
                    "collection_failures": 0,
                    "reports": (
                        []
                        if is_collect
                        else [
                            {
                                "nodeid": nodeid,
                                "when": "call",
                                "outcome": "failed",
                                "wasxfail": False,
                            }
                        ]
                    ),
                }
                payload = json.dumps(
                    receipt,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
                digest = hashlib.sha256(payload).hexdigest()
                (Path(cwd) / config["report_name"]).write_bytes(payload)
                marker = f"LHA_SCORER_RECEIPT {config['nonce']} {digest}\n"
                return ProcResult(receipt["pytest_exit_code"], marker, "", 0.0)
            report = Path(cwd) / ".lha_pytest.json"
            report.write_text(
                json.dumps(
                    {
                        "summary": {"passed": 0, "failed": 1, "error": 0, "total": 1},
                        "tests": [
                            {
                                "nodeid": "tests/test_m.py::test_m",
                                "outcome": "failed",
                                "call": {"longrepr": "E assert False"},
                            }
                        ],
                    }
                )
            )
            return ProcResult(1, "", "simulated agent-env failure", 0.0)

        def python(self):
            return "python"

        def tool(self, name):
            return name

    monkeypatch.setattr(abl, "TrustedLocalBackend", _BrokenGateExec)
    report = _run(tmp_path, _FixedLLM(2))  # a CORRECT fix
    rec = _by_cond(report)
    # gate: claimed False (internal gate failed) but truth True (scorer passed)
    assert rec["gate"].claimed_success is False
    assert rec["gate"].artifact_correct is True
    assert rec["gate"].true_success is False
    assert rec["gate"].false_success is False
    stats = {s.condition: s for s in report.stats}
    assert stats["gate"].fn == 1 and stats["gate"].recall == 0.0
    assert stats["gate"].artifact_correct_rate == 1.0
    assert stats["gate"].true_success_rate == 0.0


def test_scorer_infrastructure_failure_is_not_a_wrong_patch(tmp_path, monkeypatch):
    import lha.ablation as abl

    monkeypatch.setattr(
        abl,
        "_score",
        lambda *args, **kwargs: PytestResult(
            ScoreOutcome.INFRA_ERROR,
            127,
            "scorer: Pytest infrastructure exit 127",
        ),
    )
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    report = run_ablation(
        _base(tmp_path),
        [_task(tmp_path, src)],
        llm="stub",
        reps=1,
        out_dir=out,
        llm_client=_FixedLLM(2),
    )

    assert all(record.status == "ERROR" for record in report.records)
    assert all(record.scorer_outcome == "INFRA_ERROR" for record in report.records)
    assert all(not record.true_success and not record.false_success for record in report.records)
    assert all(stat.n == 0 and stat.errors == 1 for stat in report.stats)
    assert not (out / "results" / "task__r0.json").exists()


@pytest.mark.parametrize(
    ("returncode", "call_outcome", "expected"),
    [
        (0, "passed", ScoreOutcome.PASS),
        (1, "failed", ScoreOutcome.TEST_FAIL),
        (2, "passed", ScoreOutcome.INFRA_ERROR),
        (3, "passed", ScoreOutcome.INFRA_ERROR),
        (5, "passed", ScoreOutcome.INFRA_ERROR),
        (124, "passed", ScoreOutcome.INFRA_ERROR),
        (127, "passed", ScoreOutcome.INFRA_ERROR),
    ],
)
def test_control_plane_scorer_classifies_cross_checked_receipt(
    returncode,
    call_outcome,
    expected,
):
    nodeid = "tests/test_m.py::test_f"
    receipt = {
        "schema_version": 1,
        "nonce": "n" * 48,
        "mode": "run",
        "pytest_exit_code": returncode,
        "collected": [nodeid],
        "collection_failures": 0,
        "reports": [
            {
                "nodeid": nodeid,
                "when": "call",
                "outcome": call_outcome,
                "wasxfail": False,
            }
        ],
    }
    outcome, _passed = _classify_scorer_receipt(
        process_returncode=returncode,
        receipt=receipt,
        expected_nodeids=(nodeid,),
    )
    assert outcome is expected


def test_docker_scorer_also_runs_internal_gate_in_docker(tmp_path, monkeypatch):
    import lha.ablation as abl
    from lha.tools.shell import ProcResult

    instances = []
    image_id = "sha256:" + "a" * 64
    requested_images = []
    docker_provenance = {
        "path": "/fixed/docker",
        "sha256": "d" * 64,
        "size_bytes": 123,
        "trusted_install": True,
    }

    class _DockerIdentity:
        path = "/fixed/docker"

        @staticmethod
        def as_provenance():
            return docker_provenance

    class _RecordingDocker:
        name = "docker"

        def __init__(self, image):
            self.image = image
            self.calls = []
            instances.append(self)

        def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
            self.calls.append(list(cmd))
            if any(abl._DOCKER_IMAGE_PROBE_MARKER in part for part in cmd):
                payload = {
                    "python_version": "3.11.9",
                    "pytest_version": "9.1.1",
                    "pytest_json_report_version": "1.5.0",
                }
                return ProcResult(
                    0,
                    abl._DOCKER_IMAGE_PROBE_MARKER
                    + json.dumps(payload, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    "",
                    0.01,
                )
            if "--json-report" in cmd:
                (Path(cwd) / ".lha_pytest.json").write_text(
                    json.dumps(
                        {
                            "summary": {
                                "passed": 1,
                                "failed": 0,
                                "error": 0,
                                "total": 1,
                            },
                            "tests": [
                                {
                                    "nodeid": "tests/test_m.py::test_m",
                                    "outcome": "passed",
                                }
                            ],
                        }
                    )
                )
                return ProcResult(0, "1 passed in 0.01s\n", "", 0.01)
            config = json.loads(input)
            nodeid = "tests/test_m.py::test_f"
            reports = (
                []
                if config["mode"] == "collect"
                else [
                    {
                        "nodeid": nodeid,
                        "when": "call",
                        "outcome": "passed",
                        "wasxfail": False,
                    }
                ]
            )
            receipt = {
                "schema_version": 1,
                "nonce": config["nonce"],
                "mode": config["mode"],
                "pytest_exit_code": 0,
                "collected": [nodeid],
                "collection_failures": 0,
                "reports": reports,
            }
            payload = json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            digest = hashlib.sha256(payload).hexdigest()
            (Path(cwd) / config["report_name"]).write_bytes(payload)
            marker = f"LHA_SCORER_RECEIPT {config['nonce']} {digest}\n"
            return ProcResult(0, marker, "", 0.01)

        def python(self):
            return "python"

        def tool(self, name):
            return name

    def fake_backend(name, **kwargs):
        assert name == "docker"
        assert kwargs == {
            "image": image_id,
            "docker": "/fixed/docker",
            "operation_lease_dir": tmp_path / "out",
        }
        return _RecordingDocker(kwargs["image"])

    monkeypatch.setattr(abl, "make_backend", fake_backend)
    monkeypatch.setattr(
        abl,
        "resolve_docker_executable",
        lambda _docker="docker": _DockerIdentity(),
    )
    monkeypatch.setattr(
        abl,
        "_inspect_docker_image_id",
        lambda image, *, docker: requested_images.append((image, docker)) or image_id,
    )
    monkeypatch.setattr(
        abl,
        "TrustedLocalBackend",
        lambda: pytest.fail("Docker ablation must not construct a host gate"),
    )
    src = _repo(tmp_path / "src")
    report = run_ablation(
        _base(tmp_path),
        [_task(tmp_path, src)],
        llm="stub",
        reps=1,
        out_dir=tmp_path / "out",
        llm_client=_FixedLLM(2),
        scorer_backend="docker",
    )

    assert len(instances) == 2
    assert requested_images == [(_base(tmp_path).exec_image, "/fixed/docker")]
    assert all(instance.image == image_id for instance in instances)
    assert sum("-I" in command and "-c" in command for command in instances[0].calls) >= 2
    assert all("--json-report" not in command for command in instances[0].calls)
    assert all("--json-report" not in command for command in instances[1].calls)
    assert report.provenance is not None
    assert report.provenance.agent_backend == "docker"
    assert report.provenance.scorer_backend == "docker"
    assert report.provenance.scorer_image == _base(tmp_path).exec_image
    assert report.provenance.scorer_image_id == image_id
    assert report.provenance.docker_executable == docker_provenance
    raw = json.loads((tmp_path / "out" / "ablation_report.json").read_text())
    assert raw["provenance"]["configuration"]["docker_image_probe"] == {
        "schema_version": 1,
        "image_id": image_id,
        "network": "none",
        "minimal_pytest": "passed",
        "python_version": "3.11.9",
        "pytest_version": "9.1.1",
        "pytest_json_report_version": "1.5.0",
    }
    for record in raw["records"]:
        evidence_path = (
            tmp_path / "out" / "scorer_evidence" / f"{record['scorer_evidence_sha256']}.json"
        )
        evidence = json.loads(evidence_path.read_text())
        assert evidence["binding"]["scorer_image_id"] == image_id


def test_docker_image_resolution_failure_precedes_any_model_call(tmp_path, monkeypatch):
    import lha.ablation as abl

    llm = _FixedLLM(2)
    monkeypatch.setattr(
        abl,
        "resolve_docker_executable",
        lambda _docker="docker": (_ for _ in ()).throw(RuntimeError("invalid image binding")),
    )

    with pytest.raises(RuntimeError, match="invalid image binding"):
        run_ablation(
            _base(tmp_path),
            [_task(tmp_path, _repo(tmp_path / "src"))],
            llm="stub",
            reps=1,
            out_dir=tmp_path / "out",
            llm_client=llm,
            scorer_backend="docker",
        )

    assert llm.calls == 0


def test_docker_capability_probe_precedes_any_model_call(tmp_path, monkeypatch):
    import lha.ablation as abl
    from lha.tools.shell import ProcResult

    llm = _FixedLLM(2)
    image_id = "sha256:" + "b" * 64
    provenance = {
        "path": "/fixed/docker",
        "sha256": "d" * 64,
        "size_bytes": 123,
        "trusted_install": False,
    }

    class _DockerIdentity:
        path = "/fixed/docker"

        @staticmethod
        def as_provenance():
            return provenance

    class _UnusableDocker:
        name = "docker"

        def __init__(self, image):
            self.image = image

        def bind_control_plane(self, *, verify_digest=False):
            return provenance

        def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
            return ProcResult(1, "", "pytest-json-report is unavailable", 0.01)

        def python(self):
            return "python"

    monkeypatch.setattr(
        abl,
        "resolve_docker_executable",
        lambda _docker="docker": _DockerIdentity(),
    )
    monkeypatch.setattr(
        abl,
        "_inspect_docker_image_id",
        lambda image, *, docker: image_id,
    )
    monkeypatch.setattr(
        abl,
        "make_backend",
        lambda name, **kwargs: _UnusableDocker(kwargs["image"]),
    )

    with pytest.raises(RuntimeError, match="capability probe"):
        run_ablation(
            _base(tmp_path),
            [_task(tmp_path, _repo(tmp_path / "src"))],
            llm="stub",
            reps=1,
            out_dir=tmp_path / "out",
            llm_client=llm,
            scorer_backend="docker",
        )

    assert llm.calls == 0


def test_docker_image_resolution_keeps_only_required_host_configuration(
    monkeypatch,
):
    import lha.ablation as abl
    from lha.tools.shell import ProcResult

    image_id = "sha256:" + "b" * 64
    observed = {}
    monkeypatch.setenv("DOCKER_CONFIG", "/tmp/lha-test-docker-config")
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/lha-test-docker.sock")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-docker")

    def recording_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed.update(kwargs)
        return ProcResult(0, image_id + "\n", "", 0.0)

    monkeypatch.setattr(abl, "run", recording_run)
    monkeypatch.setattr(
        abl,
        "resolve_docker_executable",
        lambda _docker="docker": type(
            "DockerIdentity",
            (),
            {"path": "/fixed/docker"},
        )(),
    )

    assert abl._resolve_docker_image_id("lha:test") == image_id
    assert observed["cmd"][0] == "/fixed/docker"
    assert observed["env"]["DOCKER_CONFIG"] == "/tmp/lha-test-docker-config"
    assert observed["env"]["DOCKER_HOST"] == "unix:///tmp/lha-test-docker.sock"
    assert "OPENAI_API_KEY" not in observed["env"]


def test_formal_container_absence_audit_fails_on_owned_residue(monkeypatch):
    import lha.ablation as abl
    from lha.tools.shell import ProcResult

    observed = {}

    def recording_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed.update(kwargs)
        return ProcResult(0, "a" * 64 + "\n", "", 0.0)

    monkeypatch.setattr(abl, "run", recording_run)

    with pytest.raises(RuntimeError, match="still has LHA-owned containers"):
        abl._assert_no_lha_containers("/fixed/docker")

    assert observed["cmd"][0] == "/fixed/docker"
    assert observed["cmd"][-1] == "label=lha.operation_id"


def test_pinned_docker_backend_argv_uses_image_id_not_mutable_tag(tmp_path):
    from lha.sandbox import DockerBackend

    image_id = "sha256:" + "b" * 64
    mutable_tag = "lha:release"
    backend = DockerBackend(image=image_id)
    argv = backend.build_argv(
        ["python", "-V"],
        cwd=tmp_path,
        name="lha-test",
    )

    assert image_id in argv
    assert mutable_tag not in argv


def test_frozen_diff_excludes_oracle_and_junk(tmp_path):
    from lha.ablation import _frozen_diff

    src = _repo(tmp_path / "src")
    wd = tmp_path / "wd"
    import shutil as _sh

    _sh.copytree(src, wd)
    (wd / "m.py").write_text("def f():\n    return 2\n")  # source change
    (wd / "new_helper.py").write_text("x = 1\n")  # added file
    (wd / "tests" / "test_m.py").write_text("tampered")  # protected -> excluded
    (wd / "__pycache__").mkdir()
    (wd / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"junk")

    frozen = _frozen_diff(src, wd)
    assert set(frozen) == {"m.py", "new_helper.py"}


def test_frozen_diff_records_deletions(tmp_path):
    from lha.ablation import _frozen_diff

    src = _repo(tmp_path / "src")
    (src / "todelete.py").write_text("gone = 1\n")
    wd = tmp_path / "wd"
    import shutil as _sh

    _sh.copytree(src, wd)
    (wd / "todelete.py").unlink()
    frozen = _frozen_diff(src, wd)
    assert frozen == {"todelete.py": None}


def test_confusion_matrix_measures_false_positives():
    """A gate that passes a wrong fix (flaky oracle, dirty env) shows up as FP;
    nothing in the aggregation forces FP to zero."""
    records = [
        RunRecord("t1", "gate", 0, "DONE", True, False, False, True, 0, "", True, "sha1"),
        RunRecord("t2", "gate", 0, "DONE", True, True, True, False, 0, "", True, "sha2"),
        RunRecord("t3", "gate", 0, "FAILED", False, True, False, False, 0, "", False, "sha3"),
    ]
    stats = {s.condition: s for s in _aggregate(records)}
    g = stats["gate"]
    assert (g.tp, g.fp, g.tn, g.fn) == (1, 1, 0, 1)
    assert g.precision == 0.5 and g.recall == 0.5
    assert g.false_success_rate == 1 / 3  # the FP is a counted false success


def test_cache_busts_when_task_changes(tmp_path):
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    base = _base(tmp_path)
    run_ablation(base, [task], reps=1, out_dir=out, llm_client=_FixedLLM(2))
    assert (out / "results" / "task__r0.json").exists()

    # same cache dir, but the task definition changed -> fingerprint mismatch ->
    # the cell recomputes (and here the failing LLM makes that observable).
    Path(task).write_text(Path(task).read_text().replace("wrong value", "other value"))
    rep2 = run_ablation(base, [task], reps=1, out_dir=out, llm_client=_FailingLLM())
    assert all(r.status == "ERROR" for r in rep2.records)


@pytest.mark.parametrize(
    "source_path",
    [
        "ablation.py",
        "verifiers/code/pytest_verifier.py",
        "tools/policy.py",
        "tools/patch.py",
        "sandbox/local.py",
        "bench/stats.py",
    ],
)
def test_cache_fingerprint_includes_all_outcome_source(tmp_path, source_path):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    source_files = abl._source_file_digests()
    assert source_path in source_files
    original = abl._fingerprint(task, src, "stub", None, source_files=source_files)
    changed_files = dict(source_files)
    changed_files[source_path] = "0" * 64
    changed = abl._fingerprint(task, src, "stub", None, source_files=changed_files)
    assert changed != original


def test_cache_fingerprint_binds_scorer_and_runtime(tmp_path):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    source_files = abl._source_file_digests()
    local = abl._fingerprint(
        task,
        src,
        "stub",
        None,
        "trusted-local",
        source_files=source_files,
        runtime={"scorer": {"actual": "trusted-local", "image_id": None}},
    )
    docker = abl._fingerprint(
        task,
        src,
        "stub",
        None,
        "docker",
        source_files=source_files,
        runtime={"scorer": {"actual": "docker", "image_id": "sha256:" + "a" * 64}},
    )
    assert docker != local


def test_report_rejects_source_tree_drift_during_run(tmp_path, monkeypatch):
    import lha.ablation as abl

    initial = abl._source_file_digests()
    calls = 0

    def source_file_digests():
        nonlocal calls
        calls += 1
        if calls == 1:
            return initial
        changed = dict(initial)
        changed["ablation.py"] = "0" * 64
        return changed

    monkeypatch.setattr(abl, "_source_file_digests", source_file_digests)
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)

    with pytest.raises(RuntimeError, match="source tree changed during the ablation"):
        run_ablation(
            _base(tmp_path),
            [task],
            reps=1,
            out_dir=out,
            llm_client=_FixedLLM(2),
        )

    assert calls == 2
    assert not (out / "ablation_report.json").exists()
    assert not (out / "ablation_report.md").exists()


def test_formal_report_json_is_last_commit_marker(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    out = tmp_path / "out"
    out.mkdir()
    source_files = {"runtime.py": "1" * 64}
    binding = abl._FormalCorpusBinding(
        path="benchmarks/formal_ablation_manifest.json",
        sha256="2" * 64,
        preregistration_commit="3" * 40,
        git_executable={"path": "/usr/bin/git"},
    )
    revalidations = 0

    def revalidate(_binding):
        nonlocal revalidations
        revalidations += 1
        return source_files

    original_write = abl._atomic_write
    writes: list[str] = []

    def fail_json(path, text, *, anchor=None):
        writes.append(path.name)
        if path.name == "ablation_report.json":
            raise RuntimeError("injected report commit failure")
        original_write(path, text, anchor=anchor)

    monkeypatch.setattr(abl, "_revalidate_formal_checkout", revalidate)
    monkeypatch.setattr(abl, "_atomic_write", fail_json)

    with pytest.raises(RuntimeError, match="commit failure"):
        abl._write_ablation_reports(
            out,
            report_json='{"schema_version":4}',
            report_markdown="# report\n",
            formal_corpus=binding,
            source_files=source_files,
        )

    assert writes == ["ablation_report.md", "ablation_report.json"]
    assert revalidations == 2
    assert (out / "ablation_report.md").read_text() == "# report\n"
    assert not (out / "ablation_report.json").exists()


def test_completed_formal_output_directory_refuses_rerun(tmp_path, monkeypatch):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    out = tmp_path / "out"
    out.mkdir()
    records = [
        RunRecord(
            task="task",
            condition=condition,
            rep=0,
            status="ERROR",
            claimed_success=False,
            artifact_correct=False,
            true_success=False,
            false_success=False,
            repairs=0,
            scorer_outcome="INFRA_ERROR",
        )
        for condition, _ in CONDITIONS
    ]
    raw = {
        "schema_version": 4,
        "tasks": ["task"],
        "reps": 1,
        "records": [record.__dict__ for record in records],
        "fingerprint": "",
    }
    raw["fingerprint"] = abl._report_fingerprint(raw)
    (out / "ablation_report.json").write_text(json.dumps(raw))
    monkeypatch.setattr(abl, "_prepare_formal_corpus_binding", lambda *args, **kwargs: object())

    with pytest.raises(RuntimeError, match="already contains ablation_report.json"):
        run_ablation(
            _base(tmp_path),
            [task],
            llm="codex_cli",
            model="model-x",
            reps=1,
            out_dir=out,
            llm_client=None,
            scorer_backend="docker",
        )


def test_invalid_formal_report_file_is_not_overwritten(tmp_path, monkeypatch):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    out = tmp_path / "out"
    out.mkdir()
    report = out / "ablation_report.json"
    report.write_text("{incomplete")
    monkeypatch.setattr(abl, "_prepare_formal_corpus_binding", lambda *args, **kwargs: object())

    with pytest.raises(RuntimeError, match="reports are immutable"):
        run_ablation(
            _base(tmp_path),
            [task],
            llm="codex_cli",
            model="model-x",
            reps=1,
            out_dir=out,
            llm_client=None,
            scorer_backend="docker",
        )
    assert report.read_text() == "{incomplete"


def test_legacy_cache_format_is_recomputed(tmp_path):
    import lha.ablation as abl

    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    (out / "results").mkdir(parents=True)
    # pre-fingerprint cache format: a bare list
    (out / "results" / "task__r0.json").write_text("[]")
    assert abl._read_cache(out / "results" / "task__r0.json") == (None, [])
    rep = run_ablation(_base(tmp_path), [task], reps=1, out_dir=out, llm_client=_FixedLLM(2))
    assert _by_cond(rep)["trust"].true_success  # recomputed, not served stale


def test_report_shows_gate_quality_and_scorer(tmp_path):
    report = _run(tmp_path, _FixedLLM(2))
    md = report.to_markdown()
    assert "final scorer" in md
    assert "precision" in md and "recall" in md and "FP=" in md
    assert report.scorer == "trusted-local"
    assert report.fingerprint


def test_new_report_records_complete_secret_free_provenance(tmp_path):
    import lha.ablation as abl

    report = _run(tmp_path, _FixedLLM(2))
    raw = json.loads((tmp_path / "out" / "ablation_report.json").read_text())
    provenance = raw["provenance"]

    assert raw["schema_version"] == 4
    assert provenance["source_tree_sha256"] == report.provenance.source_tree_sha256
    assert provenance["source_files"]["ablation.py"]
    assert provenance["source_files"]["verifiers/code/pytest_verifier.py"]
    assert provenance["source_files"]["tools/policy.py"]
    assert provenance["source_files"]["tools/patch.py"]
    assert provenance["requested_llm_backend"] == "stub"
    assert provenance["actual_llm_backend"] == "base"
    assert provenance["model"] is None
    assert provenance["cli_version"] is None
    assert provenance["backend_library_version"] is None
    assert provenance["reasoning_effort"] is None
    assert provenance["scorer_requested"] == "trusted-local"
    assert provenance["scorer_backend"] == "trusted-local"
    assert provenance["task_files_sha256"]["task"]
    assert provenance["corpus_sha256"]["task"]
    assert provenance["input_snapshot_sha256"]["task"]
    assert provenance["task_paths"]["task"].endswith("task.yaml")
    assert provenance["corpus_paths"]["task"].endswith("src")
    assert provenance["configuration"]["repetitions"] == 1
    assert provenance["git_commit"] is None or len(provenance["git_commit"]) == 40
    assert provenance["git_dirty"] in (True, False, None)
    assert "auth" not in json.dumps(provenance).lower()
    assert raw["artifact_store"]["path"] == "artifacts"
    assert raw["scorer_evidence_store"]["path"] == "scorer_evidence"
    assert raw["scorer_evidence_store"]["schema_version"] == 2
    assert provenance["configuration"]["cache_schema"] == 8
    assert provenance["configuration"]["scorer_evidence_schema"] == 2
    digests = {record["artifact_sha256"] for record in raw["records"]}
    assert raw["artifact_store"]["count"] == len(digests)
    for digest in digests:
        artifact = tmp_path / "out" / "artifacts" / f"{digest}.json"
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    evidence_digests = {record["scorer_evidence_sha256"] for record in raw["records"]}
    assert raw["scorer_evidence_store"]["count"] == len(evidence_digests)
    for digest in evidence_digests:
        evidence = tmp_path / "out" / "scorer_evidence" / f"{digest}.json"
        assert evidence.is_file()
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == digest
        envelope = json.loads(evidence.read_text())
        assert envelope["schema_version"] == 2
        assert envelope["pytest_evidence"]["schema_version"] == 1
        assert envelope["binding"]["task"] == "task"
        assert envelope["binding"]["rep"] == 0
        assert (
            envelope["binding"]["input_snapshot_sha256"]
            == provenance["input_snapshot_sha256"]["task"]
        )
        assert envelope["binding"]["scorer_backend"] == "trusted-local"
        assert envelope["binding"]["scorer_image_id"] is None
    assert len(raw["llm_calls"]) == 1
    assert {
        name: raw["llm_calls"][0][name]
        for name in (
            "task",
            "rep",
            "cache_hit",
            "label",
            "status",
            "backend",
        )
    } == {
        "task": "task",
        "rep": 0,
        "cache_hit": False,
        "label": "first",
        "status": "succeeded",
        "backend": "base",
    }
    assert len(raw["llm_calls"][0]["patch_sha256"]) == 64
    assert raw["llm_calls"][0]["result_artifact_sha256"] == raw["records"][0]["artifact_sha256"]

    loaded = abl.load_ablation_report(tmp_path / "out" / "ablation_report.json")
    assert loaded.schema_version == 4
    assert loaded.provenance is not None
    assert loaded.provenance.source_tree_sha256 == provenance["source_tree_sha256"]


def test_old_report_without_provenance_remains_readable(tmp_path):
    import lha.ablation as abl

    old = tmp_path / "old-report.json"
    old.write_text(
        json.dumps(
            {
                "llm": "claude_cli",
                "model": "old-model",
                "reps": 1,
                "tasks": ["old-task"],
                "scorer": "trusted-local",
                "records": [],
                "stats": [],
            }
        )
    )
    loaded = abl.load_ablation_report(old)
    assert loaded.schema_version == 1
    assert loaded.provenance is None
    assert loaded.tasks == ["old-task"]


def test_codex_runtime_and_call_audit_are_structured_and_secret_free():
    import lha.ablation as abl
    from lha.llm.codex_cli import CodexCLIClient

    client = CodexCLIClient(
        model="gpt-test-snapshot",
        reasoning_effort="high",
        no_tools=True,
    )
    client._version = "codex-cli 1.2.3"
    runtime = abl._client_runtime(
        "codex_cli",
        client,
        model="gpt-test-snapshot",
        cli_path="codex",
        backend_details="codex-cli 1.2.3",
    )
    assert runtime["actual_backend"] == "codex_cli"
    assert runtime["model"] == "gpt-test-snapshot"
    assert runtime["cli_version"] == "codex-cli 1.2.3"
    assert runtime["reasoning_effort"] == "high"

    client.last_call = {
        "status": "failed",
        "cli_version": "codex-cli 1.2.3",
        "model": "gpt-test-snapshot",
        "reasoning_effort": "high",
        "attempt_count": 1,
        "event_summary": {"total_events": 3, "events": {"turn.failed": 1}},
        "error_type": "CodexProtocolError",
        "error": "must-not-be-persisted",
        "attempts": [
            {
                "attempt": 1,
                "status": "failed",
                "event_summary": {"total_events": 3},
                "error": "must-not-be-persisted",
            }
        ],
    }
    client.last_usage = {
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "output_tokens": 3,
        "cost_usd": None,
        "model": "gpt-test-snapshot",
    }
    audit = abl._safe_call_audit(
        client,
        label="first",
        status="failed",
        error=RuntimeError("must-not-be-persisted"),
    )
    assert audit["event_summary"]["total_events"] == 3
    assert audit["status"] == "failed"
    assert audit["usage"]["output_tokens"] == 3
    assert "must-not-be-persisted" not in json.dumps(audit)
