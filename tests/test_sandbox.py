"""Execution backends: env scrubbing, process-tree kill, limits, docker argv.

The docker integration test is opt-in (LHA_DOCKER_TESTS=1 and a working
daemon); everything else runs hermetically on the host.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

import lha.sandbox.base as sandbox_base
import lha.sandbox.docker as docker_backend
import lha.sandbox.local as local_backend
from lha.operation_lease import OperationLeaseStore
from lha.sandbox import DockerBackend, ResourceLimits, TrustedLocalBackend, scrub_env
from lha.sandbox.base import (
    ProcessCleanupResult,
    ProcessCleanupUnconfirmed,
)
from lha.tools.shell import ProcResult


def test_scrub_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("LHA_SECRET_PROBE", "s3cr3t")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws")
    env = scrub_env()
    assert "LHA_SECRET_PROBE" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "PATH" in env


def test_local_backend_runs_and_captures(tmp_path):
    res = TrustedLocalBackend().run(
        [sys.executable, "-c", "print('hello'); import sys; sys.exit(3)"], cwd=tmp_path
    )
    assert res.stdout.strip() == "hello"
    assert res.returncode == 3


def test_local_backend_drains_but_bounds_both_output_streams(tmp_path):
    limit = 4096
    res = TrustedLocalBackend(ResourceLimits(output_bytes=limit)).run(
        [
            sys.executable,
            "-c",
            ("import os\nos.write(1, b'o' * 200_000)\nos.write(2, b'e' * 200_000)\n"),
        ],
        cwd=tmp_path,
        timeout=10,
    )

    assert res.returncode == 125
    assert res.output_truncated is True
    assert len(res.stdout.encode()) <= limit
    assert len(res.stderr.encode()) <= limit
    assert "capture limit" in res.stderr


def test_output_capture_limit_must_be_positive():
    with pytest.raises(ValueError, match="output_bytes"):
        ResourceLimits(output_bytes=0)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_bounded_process_timeout_must_be_positive_and_finite(
    tmp_path,
    timeout,
    monkeypatch,
):
    def unexpected_spawn(*_args, **_kwargs):
        raise AssertionError("invalid timeout must fail before spawning")

    monkeypatch.setattr(sandbox_base.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(ValueError, match="timeout"):
        sandbox_base.run_bounded_process(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            timeout=timeout,
            output_bytes=1024,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("cpu_s", 0), ("memory_mb", -1), ("pids", True)],
)
def test_process_resource_limits_must_be_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        ResourceLimits(**{field: value})


def test_local_backend_child_env_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("LHA_SECRET_PROBE", "s3cr3t")
    res = TrustedLocalBackend().run(
        [sys.executable, "-c", "import os; print(os.environ.get('LHA_SECRET_PROBE', 'none'))"],
        cwd=tmp_path,
    )
    assert res.stdout.strip() == "none"


def test_local_backend_kills_whole_process_tree_on_timeout(tmp_path):
    marker = tmp_path / "marker"
    # the grandchild would create the marker ~2s after the timeout kill
    cmd = [
        "sh",
        "-c",
        f"(sleep 2 && touch {marker}) & sleep 30",
    ]
    start = time.monotonic()
    res = TrustedLocalBackend().run(cmd, cwd=tmp_path, timeout=0.5)
    assert res.returncode == 124
    assert time.monotonic() - start < 10
    time.sleep(3)
    assert not marker.exists(), "grandchild survived the process-group kill"


def test_local_backend_stops_descendant_that_holds_output_pipe(tmp_path):
    marker = tmp_path / "orphan-marker"
    res = TrustedLocalBackend().run(
        ["sh", "-c", f"(sleep 2 && touch {marker}) & exit 0"],
        cwd=tmp_path,
        timeout=10,
    )

    assert res.returncode == 0
    assert res.output_truncated is False
    time.sleep(2.5)
    assert not marker.exists(), "descendant survived after its parent exited"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_local_backend_stops_descendant_that_closed_output_pipe(tmp_path, exit_code):
    marker = tmp_path / f"closed-pipe-marker-{exit_code}"
    res = TrustedLocalBackend().run(
        [
            "sh",
            "-c",
            f"(exec >/dev/null 2>&1; sleep 1; touch {marker}) & exit {exit_code}",
        ],
        cwd=tmp_path,
        timeout=10,
    )

    assert res.returncode == exit_code
    time.sleep(1.5)
    assert not marker.exists(), "stdio-independent descendant survived leader exit"


def test_local_backend_fails_when_group_cleanup_cannot_be_confirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_backend,
        "terminate_process_group",
        lambda _process: ProcessCleanupResult(False, "probe denied"),
    )

    res = TrustedLocalBackend().run(
        [sys.executable, "-c", "print('finished')"],
        cwd=tmp_path,
    )

    assert res.returncode == 126
    assert res.ok is False
    assert res.stdout.strip() == "finished"
    assert "cleanup could not be confirmed" in res.stderr
    assert "probe denied" in res.stderr


def test_local_backend_rejects_unsupported_process_groups_before_spawn(tmp_path, monkeypatch):
    def unexpected_spawn(*_args, **_kwargs):
        raise AssertionError("unsupported hosts must fail before spawning target code")

    monkeypatch.setattr(
        local_backend,
        "process_group_cleanup_supported",
        lambda: False,
    )
    monkeypatch.setattr(local_backend, "run_bounded_process", unexpected_spawn)

    res = TrustedLocalBackend().run(
        [sys.executable, "-c", "print('must not run')"],
        cwd=tmp_path,
    )

    assert res.returncode == 126
    assert res.ok is False
    assert "requires POSIX process-group cleanup" in res.stderr


def test_process_group_cleanup_retries_transient_permission_error(monkeypatch):
    class Process:
        pid = 4312

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return -9

    process = Process()
    probes = iter(
        [
            PermissionError(1, "Operation not permitted"),
            ProcessLookupError(3, "No such process"),
        ]
    )

    def killpg(_process_group, sig):
        if sig == 0:
            raise next(probes)

    monkeypatch.setattr(sandbox_base.os, "killpg", killpg)

    result = sandbox_base.terminate_process_group(process)

    assert result.confirmed is True
    assert process.polls == 2


def test_process_group_cleanup_fails_closed_on_persistent_permission_error(
    monkeypatch,
):
    class Process:
        pid = 4313

        @staticmethod
        def poll():
            return -9

    def killpg(_process_group, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(sandbox_base.os, "killpg", killpg)
    monkeypatch.setattr(
        sandbox_base,
        "read_process_group_census",
        lambda _pgid: sandbox_base.ProcessGroupCensus(
            error="process table unavailable"
        ),
    )

    result = sandbox_base.terminate_process_group(
        Process(),
        confirmation_timeout_s=0.0,
    )

    assert result.confirmed is False
    assert "could not confirm process group 4313 cleanup" in result.detail
    assert "Operation not permitted" in result.detail


def test_process_group_cleanup_accepts_permission_error_for_zombies_only(
    monkeypatch,
):
    class Process:
        pid = 4314

        @staticmethod
        def poll():
            return -9

    def killpg(_process_group, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(sandbox_base.os, "killpg", killpg)
    monkeypatch.setattr(
        sandbox_base,
        "read_process_group_census",
        lambda pgid: sandbox_base.ProcessGroupCensus(
            (
                sandbox_base.ProcessGroupMember(
                    pid=pgid + 1,
                    pgid=pgid,
                    uid=os.geteuid(),
                    state="Z",
                ),
            )
        ),
    )

    result = sandbox_base.terminate_process_group(
        Process(),
        confirmation_timeout_s=0.0,
    )

    assert result.confirmed is True
    assert result.detail == "process group 4314 has only zombie members"


def test_process_group_cleanup_rejects_same_user_runnable_member_after_eperm(
    monkeypatch,
):
    class Process:
        pid = 4315

        @staticmethod
        def poll():
            return -9

    def killpg(_process_group, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(sandbox_base.os, "killpg", killpg)
    monkeypatch.setattr(
        sandbox_base,
        "read_process_group_census",
        lambda pgid: sandbox_base.ProcessGroupCensus(
            (
                sandbox_base.ProcessGroupMember(
                    pid=pgid + 1,
                    pgid=pgid,
                    uid=os.geteuid(),
                    state="S",
                ),
            )
        ),
    )

    result = sandbox_base.terminate_process_group(
        Process(),
        confirmation_timeout_s=0.0,
    )

    assert result.confirmed is False
    assert "same-user runnable members remain" in result.detail
    assert "4316" in result.detail


def test_process_group_cleanup_accepts_reused_pgid_after_original_leader_exit(
    monkeypatch,
):
    class Process:
        pid = 4317

        @staticmethod
        def poll():
            return -9

    def killpg(_process_group, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(sandbox_base.os, "killpg", killpg)
    monkeypatch.setattr(
        sandbox_base,
        "read_process_group_census",
        lambda pgid: sandbox_base.ProcessGroupCensus(
            (
                sandbox_base.ProcessGroupMember(
                    pid=pgid,
                    pgid=pgid,
                    uid=os.geteuid(),
                    state="R",
                ),
            )
        ),
    )

    result = sandbox_base.terminate_process_group(
        Process(),
        confirmation_timeout_s=0.0,
    )

    assert result.confirmed is True
    assert "was reused after the original leader exited" in result.detail


def test_process_group_cleanup_rechecks_empty_census_before_accepting_absence(
    monkeypatch,
):
    class Process:
        pid = 4318

        @staticmethod
        def poll():
            return -9

    probes = iter(
        [
            PermissionError(1, "Operation not permitted"),
            ProcessLookupError(3, "No such process"),
        ]
    )

    def killpg(_process_group, sig):
        if sig == 0:
            raise next(probes)

    monkeypatch.setattr(sandbox_base.os, "killpg", killpg)
    monkeypatch.setattr(
        sandbox_base,
        "read_process_group_census",
        lambda _pgid: sandbox_base.ProcessGroupCensus(),
    )

    result = sandbox_base.terminate_process_group(
        Process(),
        confirmation_timeout_s=0.0,
    )

    assert result.confirmed is True
    assert result.detail == "process group is absent"


def test_darwin_process_group_census_uses_exact_pgid_and_bounded_capture(
    monkeypatch,
):
    observed = {}

    def bounded_ps(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return ProcResult(
            0,
            "4319 4319 501 Z\n",
            "",
            0.01,
        )

    monkeypatch.setattr(sandbox_base.sys, "platform", "darwin")
    monkeypatch.setattr(
        sandbox_base.Path,
        "is_file",
        lambda path: str(path) == "/bin/ps",
    )
    monkeypatch.setattr(
        sandbox_base.os,
        "access",
        lambda path, _mode: str(path) == "/bin/ps",
    )
    monkeypatch.setattr(sandbox_base, "run_bounded_process", bounded_ps)

    census = sandbox_base.read_process_group_census(4319)

    assert observed["argv"] == [
        "/bin/ps",
        "-g",
        "4319",
        "-o",
        "pid=,pgid=,uid=,state=",
    ]
    assert observed["kwargs"]["timeout"] == 1.0
    assert observed["kwargs"]["output_bytes"] == 4 * 1024 * 1024
    assert census.error is None
    assert census.members == (
        sandbox_base.ProcessGroupMember(
            pid=4319,
            pgid=4319,
            uid=501,
            state="Z",
        ),
    )


def test_local_backend_uses_exec_launcher_instead_of_preexec(tmp_path, monkeypatch):
    observed = []

    def recording_runner(cmd, **kwargs):
        observed.append((cmd, kwargs))
        return ProcResult(0, "", "", 0.0)

    monkeypatch.setattr(local_backend, "run_bounded_process", recording_runner)
    backend = TrustedLocalBackend()
    original = [sys.executable, "-c", "print('ok')"]

    backend.run(original, cwd=tmp_path)
    backend.run(
        original,
        cwd=tmp_path,
        limits=ResourceLimits(cpu_s=2, memory_mb=128, pids=16),
    )

    assert observed[0][0] == original
    assert "preexec_fn" not in observed[0][1]
    limited = observed[1][0]
    assert limited[:3] == [sys.executable, "-m", "lha.sandbox.limit_exec"]
    assert limited[-len(original) :] == original
    assert ["--cpu-s", "2"] == limited[3:5]
    assert "preexec_fn" not in observed[1][1]


@pytest.mark.parametrize("failed_start", [2, 3])
def test_bounded_process_closes_pipes_when_thread_start_fails(failed_start, tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("process-group assertion requires POSIX")

    real_popen = sandbox_base.subprocess.Popen
    real_start = sandbox_base.threading.Thread.start
    processes = []
    start_calls = 0

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def failing_start(thread):
        nonlocal start_calls
        start_calls += 1
        if start_calls == failed_start:
            raise RuntimeError(f"thread start {failed_start} failed")
        return real_start(thread)

    monkeypatch.setattr(sandbox_base.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(sandbox_base.threading.Thread, "start", failing_start)

    with pytest.raises(RuntimeError, match=f"thread start {failed_start} failed"):
        sandbox_base.run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            timeout=10,
            output_bytes=1024,
            input="payload",
            start_new_session=True,
            on_exit=sandbox_base.terminate_process_group,
        )

    assert len(processes) == 1
    process = processes[0]
    assert process.poll() is not None
    assert process.stdin is not None and process.stdin.closed
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_interruption_records_unconfirmed_process_cleanup(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("process-group assertion requires POSIX")

    real_start = sandbox_base.threading.Thread.start
    real_cleanup = sandbox_base.terminate_process_group
    start_calls = 0

    def interrupt_first_start(thread):
        nonlocal start_calls
        start_calls += 1
        if start_calls == 1:
            raise KeyboardInterrupt("cancelled during pipe setup")
        return real_start(thread)

    def unconfirmed_cleanup(process):
        real_cleanup(process)
        return ProcessCleanupResult(False, "simulated cleanup uncertainty")

    monkeypatch.setattr(
        sandbox_base.threading.Thread,
        "start",
        interrupt_first_start,
    )

    with pytest.raises(ProcessCleanupUnconfirmed) as caught:
        sandbox_base.run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            timeout=10,
            output_bytes=1024,
            input="payload",
            start_new_session=True,
            on_exit=unconfirmed_cleanup,
        )

    assert caught.value.detail == "simulated cleanup uncertainty"
    assert isinstance(caught.value.__cause__, KeyboardInterrupt)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits")
def test_local_backend_cpu_limit_stops_spin(tmp_path):
    res = TrustedLocalBackend().run(
        [sys.executable, "-c", "while True: pass"],
        cwd=tmp_path,
        timeout=30,
        limits=ResourceLimits(cpu_s=1),
    )
    assert res.returncode != 0  # SIGXCPU (or platform equivalent), never success


# --- docker argv construction (no daemon needed) ------------------------------
def test_docker_control_executable_is_absolute_and_digest_bound(tmp_path):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n")
    fake_docker.chmod(0o755)
    backend = DockerBackend(docker=str(fake_docker))

    provenance = backend.bind_control_plane(verify_digest=True)

    assert provenance["path"] == str(fake_docker.resolve())
    assert len(str(provenance["sha256"])) == 64
    assert backend.docker == str(fake_docker.resolve())

    fake_docker.write_text("#!/bin/sh\nexit 1\n")
    with pytest.raises(RuntimeError, match="changed"):
        backend.bind_control_plane(verify_digest=True)


def test_execution_backend_refuses_to_switch_away_from_pending_run_leases(
    tmp_path,
):
    first = tmp_path / "first-run"
    second = tmp_path / "second-run"
    first.mkdir()
    second.mkdir()
    store = OperationLeaseStore(first)
    store.prepare_local([sys.executable, "-c", "pass"], cwd=tmp_path)
    backend = TrustedLocalBackend(operation_lease_dir=first)

    with pytest.raises(RuntimeError, match="pending operations"):
        backend.bind_operation_lease_dir(second)

    assert backend.operation_lease_dir == first.resolve()
    assert len(store.list()) == 1


def test_docker_argv_isolation_flags(tmp_path):
    be = DockerBackend(
        image="img:1",
        limits=ResourceLimits(memory_mb=512, pids=64),
        ro_mounts={str(tmp_path / "base"): "/base"},
    )
    argv = be.build_argv([sys.executable, "-m", "pytest"], cwd=tmp_path, name="lha-test")
    joined = " ".join(argv)
    assert "--network none" in joined
    assert "--read-only" in argv
    assert argv[argv.index("--tmpfs") + 1] == "/tmp:rw,nosuid,nodev,size=256m,mode=1777"
    assert "--memory 512m" in joined
    assert "--pids-limit 64" in joined
    assert f"{tmp_path.resolve()}:/work" in joined
    assert f"{(tmp_path / 'base').resolve()}:/base:ro" in joined
    # host interpreter path translated for the container
    assert sys.executable not in argv
    assert argv[-3:] == ["python", "-m", "pytest"]
    # no environment forwarding beyond the explicit HOME
    assert joined.count("--env") == 1


def test_docker_timeout_preserves_nonzero_cleanup_failure(tmp_path, monkeypatch):
    invocations: list[list[str]] = []
    monkeypatch.setattr(
        docker_backend,
        "terminate_process_group",
        lambda _process: ProcessCleanupResult(True, "Docker client stopped"),
    )

    def fake_bounded(argv, **kwargs):
        invocations.append(argv)
        if argv[1] == "run":
            cleanup = kwargs["on_timeout"](None)
            return ProcResult(
                124,
                "partial output",
                f"docker run stalled\ntimeout after 0.1s ({cleanup})",
                0.1,
            )
        return ProcResult(
            1,
            "",
            "daemon refused to remove the container",
            0.01,
        )

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)
    result = DockerBackend(image="img:1").run(["python", "-V"], cwd=tmp_path, timeout=0.1)

    assert result.returncode == 126
    assert result.cleanup_unconfirmed is True
    assert result.cleanup_confirmed is False
    assert "process cleanup could not be confirmed" in result.stderr
    assert result.stdout == "partial output"
    assert "docker run stalled" in result.stderr
    assert "cleanup failed with exit code 1" in result.stderr
    assert "daemon refused to remove" in result.stderr
    assert "container removed" not in result.stderr
    assert invocations[1][1:3] == ["rm", "-f"]


def test_docker_timeout_returns_a_result_when_cleanup_also_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(
        docker_backend,
        "terminate_process_group",
        lambda _process: ProcessCleanupResult(True, "Docker client stopped"),
    )

    def fake_bounded(argv, **kwargs):
        if argv[1] == "run":
            cleanup = kwargs["on_timeout"](None)
            return ProcResult(
                124,
                "",
                f"timeout after 0.1s ({cleanup})",
                0.1,
            )
        return ProcResult(
            124,
            "",
            "daemon did not answer\ntimeout after 30.0s (process killed)",
            30.0,
        )

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)
    result = DockerBackend(image="img:1").run(["python", "-V"], cwd=tmp_path, timeout=0.1)

    assert result.returncode == 126
    assert result.cleanup_unconfirmed is True
    assert result.cleanup_confirmed is False
    assert "cleanup timed out after 30s" in result.stderr
    assert "daemon did not answer" in result.stderr
    assert "may still be running" in result.stderr
    assert "container removed" not in result.stderr


def test_docker_timeout_keeps_124_when_daemon_confirms_container_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        docker_backend,
        "terminate_process_group",
        lambda _process: ProcessCleanupResult(True, "Docker client stopped"),
    )

    def fake_bounded(argv, **kwargs):
        if argv[1] == "run":
            cleanup = kwargs["on_timeout"](None)
            return ProcResult(
                124,
                "",
                f"timeout after 0.1s ({cleanup.detail})",
                0.1,
            )
        if argv[1:3] == ["rm", "-f"]:
            return ProcResult(1, "", "No such container", 0.01)
        assert argv[1:3] == ["container", "ls"]
        return ProcResult(0, "", "", 0.01)

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)

    result = DockerBackend(image="img:1").run(
        ["python", "-V"],
        cwd=tmp_path,
        timeout=0.1,
    )

    assert result.returncode == 124
    assert result.cleanup_unconfirmed is False


def test_docker_nonzero_exit_becomes_126_when_absence_probe_fails(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_bounded(argv, **_kwargs):
        calls.append(argv)
        if argv[1] == "run":
            return ProcResult(2, "", "target failed", 0.1)
        if argv[1:3] == ["rm", "-f"]:
            return ProcResult(1, "", "daemon refused removal", 0.01)
        assert argv[1:3] == ["container", "ls"]
        return ProcResult(1, "", "daemon unavailable", 0.01)

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)

    result = DockerBackend(image="img:1").run(
        ["python", "-V"],
        cwd=tmp_path,
    )

    assert result.returncode == 126
    assert result.cleanup_unconfirmed is True
    assert result.cleanup_confirmed is False
    assert "target failed" in result.stderr
    assert "daemon unavailable" in result.cleanup_detail
    assert [call[1:3] for call in calls] == [
        ["run", "--rm"],
        ["rm", "-f"],
        ["container", "ls"],
    ]


def test_docker_preserves_a_real_target_exit_126_when_cleanup_is_confirmed(tmp_path, monkeypatch):
    def fake_bounded(argv, **_kwargs):
        if argv[1] == "run":
            return ProcResult(
                126,
                "",
                "target chose exit 126",
                0.1,
                cleanup_confirmed=True,
                cleanup_detail="docker client process group stopped",
            )
        assert argv[1:3] == ["rm", "-f"]
        return ProcResult(
            0,
            "removed",
            "",
            0.01,
            cleanup_confirmed=True,
        )

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)

    result = DockerBackend(image="img:1").run(
        ["python", "-V"],
        cwd=tmp_path,
    )

    assert result.returncode == 126
    assert result.cleanup_confirmed is True
    assert result.cleanup_unconfirmed is False
    assert result.stderr == "target chose exit 126"


def test_docker_run_durably_binds_full_container_id_before_cleanup(
    tmp_path,
    monkeypatch,
):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n")
    fake_docker.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    container_id = "a" * 64
    calls: list[list[str]] = []

    class Process:
        @staticmethod
        def poll():
            return None

    def fake_bounded(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "run":
            cidfile = Path(argv[argv.index("--cidfile") + 1])
            cidfile.write_text(f"{container_id}\n")
            kwargs["on_started"](Process())
            return ProcResult(0, "ok", "", 0.1)
        assert argv[1:3] == ["container", "ls"]
        assert argv[-1] == f"id={container_id}"
        return ProcResult(0, "", "", 0.01)

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)
    backend = DockerBackend(
        image="img:1",
        docker=str(fake_docker),
        operation_lease_dir=run_dir,
    )

    result = backend.run(["python", "-V"], cwd=tmp_path)

    assert result.returncode == 0
    assert OperationLeaseStore(run_dir).list() == []
    assert not list((run_dir / "active-container-ids").iterdir())
    assert [call[1:3] for call in calls] == [
        ["run", "--rm"],
        ["container", "ls"],
    ]


def test_docker_run_waits_for_segmented_cidfile_write(
    tmp_path,
    monkeypatch,
):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n")
    fake_docker.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    container_id = "b" * 64
    calls: list[list[str]] = []

    class Process:
        cidfile: Path
        completed = False

        @classmethod
        def poll(cls):
            if not cls.completed:
                with cls.cidfile.open("a") as stream:
                    stream.write(container_id[16:] + "\n")
                cls.completed = True
            return None

    def fake_bounded(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "run":
            Process.cidfile = Path(argv[argv.index("--cidfile") + 1])
            Process.cidfile.write_text(container_id[:16])
            kwargs["on_started"](Process())
            return ProcResult(0, "ok", "", 0.1)
        assert argv[1:3] == ["container", "ls"]
        assert argv[-1] == f"id={container_id}"
        return ProcResult(0, "", "", 0.01)

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)
    backend = DockerBackend(
        image="img:1",
        docker=str(fake_docker),
        operation_lease_dir=run_dir,
    )

    result = backend.run(["python", "-V"], cwd=tmp_path)

    assert result.returncode == 0
    assert Process.completed is True
    assert OperationLeaseStore(run_dir).list() == []
    assert not list((run_dir / "active-container-ids").iterdir())


@pytest.mark.parametrize(
    "payload",
    [
        "g" * 64,
        "a" * 64 + "x",
        "a" * 66,
    ],
)
def test_docker_cidfile_rejects_complete_invalid_identity(tmp_path, payload):
    cidfile = tmp_path / "container.cid"
    cidfile.write_text(payload)

    with pytest.raises(
        docker_backend.DockerContainerIdentityError,
        match="invalid container ID|bounded standalone file",
    ):
        docker_backend._read_container_id(cidfile)


def test_docker_cidfile_rejects_hard_link(tmp_path):
    backing = tmp_path / "backing.cid"
    cidfile = tmp_path / "container.cid"
    backing.write_text("a" * 64)
    os.link(backing, cidfile)

    with pytest.raises(
        docker_backend.DockerContainerIdentityError,
        match="bounded standalone file",
    ):
        docker_backend._read_container_id(cidfile)


def test_docker_recovery_removes_the_inspected_container_id_not_its_old_name(
    tmp_path,
    monkeypatch,
):
    store = OperationLeaseStore(tmp_path)
    lease = store.activate_docker(["python", "-V"], cwd=tmp_path)
    container_id = "b" * 64
    lease = store.bind_container_id(lease.operation_id, container_id)
    calls: list[list[str]] = []

    def fake_bounded(argv, **_kwargs):
        calls.append(argv)
        if argv[1:3] == ["container", "ls"]:
            assert argv[-1] == f"id={container_id}"
            return ProcResult(0, f"{container_id}\n", "", 0.01)
        if argv[1:3] == ["container", "inspect"]:
            assert argv[-1] == container_id
            return ProcResult(
                0,
                json.dumps(
                    {
                        "Id": container_id,
                        "Name": "/renamed-after-start",
                        "Config": {"Labels": {"lha.operation_id": lease.container_identity}},
                    }
                ),
                "",
                0.01,
            )
        assert argv[1:3] == ["rm", "-f"]
        assert argv[-1] == container_id
        return ProcResult(0, container_id, "", 0.01)

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)

    cleanup = docker_backend._recover_docker_operation(
        "/fixed/docker",
        4096,
        store,
        lease,
    )

    assert cleanup.confirmed is True
    assert store.list() == []
    assert [call[1:3] for call in calls] == [
        ["container", "ls"],
        ["container", "inspect"],
        ["rm", "-f"],
    ]


def test_docker_timeout_keeps_lease_when_client_group_is_not_confirmed_stopped(
    tmp_path,
    monkeypatch,
):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n")
    fake_docker.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: list[list[str]] = []

    def fake_bounded(argv, **kwargs):
        calls.append(argv)
        assert argv[1] == "run"
        cleanup = kwargs["on_timeout"](object())
        return ProcResult(
            126,
            "",
            cleanup.detail,
            0.1,
            cleanup_confirmed=False,
            cleanup_detail=cleanup.detail,
        )

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)
    monkeypatch.setattr(
        docker_backend,
        "terminate_process_group",
        lambda _process: ProcessCleanupResult(
            False,
            "Docker client process group still exists",
        ),
    )
    backend = DockerBackend(
        image="img:1",
        docker=str(fake_docker),
        operation_lease_dir=run_dir,
    )

    result = backend.run(["python", "-V"], cwd=tmp_path, timeout=0.1)

    assert result.returncode == 126
    assert result.cleanup_unconfirmed is True
    assert len(OperationLeaseStore(run_dir).list()) == 1
    assert [call[1] for call in calls] == ["run"]


def test_docker_popen_failure_clears_prepared_lease_without_daemon_probe(
    tmp_path,
    monkeypatch,
):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n")
    fake_docker.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: list[list[str]] = []

    def fail_to_spawn(argv, **_kwargs):
        calls.append(argv)
        raise OSError("simulated Popen failure")

    monkeypatch.setattr(docker_backend, "run_bounded_process", fail_to_spawn)
    backend = DockerBackend(
        image="img:1",
        docker=str(fake_docker),
        operation_lease_dir=run_dir,
    )

    result = backend.run(["python", "-V"], cwd=tmp_path)

    assert result.returncode == 127
    assert "simulated Popen failure" in result.stderr
    assert OperationLeaseStore(run_dir).list() == []
    assert [call[1] for call in calls] == ["run"]


def test_docker_interruption_clears_lease_only_after_client_and_container_cleanup(
    tmp_path,
    monkeypatch,
):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n")
    fake_docker.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    container_id = "c" * 64
    order: list[str] = []

    class Process:
        @staticmethod
        def poll():
            return None

    def interrupted(argv, **kwargs):
        if argv[1] == "run":
            cidfile = Path(argv[argv.index("--cidfile") + 1])
            cidfile.write_text(f"{container_id}\n")
            kwargs["on_started"](Process())
            cleanup = kwargs["on_timeout"](Process())
            assert cleanup.confirmed is True
            raise KeyboardInterrupt("simulated interruption")
        assert argv[1:3] == ["container", "ls"]
        order.append("container")
        return ProcResult(0, "", "", 0.01)

    monkeypatch.setattr(docker_backend, "run_bounded_process", interrupted)

    def stop_client(_process):
        order.append("client")
        return ProcessCleanupResult(True, "Docker client stopped")

    monkeypatch.setattr(
        docker_backend,
        "terminate_process_group",
        stop_client,
    )
    backend = DockerBackend(
        image="img:1",
        docker=str(fake_docker),
        operation_lease_dir=run_dir,
    )

    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        backend.run(["python", "-V"], cwd=tmp_path)

    assert order == ["client", "container"]
    assert OperationLeaseStore(run_dir).list() == []


def test_docker_client_output_is_bounded_without_a_daemon(tmp_path):
    fake_docker = tmp_path / "fake-docker"
    fake_docker.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if len(sys.argv) > 1 and sys.argv[1] == 'run':\n"
        "    os.write(1, b'o' * 200_000)\n"
        "    os.write(2, b'e' * 200_000)\n"
        "elif len(sys.argv) > 2 and sys.argv[1:3] == ['container', 'ls']:\n"
        "    raise SystemExit(0)\n"
        "else:\n"
        "    raise SystemExit(1)\n"
    )
    fake_docker.chmod(0o755)
    limit = 4096

    result = DockerBackend(
        image="unused",
        docker=str(fake_docker),
        limits=ResourceLimits(output_bytes=limit),
    ).run(["python", "-V"], cwd=tmp_path, timeout=10)

    assert result.returncode == 125
    assert result.output_truncated is True
    assert len(result.stdout.encode()) <= limit
    assert len(result.stderr.encode()) <= limit


def test_docker_provenance_records_image_id_and_in_container_versions(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_bounded(argv, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            return ProcResult(0, f"sha256:{'a' * 64}\n", "", 0.01)
        return ProcResult(
            0,
            json.dumps({"python": "3.12.9", "pytest": "8.4.1", "ruff": "0.12.4"}),
            "",
            0.02,
        )

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)

    provenance = DockerBackend(image="img:1").provenance()

    assert provenance["image_id"] == f"sha256:{'a' * 64}"
    assert provenance["image_id_status"] == "available"
    assert provenance["versions"] == {
        "python": "3.12.9",
        "pytest": "8.4.1",
        "ruff": "0.12.4",
    }
    assert provenance["versions_status"] == "available"
    probe = calls[1]
    assert "--network" in probe and probe[probe.index("--network") + 1] == "none"
    assert "--read-only" in probe
    assert "--entrypoint" in probe
    assert not any(":/work" in value for value in probe)
    assert f"sha256:{'a' * 64}" in probe
    assert provenance["versions_bound_to_image_id"] is True


def test_docker_provenance_failure_is_explicit_and_non_raising(
    monkeypatch,
) -> None:
    calls = 0

    def fake_bounded(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("docker unavailable")
        return ProcResult(127, "", "python missing", 0.01)

    monkeypatch.setattr(docker_backend, "run_bounded_process", fake_bounded)

    provenance = DockerBackend(image="missing:1").provenance()

    assert provenance["image_id"] is None
    assert provenance["image_id_status"] == "unavailable"
    assert provenance["image_id_reason"] == "inspect_could_not_start"
    assert provenance["versions_status"] == "unavailable"
    assert provenance["versions_reason"] == "version_probe_cleanup_unconfirmed"
    assert provenance["versions_bound_to_image_id"] is False


@pytest.mark.skipif(
    os.environ.get("LHA_DOCKER_TESTS") != "1",
    reason="docker integration is opt-in (LHA_DOCKER_TESTS=1)",
)
def test_docker_backend_runs_python(tmp_path):
    assert DockerBackend.available(), "LHA_DOCKER_TESTS=1 requires a working Docker daemon"
    (tmp_path / "hello.py").write_text("print('from-container')\n")
    image = os.environ.get("LHA_DOCKER_TEST_IMAGE", "python:3.12-slim")
    res = DockerBackend(image=image).run(["python", "hello.py"], cwd=tmp_path, timeout=120)
    assert res.returncode == 0
    assert "from-container" in res.stdout


@pytest.mark.skipif(
    os.environ.get("LHA_DOCKER_TESTS") != "1",
    reason="docker integration is opt-in (LHA_DOCKER_TESTS=1)",
)
def test_docker_backend_has_read_only_root_and_bounded_tmpfs(tmp_path):
    assert DockerBackend.available(), "LHA_DOCKER_TESTS=1 requires a working Docker daemon"
    image = os.environ.get("LHA_DOCKER_TEST_IMAGE", "python:3.12-slim")
    script = """
from pathlib import Path

mounts = {}
for line in Path("/proc/mounts").read_text().splitlines():
    _, target, _, options, *_ = line.split()
    mounts[target] = set(options.split(","))

assert "ro" in mounts["/"], mounts["/"]
assert {"rw", "nosuid", "nodev"} <= mounts["/tmp"], mounts["/tmp"]
assert any(option.startswith("size=") for option in mounts["/tmp"])
Path("/tmp/lha-tmpfs-probe").write_text("ok")
print("read-only-root tmpfs-ok")
"""
    res = DockerBackend(image=image).run(
        ["python", "-c", script],
        cwd=tmp_path,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr or res.stdout
    assert "read-only-root tmpfs-ok" in res.stdout


@pytest.mark.skipif(
    os.environ.get("LHA_DOCKER_TESTS") != "1",
    reason="docker integration is opt-in (LHA_DOCKER_TESTS=1)",
)
def test_docker_release_image_runs_pytest_and_git_apply(tmp_path):
    assert DockerBackend.available(), "LHA_DOCKER_TESTS=1 requires a working Docker daemon"
    image = os.environ.get("LHA_DOCKER_TEST_IMAGE")
    if not image:
        pytest.fail("LHA_DOCKER_TEST_IMAGE must name the release candidate image")
    (tmp_path / "value.py").write_text("VALUE = 1\n")
    (tmp_path / "test_value.py").write_text(
        "from value import VALUE\n\ndef test_value():\n    assert VALUE == 1\n"
    )
    backend = DockerBackend(image=image)

    tests = backend.run(["python", "-m", "pytest", "-q"], cwd=tmp_path, timeout=120)
    assert tests.returncode == 0, tests.stderr or tests.stdout
    diff = "--- a/value.py\n+++ b/value.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    applied = backend.run(
        ["git", "apply", "--whitespace=nowarn", "-p1", "-"],
        cwd=tmp_path,
        timeout=120,
        input=diff,
    )
    assert applied.returncode == 0, applied.stderr or applied.stdout
    assert (tmp_path / "value.py").read_text() == "VALUE = 2\n"
