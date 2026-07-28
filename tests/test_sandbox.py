"""Execution backends: env scrubbing, process-tree kill, limits, docker argv.

The docker integration test is opt-in (LHA_DOCKER_TESTS=1 and a working
daemon); everything else runs hermetically on the host.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

import lha.sandbox.base as sandbox_base
import lha.sandbox.docker as docker_backend
import lha.sandbox.local as local_backend
from lha.sandbox import DockerBackend, ResourceLimits, TrustedLocalBackend, scrub_env
from lha.sandbox.base import ProcessCleanupResult
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
            (
                "import os\n"
                "os.write(1, b'o' * 200_000)\n"
                "os.write(2, b'e' * 200_000)\n"
            ),
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
def test_local_backend_stops_descendant_that_closed_output_pipe(
    tmp_path, exit_code
):
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


def test_local_backend_fails_when_group_cleanup_cannot_be_confirmed(
    tmp_path, monkeypatch
):
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


def test_local_backend_rejects_unsupported_process_groups_before_spawn(
    tmp_path, monkeypatch
):
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


def test_local_backend_uses_exec_launcher_instead_of_preexec(
    tmp_path, monkeypatch
):
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
def test_bounded_process_closes_pipes_when_thread_start_fails(
    failed_start, tmp_path, monkeypatch
):
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


def test_interruption_records_unconfirmed_process_cleanup(
    tmp_path, monkeypatch
):
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

    with pytest.raises(KeyboardInterrupt) as caught:
        sandbox_base.run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            timeout=10,
            output_bytes=1024,
            input="payload",
            start_new_session=True,
            on_exit=unconfirmed_cleanup,
        )

    assert any(
        "simulated cleanup uncertainty" in note
        for note in getattr(caught.value, "__notes__", ())
    )


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

    assert result.returncode == 124
    assert result.stdout == "partial output"
    assert "docker run stalled" in result.stderr
    assert "cleanup failed with exit code 1" in result.stderr
    assert "daemon refused to remove" in result.stderr
    assert "container removed" not in result.stderr
    assert invocations[1][1:3] == ["rm", "-f"]


def test_docker_timeout_returns_a_result_when_cleanup_also_times_out(
    tmp_path, monkeypatch
):
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

    assert result.returncode == 124
    assert "cleanup timed out after 30s" in result.stderr
    assert "daemon did not answer" in result.stderr
    assert "may still be running" in result.stderr
    assert "container removed" not in result.stderr


def test_docker_client_output_is_bounded_without_a_daemon(tmp_path):
    fake_docker = tmp_path / "fake-docker"
    fake_docker.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "os.write(1, b'o' * 200_000)\n"
        "os.write(2, b'e' * 200_000)\n"
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
            json.dumps(
                {"python": "3.12.9", "pytest": "8.4.1", "ruff": "0.12.4"}
            ),
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
    assert provenance["versions_reason"] == "version_probe_failed"
    assert provenance["versions_bound_to_image_id"] is False


@pytest.mark.skipif(
    os.environ.get("LHA_DOCKER_TESTS") != "1",
    reason="docker integration is opt-in (LHA_DOCKER_TESTS=1)",
)
def test_docker_backend_runs_python(tmp_path):
    assert DockerBackend.available(), (
        "LHA_DOCKER_TESTS=1 requires a working Docker daemon"
    )
    (tmp_path / "hello.py").write_text("print('from-container')\n")
    image = os.environ.get("LHA_DOCKER_TEST_IMAGE", "python:3.12-slim")
    res = DockerBackend(image=image).run(
        ["python", "hello.py"], cwd=tmp_path, timeout=120
    )
    assert res.returncode == 0
    assert "from-container" in res.stdout


@pytest.mark.skipif(
    os.environ.get("LHA_DOCKER_TESTS") != "1",
    reason="docker integration is opt-in (LHA_DOCKER_TESTS=1)",
)
def test_docker_backend_has_read_only_root_and_bounded_tmpfs(tmp_path):
    assert DockerBackend.available(), (
        "LHA_DOCKER_TESTS=1 requires a working Docker daemon"
    )
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
    assert DockerBackend.available(), (
        "LHA_DOCKER_TESTS=1 requires a working Docker daemon"
    )
    image = os.environ.get("LHA_DOCKER_TEST_IMAGE")
    if not image:
        pytest.fail("LHA_DOCKER_TEST_IMAGE must name the release candidate image")
    (tmp_path / "value.py").write_text("VALUE = 1\n")
    (tmp_path / "test_value.py").write_text(
        "from value import VALUE\n\n"
        "def test_value():\n"
        "    assert VALUE == 1\n"
    )
    backend = DockerBackend(image=image)

    tests = backend.run(["python", "-m", "pytest", "-q"], cwd=tmp_path, timeout=120)
    assert tests.returncode == 0, tests.stderr or tests.stdout
    diff = (
        "--- a/value.py\n"
        "+++ b/value.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    applied = backend.run(
        ["git", "apply", "--whitespace=nowarn", "-p1", "-"],
        cwd=tmp_path,
        timeout=120,
        input=diff,
    )
    assert applied.returncode == 0, applied.stderr or applied.stdout
    assert (tmp_path / "value.py").read_text() == "VALUE = 2\n"
