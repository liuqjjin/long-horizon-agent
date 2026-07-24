"""Execution backends: env scrubbing, process-tree kill, limits, docker argv.

The docker integration test is opt-in (LHA_DOCKER_TESTS=1 and a working
daemon); everything else runs hermetically on the host.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from lha.sandbox import DockerBackend, ResourceLimits, TrustedLocalBackend, scrub_env


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
    assert "--memory 512m" in joined
    assert "--pids-limit 64" in joined
    assert f"{tmp_path.resolve()}:/work" in joined
    assert f"{(tmp_path / 'base').resolve()}:/base:ro" in joined
    # host interpreter path translated for the container
    assert sys.executable not in argv
    assert argv[-3:] == ["python", "-m", "pytest"]
    # no environment forwarding beyond the explicit HOME
    assert joined.count("--env") == 1


@pytest.mark.skipif(
    os.environ.get("LHA_DOCKER_TESTS") != "1" or not DockerBackend.available(),
    reason="docker integration is opt-in (LHA_DOCKER_TESTS=1 + running daemon)",
)
def test_docker_backend_runs_python(tmp_path):
    (tmp_path / "hello.py").write_text("print('from-container')\n")
    res = DockerBackend().run(["python", "hello.py"], cwd=tmp_path, timeout=120)
    assert res.returncode == 0
    assert "from-container" in res.stdout
