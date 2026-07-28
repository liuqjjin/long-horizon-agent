"""Durable ownership of target processes across a harness SIGKILL."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

import lha.operation_lease as operation_lease
import lha.sandbox.docker as docker_backend
from lha.operation_lease import (
    OperationLeaseError,
    OperationLeaseStore,
    recover_active_operations,
    recover_local_operation,
)
from lha.sandbox import DockerBackend, TrustedLocalBackend
from lha.sandbox.base import ProcessCleanupResult
from lha.tools.shell import ProcResult

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "killpg"),
    reason="durable process-group leases require POSIX",
)


def _wait_for_active(store: OperationLeaseStore, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        leases = store.list()
        if len(leases) == 1 and leases[0].phase == "ACTIVE":
            return leases[0]
        time.sleep(0.01)
    raise AssertionError("operation lease did not become active")


def test_kernel_boot_identity_does_not_depend_on_wall_clock(
    monkeypatch,
):
    expected = "kern-boottime:darwin:1710000000:123456"
    monkeypatch.setattr(operation_lease.sys, "platform", "darwin")
    monkeypatch.setattr(operation_lease, "_linux_boot_identity", lambda: None)
    monkeypatch.setattr(
        operation_lease,
        "_kernel_boot_time_identity",
        lambda: expected,
    )

    first = operation_lease._boot_identity()
    monkeypatch.setattr(operation_lease.time, "time", lambda: 9_999_999_999.0)
    monkeypatch.setattr(operation_lease.time, "monotonic", lambda: 1.0)
    second = operation_lease._boot_identity()

    assert first == expected
    assert second == expected


def test_prepare_fails_closed_when_kernel_boot_identity_is_unavailable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(operation_lease, "_boot_identity", lambda: None)
    store = OperationLeaseStore(tmp_path)

    with pytest.raises(OperationLeaseError, match="boot identity is unavailable"):
        store.prepare_local([sys.executable, "-c", "pass"], cwd=tmp_path)

    assert store.list() == []


def test_recovery_retains_lease_when_kernel_boot_identity_is_unavailable(
    tmp_path,
    monkeypatch,
):
    store = OperationLeaseStore(tmp_path)
    lease = store.prepare_local([sys.executable, "-c", "pass"], cwd=tmp_path)
    monkeypatch.setattr(operation_lease, "_boot_identity", lambda: None)

    recovered = recover_local_operation(store, lease)

    assert recovered.confirmed is False
    assert "boot identity is unavailable" in recovered.detail
    assert store.list() == [lease]


def test_directory_fsync_failure_prevents_operation_preparation(
    tmp_path,
    monkeypatch,
):
    store = OperationLeaseStore(tmp_path)

    def fail_fsync(_descriptor):
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(operation_lease.os, "fsync", fail_fsync)

    with pytest.raises(OperationLeaseError, match="synced durably"):
        store.prepare_local([sys.executable, "-c", "pass"], cwd=tmp_path)

    assert list((tmp_path / "active-operations").glob("*.json")) == []


def test_recovery_fails_closed_when_process_birth_identity_is_unknown(
    tmp_path,
    monkeypatch,
):
    store = OperationLeaseStore(tmp_path)
    preparing = store.prepare_local([sys.executable, "-c", "pass"], cwd=tmp_path)
    active = store.activate_local(
        preparing.operation_id,
        pid=os.getpid(),
        pgid=os.getpgrp(),
    )
    monkeypatch.setattr(operation_lease, "_group_exists", lambda _pgid: True)
    monkeypatch.setattr(operation_lease, "_process_snapshot", lambda _pid: None)
    monkeypatch.setattr(operation_lease, "_process_exists", lambda _pid: True)

    def must_not_kill(_pgid, _signal):
        raise AssertionError("unidentified process group must not be killed")

    monkeypatch.setattr(operation_lease.os, "killpg", must_not_kill)

    recovered = recover_local_operation(store, active)

    assert recovered.confirmed is False
    assert "identity is unavailable" in recovered.detail
    assert store.list() == [active]


def test_recovery_fails_closed_when_process_group_identity_changed(
    tmp_path,
    monkeypatch,
):
    store = OperationLeaseStore(tmp_path)
    preparing = store.prepare_local([sys.executable, "-c", "pass"], cwd=tmp_path)
    active = store.activate_local(
        preparing.operation_id,
        pid=os.getpid(),
        pgid=os.getpgrp(),
    )
    assert active.process_identity is not None
    assert active.pgid is not None
    monkeypatch.setattr(operation_lease, "_group_exists", lambda _pgid: True)
    monkeypatch.setattr(
        operation_lease,
        "_process_snapshot",
        lambda _pid: operation_lease._ProcessSnapshot(
            birth_identity=active.process_identity,
            pgid=active.pgid + 1,
        ),
    )

    def must_not_kill(_pgid, _signal):
        raise AssertionError("mismatched process group must not be killed")

    monkeypatch.setattr(operation_lease.os, "killpg", must_not_kill)

    recovered = recover_local_operation(store, active)

    assert recovered.confirmed is False
    assert "now belongs to process group" in recovered.detail
    assert store.list() == [active]


def test_local_backend_clears_lease_only_after_confirmed_cleanup(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    result = TrustedLocalBackend(operation_lease_dir=tmp_path).run(
        [sys.executable, "-c", "print('complete')"],
        cwd=workdir,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.cleanup_confirmed is True
    assert OperationLeaseStore(tmp_path).list() == []


def test_local_backend_fails_closed_when_popen_and_lease_cleanup_fail(
    tmp_path,
    monkeypatch,
):
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    def fail_to_start(*_args, **_kwargs):
        raise OSError("popen failed")

    def fail_to_clear(_self, _operation_id):
        raise OperationLeaseError("directory fsync failed")

    monkeypatch.setattr("lha.sandbox.local.run_bounded_process", fail_to_start)
    monkeypatch.setattr(OperationLeaseStore, "clear", fail_to_clear)

    result = TrustedLocalBackend(operation_lease_dir=tmp_path).run(
        [sys.executable, "-c", "pass"],
        cwd=workdir,
        timeout=10,
    )

    assert result.returncode == 126
    assert result.cleanup_confirmed is False
    assert "lease cleanup could not be confirmed" in result.stderr
    assert "directory fsync failed" in result.cleanup_detail


def test_resume_reaps_sigkill_orphan_before_delayed_write(tmp_path):
    """Kill the harness itself, not its child, then recover from the lease."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    marker = tmp_path / "orphan-write"
    worker = (
        "import sys\n"
        "from lha.sandbox import TrustedLocalBackend\n"
        f"backend = TrustedLocalBackend(operation_lease_dir={str(tmp_path)!r})\n"
        "backend.run(\n"
        "    [sys.executable, '-c', "
        f"\"import time; time.sleep(2); open({str(marker)!r}, 'w').write('bad')\"],\n"
        f"    cwd={str(workdir)!r}, timeout=20,\n"
        ")\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-I", "-c", worker],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    store = OperationLeaseStore(tmp_path)
    lease = _wait_for_active(store)

    os.kill(parent.pid, signal.SIGKILL)
    assert parent.wait(timeout=5) == -signal.SIGKILL

    recovered = recover_active_operations(tmp_path)
    assert recovered.confirmed is True
    assert recovered.recovered_operation_ids == (lease.operation_id,)
    assert recovered.quarantined_operation_ids == ()
    time.sleep(2.5)
    assert not marker.exists(), "orphan wrote after the harness was killed"
    assert store.list() == []


def test_corrupt_lease_returns_typed_quarantine(tmp_path):
    directory = tmp_path / "active-operations"
    directory.mkdir()
    operation_id = "a" * 32
    (directory / f"{operation_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sha256": "0" * 64,
                "payload": {},
            }
        )
    )

    recovered = recover_active_operations(tmp_path)

    assert recovered.confirmed is False
    assert recovered.requires_quarantine is True
    assert "integrity check" in recovered.detail


def test_unresolved_preparing_lease_returns_typed_quarantine(tmp_path):
    store = OperationLeaseStore(tmp_path)
    lease = store.prepare_local(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
    )

    recovered = recover_active_operations(tmp_path)

    assert recovered.confirmed is False
    assert recovered.quarantined_operation_ids == (lease.operation_id,)
    assert "remained PREPARING" in recovered.detail


def test_docker_recovery_validates_label_before_removal(tmp_path, monkeypatch):
    store = OperationLeaseStore(tmp_path)
    lease = store.activate_docker(["python", "-V"], cwd=tmp_path)
    removed: list[str] = []

    monkeypatch.setattr(
        docker_backend,
        "_container_absence_probe",
        lambda *_args, **_kwargs: ProcessCleanupResult(
            False,
            f"container {lease.container_name} still exists",
        ),
    )
    monkeypatch.setattr(
        docker_backend,
        "_docker_control",
        lambda *_args, **_kwargs: ProcResult(
            0,
            json.dumps(
                {
                    "Id": "b" * 64,
                    "Name": f"/{lease.container_name}",
                    "Config": {
                        "Labels": {
                            "lha.operation_id": lease.operation_id,
                        }
                    },
                }
            ),
            "",
            0.01,
        ),
    )

    def remove(_docker, name, **_kwargs):
        removed.append(name)
        return ProcessCleanupResult(True, f"container {name} removed")

    monkeypatch.setattr(docker_backend, "_remove_container", remove)

    recovered = DockerBackend(
        image="image:id",
        operation_lease_dir=tmp_path,
    ).recover_active_operations(tmp_path)

    assert recovered.confirmed is True
    assert removed == [lease.container_name]
    assert store.list() == []


def test_docker_run_persists_identity_before_client_spawn(tmp_path, monkeypatch):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    observed: list[tuple[list[str], object]] = []

    def run_client(argv, **_kwargs):
        leases = OperationLeaseStore(tmp_path).list()
        assert len(leases) == 1
        assert leases[0].phase == "ACTIVE"
        observed.append((argv, leases[0]))
        return ProcResult(
            0,
            "complete",
            "",
            0.01,
            cleanup_confirmed=True,
            cleanup_detail="Docker client exited",
        )

    monkeypatch.setattr(docker_backend, "run_bounded_process", run_client)
    monkeypatch.setattr(
        docker_backend,
        "_container_absence_probe",
        lambda *_args, **_kwargs: ProcessCleanupResult(
            True,
            "container is absent",
        ),
    )

    result = DockerBackend(
        image="image:id",
        operation_lease_dir=tmp_path,
    ).run(["python", "-V"], cwd=workdir)

    assert result.returncode == 0
    assert len(observed) == 1
    argv, lease = observed[0]
    assert ["--name", lease.container_name] == argv[
        argv.index("--name") : argv.index("--name") + 2
    ]
    assert f"lha.operation_id={lease.operation_id}" in argv
    assert OperationLeaseStore(tmp_path).list() == []


def test_docker_recovery_quarantines_identity_mismatch(tmp_path, monkeypatch):
    store = OperationLeaseStore(tmp_path)
    lease = store.activate_docker(["python", "-V"], cwd=tmp_path)
    monkeypatch.setattr(
        docker_backend,
        "_container_absence_probe",
        lambda *_args, **_kwargs: ProcessCleanupResult(
            False,
            f"container {lease.container_name} still exists",
        ),
    )
    monkeypatch.setattr(
        docker_backend,
        "_docker_control",
        lambda *_args, **_kwargs: ProcResult(
            0,
            json.dumps(
                {
                    "Id": "b" * 64,
                    "Name": f"/{lease.container_name}",
                    "Config": {"Labels": {"lha.operation_id": "not-ours"}},
                }
            ),
            "",
            0.01,
        ),
    )

    recovered = DockerBackend(image="image:id").recover_active_operations(
        tmp_path
    )

    assert recovered.confirmed is False
    assert recovered.quarantined_operation_ids == (lease.operation_id,)
    assert store.list() == [lease]
