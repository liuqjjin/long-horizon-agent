from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pytest

from lha.bench.terminal_control import (
    ControlRecordExists,
    ControlStoreError,
    RegisteredAttempt,
    SecureDirectory,
    command_digest,
    evaluation_lock,
    initialize_control_store,
    open_attempt_store,
    terminal_attempt_id,
    terminal_control_root,
    write_model_started,
)

EVALUATION_ID = "a" * 32
PROTOCOL_SHA256 = "b" * 64


def _attempts() -> tuple[RegisteredAttempt, ...]:
    values = []
    cases: tuple[tuple[Literal["smoke", "scored"], int], ...] = (
        ("smoke", 3),
        ("scored", 20),
    )
    for run_kind, count in cases:
        for index in range(count):
            instance_id = f"terminal-bench/{run_kind}-{index:02d}"
            values.append(
                RegisteredAttempt(
                    attempt_id=terminal_attempt_id(
                        EVALUATION_ID,
                        run_kind,
                        instance_id,
                    ),
                    run_kind=run_kind,
                    instance_id=instance_id,
                    command_sha256=command_digest(("harbor", "run", instance_id)),
                )
            )
    return tuple(values)


def _initialize(tmp_path: Path):
    output = (tmp_path / "jobs").resolve()
    registration = initialize_control_store(
        evaluation_id=EVALUATION_ID,
        protocol_sha256=PROTOCOL_SHA256,
        output_root=output,
        attempts=_attempts(),
    )
    return output, registration


def test_control_root_is_private_sibling_and_attempts_are_fixed(tmp_path):
    output, registration = _initialize(tmp_path)
    control = terminal_control_root(output, EVALUATION_ID)

    assert output.is_dir()
    assert control.parent == output.parent / ".lha-control"
    assert control not in output.parents and output not in control.parents
    assert control.stat().st_mode & 0o777 == 0o700
    assert len(registration.attempts) == 23
    assert all((control / item.attempt_id).is_dir() for item in registration.attempts)
    assert all(
        (control / item.attempt_id).stat().st_mode & 0o777 == 0o700
        for item in registration.attempts
    )

    with pytest.raises(ControlRecordExists, match="output root already exists"):
        initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )


def test_immutable_marker_prevents_a_second_model_start(tmp_path):
    output, registration = _initialize(tmp_path)
    attempt = registration.attempts[0]
    with open_attempt_store(output, EVALUATION_ID, attempt.attempt_id) as store:
        marker = write_model_started(
            store,
            evaluation_id=EVALUATION_ID,
            attempt_id=attempt.attempt_id,
            protocol_sha256=PROTOCOL_SHA256,
            run_kind=attempt.run_kind,
            instance_id=attempt.instance_id,
            container_id="c" * 64,
        )
        assert marker.instance_id == attempt.instance_id
        assert store.has("MODEL_STARTED.json")
        with pytest.raises(ControlRecordExists, match="already exists"):
            write_model_started(
                store,
                evaluation_id=EVALUATION_ID,
                attempt_id=attempt.attempt_id,
                protocol_sha256=PROTOCOL_SHA256,
                run_kind=attempt.run_kind,
                instance_id=attempt.instance_id,
                container_id="c" * 64,
            )


def test_symlink_and_hardlink_records_never_touch_sentinel(tmp_path):
    store_path = tmp_path / "control"
    store_path.mkdir(mode=0o700)
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"unchanged")
    sentinel.chmod(0o600)

    with SecureDirectory(store_path) as store:
        (store_path / "events.jsonl").symlink_to(sentinel)
        with pytest.raises(ControlRecordExists):
            store.write_once("events.jsonl", b"forged")
        assert sentinel.read_bytes() == b"unchanged"

        os.link(sentinel, store_path / "hardlink.json")
        with pytest.raises(ControlStoreError, match="private regular file"):
            store.read("hardlink.json")
        assert sentinel.read_bytes() == b"unchanged"


def test_open_directory_fd_survives_parent_path_replacement(tmp_path):
    original = tmp_path / "control"
    original.mkdir(mode=0o700)
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    sentinel = replacement / "record.json"
    sentinel.write_bytes(b"replacement")
    sentinel.chmod(0o600)

    store = SecureDirectory(original)
    renamed = tmp_path / "control-original"
    original.rename(renamed)
    original.symlink_to(replacement, target_is_directory=True)
    try:
        store.write_once("record.json", b"trusted")
    finally:
        store.close()

    assert sentinel.read_bytes() == b"replacement"
    assert (renamed / "record.json").read_bytes() == b"trusted"


def test_unsafe_control_directory_is_rejected(tmp_path):
    directory = tmp_path / "control"
    directory.mkdir(mode=0o755)
    with pytest.raises(ControlStoreError, match="private and owner-only"):
        SecureDirectory(directory)


def test_evaluation_lock_rejects_concurrent_controller(tmp_path):
    output, _registration = _initialize(tmp_path)
    with evaluation_lock(output, EVALUATION_ID) as descriptor:
        assert descriptor >= 0
        with pytest.raises(ControlStoreError, match="already active"):
            with evaluation_lock(output, EVALUATION_ID):
                pass
    with evaluation_lock(output, EVALUATION_ID) as descriptor:
        assert descriptor >= 0
