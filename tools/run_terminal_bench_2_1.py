#!/usr/bin/env python3
"""Run or validate the preregistered Terminal-Bench 2.1 fixed-20 evaluation.

The runner is restartable: rerunning ``run`` with the same arguments resumes
only attempts that have no durable command envelope. It never creates a second
scored attempt for a registered task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path

from lha.bench.terminal_bench import (
    HarborExecutionManifest,
    TerminalBenchProtocol,
    build_harbor_commands,
    create_protocol,
    derive_terminal_bench_records,
    initialize_terminal_evaluation,
    run_terminal_phase,
    seal_smoke_phase,
    summarize_records,
    validate_harbor_results,
    write_protocol,
)
from lha.bench.terminal_control import (
    SmokeSeal,
    open_control_store,
    terminal_control_root,
)
from lha.bench.terminal_public_evidence import (
    export_terminal_bench_public_evidence,
    validate_terminal_bench_public_evidence,
)

_SMOKE_MANIFEST = "smoke_manifest.json"
_SMOKE_SEAL = "smoke_seal.json"
_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_PROTOCOL_BYTES = 4 * 1024 * 1024


def _file_fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _open_regular_file(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    """Open one link-free inode and prove lstat/open referred to the same file."""
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a regular file with one link: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely: {path}") from exc
    after = os.fstat(descriptor)
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
    ):
        os.close(descriptor)
        raise ValueError(f"{label} changed while it was opened: {path}")
    return descriptor, after


def _digest_descriptor(
    descriptor: int,
    expected: os.stat_result,
    *,
    label: str,
) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    if (
        size != expected.st_size
        or _file_fingerprint(after) != _file_fingerprint(expected)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
    ):
        raise ValueError(f"{label} changed while it was read")
    return digest.hexdigest()


def _read_regular_bytes(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    descriptor, before = _open_regular_file(path, label=label)
    try:
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds its size limit")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, _COPY_CHUNK_BYTES))
            if not chunk:
                raise ValueError(f"{label} ended before its stated size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew while it was read")
        if _file_fingerprint(os.fstat(descriptor)) != _file_fingerprint(before):
            raise ValueError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_staging_directory(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"staging directory is unavailable: {path}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise ValueError(f"staging directory is not owner-controlled: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"staging directory cannot be opened safely: {path}") from exc
    after = os.fstat(descriptor)
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or not stat.S_ISDIR(after.st_mode)
        or after.st_uid != os.getuid()
        or stat.S_IMODE(after.st_mode) & 0o022
    ):
        os.close(descriptor)
        raise ValueError(f"staging directory changed while it was opened: {path}")
    return descriptor


def _validate_staged_file(
    target: Path,
    source_digest: str,
    *,
    mode: int,
) -> Path:
    descriptor, before = _open_regular_file(target, label="staged input")
    try:
        if before.st_uid != os.getuid():
            raise ValueError(f"staged input owner changed: {target}")
        if stat.S_IMODE(before.st_mode) != mode:
            raise ValueError(f"staged input mode changed: {target}")
        if _digest_descriptor(descriptor, before, label="staged input") != source_digest:
            raise ValueError(f"staged input changed: {target}")
    finally:
        os.close(descriptor)
    return target


def _stage_once(source: Path, target: Path, *, mode: int) -> Path:
    """Copy a public input once, or prove an existing staged copy is identical."""
    source = source.absolute()
    target = target.absolute()
    if source != source.resolve(strict=True):
        raise ValueError("input path must not contain symlink components")
    source_descriptor, source_info = _open_regular_file(source, label="input")
    try:
        source_digest = _digest_descriptor(
            source_descriptor,
            source_info,
            label="input",
        )
        try:
            os.lstat(target)
        except FileNotFoundError:
            pass
        else:
            return _validate_staged_file(target, source_digest, mode=mode)

        directory_fd = _open_staging_directory(target.parent)
        try:
            temporary_name = f".{target.name}.{secrets.token_hex(16)}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            flags |= getattr(os, "O_NOFOLLOW", 0)
            temporary_descriptor = os.open(
                temporary_name,
                flags,
                mode,
                dir_fd=directory_fd,
            )
            linked = False
            try:
                os.fchmod(temporary_descriptor, mode)
                os.lseek(source_descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = os.read(source_descriptor, _COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        written = os.write(temporary_descriptor, view)
                        if written <= 0:
                            raise OSError("staged input write made no progress")
                        view = view[written:]
                    digest.update(chunk)
                    size += len(chunk)
                if (
                    size != source_info.st_size
                    or digest.hexdigest() != source_digest
                    or _file_fingerprint(os.fstat(source_descriptor))
                    != _file_fingerprint(source_info)
                ):
                    raise ValueError("input changed while it was staged")
                os.fsync(temporary_descriptor)
                os.link(
                    temporary_name,
                    target.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                linked = True
            except FileExistsError:
                # A concurrent creator cannot be overwritten. Its file must
                # independently prove the same bytes and mode below.
                pass
            finally:
                os.close(temporary_descriptor)
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            os.fsync(directory_fd)
            if not linked:
                return _validate_staged_file(target, source_digest, mode=mode)
        finally:
            os.close(directory_fd)
    finally:
        os.close(source_descriptor)
    return _validate_staged_file(target, source_digest, mode=mode)


def _load_or_create_protocol(args: argparse.Namespace) -> tuple[
    TerminalBenchProtocol,
    Path,
    Path,
    Path,
    bool,
]:
    work_dir = Path(args.work_dir).resolve()
    inputs = work_dir / "inputs"
    wheel = _stage_once(
        Path(args.wheel),
        inputs / Path(args.wheel).name,
        mode=0o644,
    )
    codex_binary = _stage_once(
        Path(args.codex_binary),
        inputs / "codex-x86_64-unknown-linux-musl",
        mode=0o755,
    )
    protocol_path = inputs / "protocol.json"
    expected = create_protocol(
        evaluation_id=args.evaluation_id,
        output_root=work_dir / "jobs",
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        codex_cli_version=args.codex_cli_version,
        codex_target="x86_64-unknown-linux-musl",
        codex_binary_path=codex_binary,
        broker_image_id=args.broker_image_id,
        wheel_path=wheel,
    )
    try:
        os.lstat(protocol_path)
    except FileNotFoundError:
        created = True
    else:
        created = False
    if created:
        write_protocol(expected, protocol_path)
    try:
        protocol_bytes = _read_regular_bytes(
            protocol_path,
            label="protocol",
            maximum_bytes=_MAX_PROTOCOL_BYTES,
        )
        recorded = TerminalBenchProtocol.model_validate_json(protocol_bytes)
    except (OSError, ValueError) as exc:
        raise ValueError("existing protocol is unreadable") from exc
    if recorded != expected:
        raise ValueError("existing protocol does not match the supplied run arguments")
    return recorded, protocol_path, wheel, codex_binary, created


def _load_sealed_smoke(
    protocol: TerminalBenchProtocol,
) -> tuple[SmokeSeal, HarborExecutionManifest] | None:
    with open_control_store(protocol.output_root, protocol.evaluation_id) as store:
        if not store.has(_SMOKE_SEAL):
            return None
        if not store.has(_SMOKE_MANIFEST):
            raise ValueError("smoke seal exists without its manifest")
        return (
            SmokeSeal.model_validate_json(store.read(_SMOKE_SEAL)),
            HarborExecutionManifest.model_validate_json(store.read(_SMOKE_MANIFEST)),
        )


def _verified_owner_directory(path: Path, *, label: str) -> Path:
    """Return one normalized directory that other host users cannot replace."""
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise ValueError(f"{label} is not an owner-controlled directory: {path}")
    normalized = path.resolve(strict=True)
    if path != normalized:
        raise ValueError(f"{label} must not contain symlink components")
    return normalized


def _paths_overlap(left: Path, right: Path) -> bool:
    return (
        left == right
        or left.is_relative_to(right)
        or right.is_relative_to(left)
    )


def _preflight_neutral_paths(
    args: argparse.Namespace,
    *,
    auth_path: Path,
) -> tuple[Path, Path]:
    public_root = _verified_owner_directory(
        Path(args.public_path_root),
        label="public path root",
    )
    if (
        public_root == Path(public_root.anchor)
        or (
            len(public_root.parts) >= 2
            and public_root.parts[1].casefold() in {"home", "users"}
        )
    ):
        raise ValueError("public path root is not neutral")

    private_roots = (
        Path.home().resolve(),
        Path(__file__).resolve().parents[1],
        auth_path.parent,
    )
    if any(_paths_overlap(public_root, private_root) for private_root in private_roots):
        raise ValueError("public path root overlaps a private path root")

    supplied_work = Path(args.work_dir)
    if not supplied_work.is_absolute():
        raise ValueError("work directory must be absolute")
    normalized_work = supplied_work.resolve(strict=False)
    if supplied_work != normalized_work:
        raise ValueError("work directory must not contain symlink components")
    if not (
        normalized_work == public_root
        or normalized_work.is_relative_to(public_root)
    ):
        raise ValueError("work directory must be inside --public-path-root")
    supplied_work.mkdir(parents=True, exist_ok=True)
    work_dir = _verified_owner_directory(supplied_work, label="work directory")
    return work_dir, public_root


def _run(args: argparse.Namespace) -> int:
    public_out = Path(args.public_out)
    try:
        os.lstat(public_out)
    except FileNotFoundError:
        pass
    else:
        raise ValueError(
            "--public-out already exists; use the validate subcommand to inspect "
            "an existing evidence package"
        )

    auth_path = Path(args.auth).resolve(strict=True)
    work_dir, public_root = _preflight_neutral_paths(args, auth_path=auth_path)

    protocol, protocol_path, wheel, codex_binary, _created = _load_or_create_protocol(args)
    smoke_commands = build_harbor_commands(
        protocol,
        "smoke",
        protocol_path=protocol_path,
        wheel_path=wheel,
        codex_binary_path=codex_binary,
    )
    scored_commands = build_harbor_commands(
        protocol,
        "scored",
        protocol_path=protocol_path,
        wheel_path=wheel,
        codex_binary_path=codex_binary,
    )
    # Initialization is idempotent only for an exact, command-free partial
    # setup. Calling it on every resume closes gaps between the registration
    # record and creation of all 23 attempt directories.
    initialize_terminal_evaluation(
        protocol,
        (*smoke_commands, *scored_commands),
        protocol_path=protocol_path,
    )

    public_out = public_out.resolve()

    sealed = _load_sealed_smoke(protocol)
    if sealed is None:
        run_terminal_phase(
            protocol,
            "smoke",
            smoke_commands,
            protocol_path=protocol_path,
            auth_path=auth_path,
        )
        smoke_seal, smoke_manifest = seal_smoke_phase(
            protocol,
            smoke_commands,
            protocol_path=protocol_path,
        )
    else:
        smoke_seal, smoke_manifest = sealed

    run_terminal_phase(
        protocol,
        "scored",
        scored_commands,
        protocol_path=protocol_path,
        auth_path=auth_path,
    )
    scored_manifest_path = protocol_path.parent / "scored_manifest.json"
    scored_manifest = validate_harbor_results(
        protocol,
        "scored",
        scored_commands,
        protocol_path=protocol_path,
        manifest_path=scored_manifest_path,
    )
    records = derive_terminal_bench_records(
        protocol,
        scored_commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    summary = summarize_records(
        protocol,
        records,
        commands=scored_commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    control_root = terminal_control_root(protocol.output_root, protocol.evaluation_id)
    validation = export_terminal_bench_public_evidence(
        protocol,
        protocol_path=protocol_path,
        smoke_manifest=smoke_manifest,
        smoke_manifest_path=control_root / _SMOKE_MANIFEST,
        smoke_seal=smoke_seal,
        smoke_seal_path=control_root / _SMOKE_SEAL,
        scored_manifest=scored_manifest,
        scored_manifest_path=scored_manifest_path,
        records=records,
        summary=summary,
        scored_commands=scored_commands,
        output_dir=public_out,
        public_path_root=public_root,
        auth_parent=auth_path.parent,
    )
    print(summary.to_markdown())
    print(json.dumps(validation.model_dump(mode="json"), sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    result = validate_terminal_bench_public_evidence(args.package)
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or validate the Terminal-Bench 2.1 fixed-20 protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="start or safely resume one preregistered fixed-20 evaluation",
    )
    run.add_argument("--evaluation-id", required=True, help="32 lowercase hexadecimal chars")
    run.add_argument("--work-dir", required=True, help="neutral directory outside the checkout")
    run.add_argument("--public-path-root", required=True, help="allowed neutral path prefix")
    run.add_argument("--public-out", required=True, help="new public evidence directory")
    run.add_argument("--model", required=True)
    run.add_argument("--reasoning-effort", required=True)
    run.add_argument("--codex-cli-version", required=True)
    run.add_argument("--codex-binary", required=True)
    run.add_argument("--broker-image-id", required=True, help="immutable sha256 Docker image ID")
    run.add_argument("--wheel", required=True)
    run.add_argument("--auth", required=True, help="Codex auth.json; never copied into evidence")
    run.set_defaults(func=_run)

    validate = subparsers.add_parser("validate", help="recompute a public evidence package")
    validate.add_argument("package")
    validate.set_defaults(func=_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
