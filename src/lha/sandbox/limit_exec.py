"""Apply requested POSIX resource limits, then replace this process."""

from __future__ import annotations

import argparse
import os
import sys


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("resource limits must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-s", type=_positive_int)
    parser.add_argument("--memory-mb", type=_positive_int)
    parser.add_argument("--pids", type=_positive_int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        print("resource-limit launcher requires a command", file=sys.stderr)
        return 126

    try:
        import resource

        requested = (
            (args.cpu_s, resource.RLIMIT_CPU, 1),
            (args.memory_mb, resource.RLIMIT_AS, 1024 * 1024),
            (
                args.pids,
                getattr(resource, "RLIMIT_NPROC", None),
                1,
            ),
        )
        for value, resource_name, multiplier in requested:
            if value is None:
                continue
            if resource_name is None:
                raise OSError("requested resource limit is unavailable")
            bound = value * multiplier
            resource.setrlimit(resource_name, (bound, bound))
        os.execvpe(command[0], command, dict(os.environ))
    except (ImportError, OSError, ValueError) as error:
        print(
            f"could not apply requested resource limits: {type(error).__name__}",
            file=sys.stderr,
        )
        return 126
    return 126


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(main())
