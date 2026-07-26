"""Todo command line."""

from __future__ import annotations

import argparse

from .formatters import render_json, render_text
from .service import find_item, load_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="todo")
    parser.add_argument("--file", required=True)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--id", type=int)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        items = load_items(args.file)
        if args.id is not None:
            items = [find_item(items, args.id)]
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 0
    print(render_json(items) if args.format == "json" else render_text(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

