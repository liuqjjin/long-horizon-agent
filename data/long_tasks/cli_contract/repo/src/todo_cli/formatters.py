"""Stable machine and human output."""

from __future__ import annotations

import json
from typing import Any


def render_json(items: list[dict[str, Any]]) -> str:
    return "RESULT " + json.dumps({"items": items}, sort_keys=True)


def render_text(items: list[dict[str, Any]]) -> str:
    return "\n".join(f"{item['id']}: {item['title']}" for item in items)

