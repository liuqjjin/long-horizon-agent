"""Typed service configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServiceConfig:
    debug: bool
    ports: tuple[int, ...]

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ServiceConfig":
        return cls(
            debug=bool(values.get("debug", False)),
            ports=tuple(values.get("ports", ())),
        )

