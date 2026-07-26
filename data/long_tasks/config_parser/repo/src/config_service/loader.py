"""Merge defaults, file values, and environment overrides."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .model import ServiceConfig


def load_config(
    defaults: Mapping[str, Any],
    file_values: Mapping[str, Any],
    environ: Mapping[str, str],
) -> ServiceConfig:
    values = dict(file_values)
    values.update(defaults)
    if "APP_DEBUG" in environ:
        values["debug"] = environ["APP_DEBUG"]
    if "APP_PORTS" in environ:
        values["ports"] = environ["APP_PORTS"].split(",")
    return ServiceConfig.from_mapping(values)

