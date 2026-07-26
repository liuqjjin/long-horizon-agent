from __future__ import annotations

import pytest

from config_service import load_config


def test_file_values_override_defaults_and_are_typed():
    config = load_config(
        {"debug": False, "ports": [8000]},
        {"debug": True, "ports": [9000, 9001]},
        {},
    )
    assert config.debug is True
    assert config.ports == (9000, 9001)


def test_environment_false_is_not_truthy_and_ports_are_integers():
    config = load_config(
        {"debug": True, "ports": [8000]},
        {},
        {"APP_DEBUG": "false", "APP_PORTS": "7000, 7001"},
    )
    assert config.debug is False
    assert config.ports == (7000, 7001)


def test_invalid_boolean_is_rejected():
    with pytest.raises(ValueError, match="boolean"):
        load_config({}, {}, {"APP_DEBUG": "sometimes"})

