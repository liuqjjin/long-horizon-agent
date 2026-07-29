"""Oracle tests for basen.to_base."""

from basen import to_base


def test_zero_renders_single_zero():
    assert to_base(0, 2) == "0"
    assert to_base(0, 10) == "0"
    assert to_base(0, 16) == "0"


def test_most_significant_digit_first_binary():
    assert to_base(6, 2) == "110"
    assert to_base(8, 2) == "1000"
    assert to_base(255, 2) == "11111111"


def test_single_digit_values():
    assert to_base(1, 2) == "1"
    assert to_base(5, 10) == "5"
    assert to_base(10, 16) == "a"
    assert to_base(15, 16) == "f"


def test_hex_uses_letter_digits():
    assert to_base(255, 16) == "ff"
    assert to_base(31, 16) == "1f"
    assert to_base(4096, 16) == "1000"


def test_base_ten_is_not_reversed():
    assert to_base(123, 10) == "123"
    assert to_base(120, 10) == "120"


def test_boundary_bases():
    assert to_base(2, 2) == "10"
    assert to_base(16, 16) == "10"


def test_returns_str():
    assert isinstance(to_base(0, 2), str)
    assert isinstance(to_base(42, 16), str)
