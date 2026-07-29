"""Oracle tests for to_roman."""

import pytest
from roman import to_roman


def test_subtractive_ones():
    assert to_roman(4) == "IV"
    assert to_roman(9) == "IX"


def test_subtractive_tens():
    assert to_roman(40) == "XL"
    assert to_roman(90) == "XC"


def test_subtractive_hundreds():
    assert to_roman(400) == "CD"
    assert to_roman(900) == "CM"


def test_non_subtractive_still_additive():
    assert to_roman(3) == "III"
    assert to_roman(8) == "VIII"
    assert to_roman(38) == "XXXVIII"


def test_no_illegal_subtractive_pairs():
    # 49 must be XLIX, never "IL"; 99 must be XCIX, never "IC";
    # 999 must be CMXCIX, never "IM".
    assert to_roman(49) == "XLIX"
    assert to_roman(99) == "XCIX"
    assert to_roman(999) == "CMXCIX"


def test_combined_numbers():
    assert to_roman(14) == "XIV"
    assert to_roman(1994) == "MCMXCIV"
    assert to_roman(2444) == "MMCDXLIV"
    assert to_roman(3549) == "MMMDXLIX"


def test_boundaries():
    assert to_roman(1) == "I"
    assert to_roman(3999) == "MMMCMXCIX"


@pytest.mark.parametrize("bad", [0, -1, -5, 4000, 5000])
def test_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        to_roman(bad)


@pytest.mark.parametrize("bad", [True, 3.0, "5", None])
def test_non_int_raises(bad):
    with pytest.raises(TypeError):
        to_roman(bad)
