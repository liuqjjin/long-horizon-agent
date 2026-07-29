"""Oracle tests for stats.median."""

import pytest
from stats import median


def test_odd_basic():
    assert median([1, 2, 3]) == 2


def test_single_element():
    assert median([5]) == 5


def test_even_basic_average():
    # The two middle elements are 2 and 3; the median is their mean.
    assert median([1, 2, 3, 4]) == 2.5


def test_even_pair():
    assert median([1, 2]) == 1.5


def test_even_clean_average():
    # Middles 20 and 30 -> 25.0, distinct from both lower and upper middle.
    assert median([10, 20, 30, 40]) == 25.0


def test_even_zero_crossing():
    assert median([0, 1]) == 0.5


def test_even_with_duplicates():
    assert median([2, 2, 4, 4]) == 3.0


def test_negative_even():
    assert median([-3, -1]) == -2.0


def test_negative_odd():
    assert median([-5, -3, -1]) == -3


def test_floats_even():
    assert median([1.5, 2.5, 3.5, 4.5]) == 3.0


def test_unsorted_input_is_sorted():
    assert median([4, 2, 1, 3]) == 2.5


def test_empty_raises():
    with pytest.raises(ValueError):
        median([])
