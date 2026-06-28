from search import bsearch


def test_finds_first_element():
    assert bsearch([1, 3, 5, 7], 1) == 0


def test_finds_last_element():
    assert bsearch([1, 3, 5, 7], 7) == 3


def test_finds_interior_elements():
    assert bsearch([1, 3, 5, 7], 3) == 1
    assert bsearch([1, 3, 5, 7], 5) == 2


def test_empty_list_returns_minus_one():
    assert bsearch([], 5) == -1


def test_single_element():
    assert bsearch([42], 42) == 0
    assert bsearch([42], 7) == -1


def test_target_absent_returns_minus_one():
    assert bsearch([1, 3, 5, 7], 4) == -1
    assert bsearch([1, 3, 5, 7], 0) == -1
    assert bsearch([1, 3, 5, 7], 9) == -1


def test_negative_values():
    assert bsearch([-5, -3, -1, 2, 4], -5) == 0
    assert bsearch([-5, -3, -1, 2, 4], -1) == 2
    assert bsearch([-5, -3, -1, 2, 4], 4) == 4


def test_duplicates_return_a_valid_index():
    arr = [1, 2, 2, 2, 3]
    idx = bsearch(arr, 2)
    assert idx != -1
    assert arr[idx] == 2


def test_even_length_boundaries():
    arr = [10, 20, 30, 40, 50, 60]
    assert bsearch(arr, 10) == 0
    assert bsearch(arr, 60) == 5
