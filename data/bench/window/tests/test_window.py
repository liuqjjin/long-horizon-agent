from window import moving_average


def test_basic_window_of_two():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_window_of_three():
    assert moving_average([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]


def test_output_length():
    assert len(moving_average(list(range(10)), 4)) == 7


def test_width_one_returns_each_element():
    assert moving_average([5, 10, 15], 1) == [5.0, 10.0, 15.0]


def test_window_equals_length_is_single_average():
    assert moving_average([2, 4, 6], 3) == [4.0]


def test_empty_input_is_empty():
    assert moving_average([], 1) == []


def test_negatives_and_duplicates():
    assert moving_average([-1, -1, 2, 2], 2) == [-1.0, 0.5, 2.0]
