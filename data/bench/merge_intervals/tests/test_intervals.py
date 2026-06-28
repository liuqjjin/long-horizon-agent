from intervals import merge


def test_touching_intervals_merge():
    # Intervals that meet at a single point must collapse into one.
    assert merge([[1, 2], [2, 3]]) == [[1, 3]]


def test_overlapping_intervals_merge():
    assert merge([[1, 4], [2, 5]]) == [[1, 5]]


def test_disjoint_intervals_with_gap_stay_separate():
    # A gap of one integer is NOT a touch; these must not be merged.
    assert merge([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_unsorted_input_is_sorted_and_merged():
    assert merge([[3, 4], [1, 2], [2, 3]]) == [[1, 4]]


def test_fully_contained_interval_keeps_outer_bounds():
    # A nested interval must not shrink the surrounding one.
    assert merge([[1, 5], [2, 3]]) == [[1, 5]]


def test_chained_touching_intervals_collapse():
    assert merge([[1, 2], [2, 3], [3, 4]]) == [[1, 4]]


def test_touching_at_a_point_with_negatives():
    assert merge([[-3, -1], [-1, 0]]) == [[-3, 0]]


def test_duplicate_intervals_collapse():
    assert merge([[1, 3], [1, 3]]) == [[1, 3]]


def test_zero_width_point_touches_neighbor():
    # A degenerate [1, 1] interval touches [1, 2] and merges.
    assert merge([[1, 1], [1, 2]]) == [[1, 2]]


def test_empty_input_returns_empty_list():
    assert merge([]) == []


def test_single_interval_is_unchanged():
    assert merge([[5, 7]]) == [[5, 7]]
