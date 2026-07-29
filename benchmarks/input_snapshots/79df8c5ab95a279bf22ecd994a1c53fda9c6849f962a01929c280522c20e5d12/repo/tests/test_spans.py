from spans import merge_spans


def test_empty_input():
    assert merge_spans([]) == []


def test_disjoint_spans_stay_separate():
    assert merge_spans([(1, 2), (4, 6)]) == [(1, 2), (4, 6)]


def test_overlapping_spans_merge():
    assert merge_spans([(1, 3), (2, 4)]) == [(1, 4)]


def test_unsorted_input_still_merges():
    assert merge_spans([(5, 7), (1, 3), (2, 4)]) == [(1, 4), (5, 7)]


def test_touching_spans_do_not_merge():
    # [1,2) and [2,3) share no point: touching is not overlapping.
    assert merge_spans([(1, 2), (2, 3)]) == [(1, 2), (2, 3)]


def test_unsorted_touching_spans_sort_but_stay_separate():
    assert merge_spans([(2, 3), (0, 1), (1, 2)]) == [(0, 1), (1, 2), (2, 3)]


def test_contained_span_is_absorbed():
    assert merge_spans([(1, 10), (2, 3)]) == [(1, 10)]


def test_duplicate_spans_collapse():
    assert merge_spans([(1, 3), (1, 3)]) == [(1, 3)]


def test_chain_of_overlaps_merges_into_one():
    assert merge_spans([(1, 5), (2, 3), (4, 8)]) == [(1, 8)]


def test_result_is_sorted_by_start():
    assert merge_spans([(8, 9), (4, 6), (0, 2)]) == [(0, 2), (4, 6), (8, 9)]
