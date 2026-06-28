import pytest
from paginate import num_pages


def test_exact_multiples_have_no_extra_page():
    # When items divide evenly, the count equals the exact division.
    assert num_pages(9, 3) == 3
    assert num_pages(6, 3) == 2
    assert num_pages(3, 3) == 1


def test_partial_final_page_is_counted():
    # A leftover item needs its own page.
    assert num_pages(10, 3) == 4
    assert num_pages(7, 3) == 3
    assert num_pages(1, 3) == 1


def test_boundaries_around_one_page():
    assert num_pages(2, 3) == 1  # fewer than a full page still needs one page
    assert num_pages(4, 3) == 2  # one over a full page needs a second page


def test_single_item_and_single_per_page():
    assert num_pages(1, 1) == 1
    assert num_pages(5, 1) == 5


def test_zero_items_is_zero_pages():
    assert num_pages(0, 3) == 0
    assert num_pages(0, 1) == 0


def test_large_values():
    assert num_pages(100, 7) == 15
    assert num_pages(1000, 999) == 2


def test_non_positive_per_page_raises():
    with pytest.raises(ValueError):
        num_pages(10, 0)
    with pytest.raises(ValueError):
        num_pages(10, -3)


def test_negative_total_raises():
    with pytest.raises(ValueError):
        num_pages(-1, 3)
