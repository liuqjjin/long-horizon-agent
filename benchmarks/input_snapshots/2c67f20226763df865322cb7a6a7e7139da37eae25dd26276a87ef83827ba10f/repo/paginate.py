"""Pagination helpers."""


def num_pages(total_items: int, per_page: int) -> int:
    """Return how many pages of size ``per_page`` are needed for ``total_items``.

    ``per_page`` must be a positive integer. ``total_items`` must not be
    negative.
    """
    if per_page <= 0:
        raise ValueError("per_page must be positive")
    if total_items < 0:
        raise ValueError("total_items must not be negative")
    return total_items // per_page
