import pytest
from lru import LRUCache


def test_basic_put_get():
    c = LRUCache(2)
    c.put("a", 1)
    assert c.get("a") == 1
    assert c.get("missing") is None


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        LRUCache(0)


def test_evicts_least_recently_used_on_overflow():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # over capacity -> "a" is the LRU entry
    assert c.get("a") is None
    assert c.get("b") == 2 and c.get("c") == 3


def test_get_refreshes_recency():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1  # "a" is now most-recently-used
    c.put("c", 3)  # evicts "b", not "a"
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3


def test_put_of_existing_key_refreshes_recency():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 9)  # update counts as use -> "a" is most-recently-used
    c.put("c", 3)  # evicts "b"
    assert c.get("b") is None
    assert c.get("a") == 9 and c.get("c") == 3


def test_update_of_existing_key_never_evicts():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 5)  # same key: no new entry, nothing may be evicted
    assert len(c) == 2
    assert c.get("a") == 5 and c.get("b") == 2


def test_capacity_one_churn():
    c = LRUCache(1)
    c.put("a", 1)
    assert c.get("a") == 1
    c.put("b", 2)
    assert c.get("a") is None
    assert c.get("b") == 2
