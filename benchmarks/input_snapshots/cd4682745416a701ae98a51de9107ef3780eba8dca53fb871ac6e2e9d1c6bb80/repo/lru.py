"""A bounded least-recently-used cache.

Both reads and writes count as "use": ``get(key)`` makes the key the
most-recently-used entry, and ``put`` of an existing key updates its value and
refreshes it the same way. When a ``put`` of a new key would exceed
``capacity``, the least-recently-used entry is evicted first. ``get`` of a
missing key returns ``None``.
"""


class LRUCache:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._data: dict = {}  # dicts preserve insertion order

    def get(self, key):
        if key not in self._data:
            return None
        return self._data[key]

    def put(self, key, value) -> None:
        if key in self._data:
            self._data[key] = value
            return
        if len(self._data) >= self.capacity:
            oldest = next(iter(self._data))
            del self._data[oldest]
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)
