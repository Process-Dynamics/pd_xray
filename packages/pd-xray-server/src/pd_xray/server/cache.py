"""Thread-safe LRU cache bounded by size in bytes."""

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from typing import Any

import numpy as np


def size_of(value: Any) -> int:
    """Approximate memory use of a cached value in bytes."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, tuple):
        return sum(size_of(item) for item in value)
    return 256


class LRUCache:
    """Least-recently-used cache bounded by total size.

    `get_or_compute` computes a missing value once: concurrent callers asking for the same key wait for the first
    caller's result instead of reading the same frame again.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._items: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._pending: dict[Hashable, Future[Any]] = {}
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def nbytes(self) -> int:
        """Current total size in bytes."""
        return self._bytes

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Cached value for `key`, calling `compute()` if missing. Exceptions from `compute` aren't cached."""
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key][0]
            future = self._pending.get(key)
            if future is None:
                future = self._pending[key] = Future()
                owner = True
            else:
                owner = False
        if not owner:
            return future.result()
        try:
            value = compute()
        except BaseException as error:
            with self._lock:
                del self._pending[key]
            future.set_exception(error)
            raise
        size = size_of(value)
        with self._lock:
            del self._pending[key]
            if size <= self.max_bytes:
                self._items[key] = (value, size)
                self._bytes += size
                while self._bytes > self.max_bytes:
                    self._bytes -= self._items.popitem(last=False)[1][1]
        future.set_result(value)
        return value

    def discard(self, predicate: Callable[[Hashable], bool]) -> None:
        """Remove every entry whose key matches `predicate`."""
        with self._lock:
            for key in [key for key in self._items if predicate(key)]:
                self._bytes -= self._items.pop(key)[1]
