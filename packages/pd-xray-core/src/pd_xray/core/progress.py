"""Progress reporting and cancellation for long-running operations.

Library functions accept optional `progress` and `cancel` arguments. The caller (CLI, server, notebook) decides how to
display progress. Libraries never print.
"""

import threading
from collections.abc import Callable, Iterable, Iterator, Sized
from dataclasses import dataclass
from typing import TypeVar

from pd_xray.core.errors import OperationCancelledError

T = TypeVar("T")


@dataclass(frozen=True)
class Progress:
    """A progress update.

    Attributes:
        current: Units of work completed (frames, bytes, …).
        total: Total units of work, or None if unknown.
        message: Short human-readable description of the current activity.
    """

    current: int
    total: int | None = None
    message: str = ""

    @property
    def fraction(self) -> float | None:
        """Completed fraction in [0, 1], or None if the total is unknown."""
        if not self.total:
            return None
        return min(self.current / self.total, 1.0)


ProgressCallback = Callable[[Progress], None]
"""Called with each `Progress` update. Must be fast and thread-safe; callers throttle if needed."""


class CancelToken:
    """Thread-safe flag a caller sets to ask a running operation to stop.

    Operations call `raise_if_cancelled()` at safe points (e.g. between frames).
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation was requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise `OperationCancelledError` if cancellation was requested."""
        if self._event.is_set():
            raise OperationCancelledError("Operation was cancelled.")


def track(
    items: Iterable[T],
    total: int | None = None,
    message: str = "",
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> Iterator[T]:
    """Yield from `items`, checking for cancellation before each item and reporting progress after it.

    Args:
        items: Work items, e.g. frame indices.
        total: Number of items. Taken from `len(items)` when omitted and available.
        message: Message attached to each progress update.
        progress: Optional progress callback.
        cancel: Optional cancel token.

    Raises:
        OperationCancelledError: If `cancel` is cancelled.
    """
    if total is None and isinstance(items, Sized):
        total = len(items)
    if progress is not None:
        progress(Progress(0, total, message))
    for done, item in enumerate(items, start=1):
        if cancel is not None:
            cancel.raise_if_cancelled()
        yield item
        if progress is not None:
            progress(Progress(done, total, message))
