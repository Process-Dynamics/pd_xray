"""A dataset whose frames are another dataset's frames passed through a function (e.g. a processing pipeline)."""

import itertools
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import PdXrayError, frame_axes
from pd_xray.io.dataset import Dataset, split_key

FrameFunction = Callable[[NDArray[Any]], NDArray[Any]]


class MappedDataset(Dataset):
    """Frames of `source` passed through `function`, computed when read. Create it with `map_frames`.

    Shape and dtype come from processing the first frame once (kept, so it isn't processed again). The frame size
    may change (crop, rotate); the number of axes may not.
    """

    def __init__(
        self, source: Dataset, function: FrameFunction, workers: int = 1, metadata: dict[str, Any] | None = None
    ) -> None:
        if workers < 1:
            raise PdXrayError(f"workers must be at least 1, got {workers}.")
        self.source, self.function, self.workers = source, function, workers
        self._n = len(frame_axes(source.axes))
        counts = source.shape[: self._n]
        if 0 in counts:
            raise PdXrayError("The dataset has no frames to process.")
        self._first_index = (0,) * self._n
        self._first = np.asarray(function(source[self._first_index] if self._n else source[...]))
        if self._first.ndim != source.ndim - self._n:
            raise PdXrayError(f"The function turned {source.ndim - self._n}D frames into {self._first.ndim}D ones.")
        super().__init__(
            counts + self._first.shape, self._first.dtype, source.axes, {**source.metadata, **(metadata or {})}
        )
        self.warnings = list(source.warnings)

    def _frame(self, index: tuple[int, ...]) -> NDArray[Any]:
        if index == self._first_index:
            return self._first.copy()
        return np.asarray(self.function(self.source[index] if index else self.source[...]))

    def __getitem__(self, key: Any) -> NDArray[Any]:
        key = key if isinstance(key, tuple) else (key,)
        if Ellipsis in key:
            at = key.index(Ellipsis)
            key = key[:at] + (slice(None),) * (self.ndim - len(key) + 1) + key[at + 1 :]
        frame_key = key[: self._n] + (slice(None),) * max(self._n - len(key), 0)
        selected = [split_key(k, n)[0] for k, n in zip(frame_key, self.shape[: self._n], strict=True)]
        indices = [[s] if isinstance(s, int) else list(s) for s in selected]
        kept = [len(i) for s, i in zip(selected, indices, strict=True) if not isinstance(s, int)]
        frames = [self._frame(index) for index in itertools.product(*indices)]
        stacked = np.stack(frames) if frames else np.empty((0, *self._first.shape), self.dtype)
        return np.asarray(stacked.reshape(*kept, *self._first.shape)[(slice(None),) * len(kept) + key[self._n :]])

    def iter_frames(self, first: Sequence[int] | None = None) -> Iterator[NDArray[Any]]:
        """Frames in order. With `workers` > 1, the source is read in this thread and up to `workers` frames are
        processed at once in a thread pool (about twice `workers` frames held in memory)."""
        if self.workers == 1:
            yield from super().iter_frames(first)
            return
        pending: deque[Future[NDArray[Any]]] = deque()
        with ThreadPoolExecutor(self.workers, thread_name_prefix="pd-xray-map") as pool:
            try:
                for frame in self.source.iter_frames(first):
                    pending.append(pool.submit(self.function, frame))
                    if len(pending) >= 2 * self.workers:
                        yield np.asarray(pending.popleft().result())
                while pending:
                    yield np.asarray(pending.popleft().result())
            finally:  # stopped early (cancel, error): don't process the frames still queued
                for future in pending:
                    future.cancel()

    def close(self) -> None:
        """Does nothing: the source belongs to the caller."""


def map_frames(
    dataset: Dataset, function: FrameFunction, workers: int = 1, metadata: dict[str, Any] | None = None
) -> MappedDataset:
    """A lazy dataset of `function(frame)` for every frame of `dataset`, e.g. a `pd_xray.processing.Pipeline`.

    Pass it to `convert` to process a whole dataset into a file:
    `convert(map_frames(dataset, pipeline, workers=8), "out.tif")`.

    Args:
        dataset: Source dataset. It stays open; close it yourself.
        function: Takes one frame (y, x) or (y, x, channel), returns one frame with the same number of axes. Must be
            thread-safe if `workers` > 1 (pipelines are).
        workers: Frames processed at once when iterating (`iter_frames`, `convert`). Reading stays sequential.
        metadata: Added to the source's metadata.

    Raises:
        PdXrayError: If the dataset has no frames, or `function` changes the number of axes.
    """
    return MappedDataset(dataset, function, workers, metadata)
