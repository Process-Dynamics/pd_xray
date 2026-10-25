"""Lazy, read-only datasets: array-like views of data in a file that read only what is indexed."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import Axes, validate_axes

FRAME = "frame"
"""Axis name readers use for a frame axis whose meaning the file doesn't state. Users can rename it."""


class Dataset(ABC):
    """Base class for lazy datasets.

    Indexing (`ds[i]`, `ds[10:20, :512]`) reads only the requested region and returns a NumPy array in the native
    dtype. Datasets have `shape`, `dtype` and `ndim`, so they also work with `dask.array.from_array`.

    Subclasses implement `__getitem__` and `close`.

    Attributes:
        shape: Array shape.
        dtype: Native dtype of the data.
        axes: Axis names (see `pd_xray.core.axes`), ending with ("y", "x") or ("y", "x", "channel").
        metadata: File/dataset attributes as plain Python values.
        warnings: Problems found while opening (e.g. missing frames), for the user to see.
    """

    def __init__(
        self,
        shape: Sequence[int],
        dtype: np.dtype[Any],
        axes: Sequence[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.axes: Axes = validate_axes(axes, len(self.shape))
        self.metadata = metadata or {}
        self.warnings: list[str] = []

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """Uncompressed size of the whole dataset in bytes."""
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @abstractmethod
    def __getitem__(self, key: Any) -> NDArray[Any]:
        """Read the indexed region."""

    @abstractmethod
    def close(self) -> None:
        """Release file handles. Safe to call more than once."""

    def __len__(self) -> int:
        return self.shape[0]

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, axes={self.axes})"


def split_key(key: Any, n: int) -> tuple[int | range | list[int], tuple[Any, ...]]:
    """Split an index into a resolved first-axis index and the rest of the key.

    For readers that fetch frames one at a time. Negative indices and slices are resolved against `n`.

    Returns:
        (first, rest): `first` is an int (axis dropped), a range, or a list of ints (from a list or boolean mask).
        `rest` indexes the remaining axes.

    Raises:
        IndexError: If a first-axis index is out of range.
    """
    key = key if isinstance(key, tuple) else (key,)
    if not key or key[0] is Ellipsis:
        first, rest = slice(None), key
    else:
        first, rest = key[0], key[1:]
    if isinstance(first, slice):
        return range(*first.indices(n)), rest
    if isinstance(first, (int, np.integer)):
        index = int(first) + n if first < 0 else int(first)
        if not 0 <= index < n:
            raise IndexError(f"Index {first} is out of range for an axis of length {n}.")
        return index, rest
    array = np.asarray(first)
    if array.dtype == bool:
        array = np.flatnonzero(array)
    indices = [int(i) + n if i < 0 else int(i) for i in array.ravel()]
    if any(not 0 <= i < n for i in indices):
        raise IndexError(f"Index list {array.tolist()} is out of range for an axis of length {n}.")
    return indices, rest


class ArrayDataset(Dataset):
    """A `Dataset` backed by an in-memory or memory-mapped NumPy array."""

    def __init__(self, array: NDArray[Any], axes: Sequence[str], metadata: dict[str, Any] | None = None) -> None:
        super().__init__(array.shape, array.dtype, axes, metadata)
        self._array = array

    def __getitem__(self, key: Any) -> NDArray[Any]:
        return np.asarray(self._array[key])

    def close(self) -> None:
        pass
