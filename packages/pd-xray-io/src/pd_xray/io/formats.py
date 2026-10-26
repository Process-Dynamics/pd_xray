"""Picks a format reader for a file and opens it as a `Dataset`."""

import importlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from pd_xray.core import PdXrayError
from pd_xray.io.dataset import Dataset


class UnsupportedFormatError(PdXrayError):
    """Raised when no reader handles a file's extension."""


class FileStore(Protocol):
    """Somewhere other than the local disk that files are read from, e.g. a NAS or S3 storage from `pd_xray.storage`
    (any object with these methods works). Paths are POSIX strings relative to its root.

    Readers ask only for the bytes they need, through `open`'s seekable file.
    """

    def open(self, path: str) -> BinaryIO:
        """The file at `path`, opened for reading. Raises FileNotFoundError if missing."""
        ...

    def stat(self, path: str) -> Any:
        """The entry at `path` (with an `is_dir` attribute). Raises FileNotFoundError if missing."""
        ...

    def list(self, path: str) -> Sequence[Any]:
        """Entries directly inside the folder `path` (with `name`, `path` and `is_dir` attributes)."""
        ...


@contextmanager
def readable(path: str | Path, storage: FileStore | None) -> Iterator[str | Path | BinaryIO]:
    """What a format library can open: `path` itself for local files, else the file opened from `storage` (closed
    on exit)."""
    if storage is None:
        yield path
        return
    with storage.open(str(path)) as file:
        yield file


# Extension -> "module:function". Modules are imported only when used, so a format's dependency is only needed when
# opening that format. Each function has the signature
# `(path: Path, axes: Sequence[str] | None, storage: FileStore | None, **options) -> Dataset`.
_OPENERS: dict[str, str] = {
    ".h5": "pd_xray.io.hdf5:open_hdf5",
    ".hdf5": "pd_xray.io.hdf5:open_hdf5",
    ".hdf": "pd_xray.io.hdf5:open_hdf5",
    ".nxs": "pd_xray.io.hdf5:open_hdf5",
    ".tif": "pd_xray.io.tiff:open_tiff",
    ".tiff": "pd_xray.io.tiff:open_tiff",
    ".png": "pd_xray.io.image:open_image",
    ".jpg": "pd_xray.io.image:open_image",
    ".jpeg": "pd_xray.io.image:open_image",
}


def supported_extensions() -> list[str]:
    """File extensions `open_dataset` can open."""
    return sorted(_OPENERS)


def open_dataset(
    path: str | Path, axes: Sequence[str] | None = None, storage: FileStore | None = None, **options: Any
) -> Dataset:
    """Open a file as a lazy `Dataset`, choosing the reader by file extension.

    Args:
        path: Local file path, or a path in `storage`.
        axes: Axis names overriding what the reader detects, e.g. ("angle", "y", "x").
        storage: Read the file from here (e.g. an `SFTPStorage`) instead of the local disk. Only the bytes needed
            are fetched; the file stays open until the dataset is closed.
        **options: Reader-specific options, e.g. `dataset="projections"` for HDF5.

    Raises:
        FileNotFoundError: If `path` doesn't exist.
        UnsupportedFormatError: If no reader handles the extension.
    """
    path = Path(path)
    if storage is None and not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    target = _OPENERS.get(path.suffix.lower())
    if target is None:
        raise UnsupportedFormatError(
            f"Can't open '{path.name}': unsupported extension. Supported: {', '.join(supported_extensions())}."
        )
    module_name, func_name = target.split(":")
    opener = getattr(importlib.import_module(module_name), func_name)
    dataset: Dataset = opener(path, axes, storage, **options)
    return dataset
