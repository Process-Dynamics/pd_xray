"""A sequence of single-frame image files (TIFF/PNG/JPG) opened as one dataset."""

import fnmatch
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import CHANNEL, PdXrayError, X, Y
from pd_xray.io.dataset import FRAME, Dataset, split_key
from pd_xray.io.formats import FileStore

Reader = Callable[[Path, FileStore | None], tuple[NDArray[Any], dict[str, Any]]]


def _frame_reader(path: Path) -> Reader:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        from pd_xray.io.tiff import read_tiff_frame

        return read_tiff_frame
    if suffix in (".png", ".jpg", ".jpeg"):
        from pd_xray.io.image import read_image

        return read_image
    raise PdXrayError(f"Unsupported file in sequence: {path.name}")


def natural_key(path: Path) -> list[Any]:
    """Sort key that orders embedded numbers numerically: frame_2 < frame_10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


class FileSequence(Dataset):
    """Frames stored one per file. Frame `i` is `files[i]`, read on demand.

    Shape, dtype and metadata come from the first file. A file that doesn't match raises when read.

    Attributes:
        files: The files, in frame order.
    """

    def __init__(
        self, files: Sequence[Path], axes: Sequence[str] | None = None, storage: FileStore | None = None
    ) -> None:
        if not files:
            raise PdXrayError("No files to open as a sequence.")
        self.files = list(files)
        self.storage = storage
        self._read = _frame_reader(self.files[0])
        first, metadata = self._read(self.files[0], storage)
        default_axes = (FRAME, Y, X) if first.ndim == 2 else (FRAME, Y, X, CHANNEL)
        super().__init__((len(self.files), *first.shape), first.dtype, axes or default_axes, metadata)

    def _frame(self, index: int) -> NDArray[Any]:
        frame = self._read(self.files[index], self.storage)[0]
        if frame.shape != self.shape[1:] or frame.dtype != self.dtype:
            raise PdXrayError(
                f"{self.files[index].name} is {frame.shape} {frame.dtype}, but the sequence is "
                f"{self.shape[1:]} {self.dtype}."
            )
        return frame

    def __getitem__(self, key: Any) -> NDArray[Any]:
        first, rest = split_key(key, self.shape[0])
        if isinstance(first, int):
            return np.asarray(self._frame(first)[rest])
        frames = [self._frame(i) for i in first]
        stack = np.stack(frames) if frames else np.empty((0, *self.shape[1:]), self.dtype)
        return stack[(slice(None),) + rest]

    def close(self) -> None:
        pass


def open_sequence(
    source: str | Path | Sequence[str | Path],
    pattern: str = "*",
    axes: Sequence[str] | None = None,
    storage: FileStore | None = None,
) -> FileSequence:
    """Open single-frame files as one dataset.

    Args:
        source: A directory (files matching `pattern`, in natural order) or an explicit list of files (kept in the
            given order).
        pattern: Glob pattern within the directory, e.g. "*_img.tif" or "*_dar.tif".
        axes: Axis names. Defaults to ("frame", "y", "x").
        storage: Read the files from here instead of the local disk (see `open_dataset`); paths are in it.

    Example:
        >>> images = open_sequence("data/0042", "*_img.tif")
        >>> darks = open_sequence("data/0042", "*_dar.tif")
    """
    files = find_files(source, pattern, storage) if isinstance(source, (str, Path)) else [Path(p) for p in source]
    return FileSequence(files, axes, storage)


def find_files(directory: str | Path, pattern: str, storage: FileStore | None = None) -> list[Path]:
    """Files directly in `directory` (local, or in `storage`) matching the glob `pattern`, in natural order.

    Raises:
        NotADirectoryError: If `directory` isn't a folder.
        PdXrayError: If no file matches.
    """
    directory = Path(directory)
    if storage is not None:
        if not storage.stat(str(directory)).is_dir:
            raise NotADirectoryError(f"Not a directory: {directory}")
        found = [
            Path(e.path) for e in storage.list(str(directory)) if not e.is_dir and fnmatch.fnmatch(e.name, pattern)
        ]
    elif not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    else:
        found = [p for p in directory.glob(pattern) if p.is_file()]
    files = sorted(found, key=natural_key)
    if not files:
        raise PdXrayError(f"No files match '{pattern}' in {directory}.")
    return files
