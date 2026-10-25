"""TIFF reader for single- and multi-page files (extra `[tiff]`: tifffile)."""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import CHANNEL, TIME, PdXrayError, X, Y, Z, import_optional, validate_axes
from pd_xray.io.dataset import FRAME, Dataset, split_key

# tifffile axis codes -> our axis names. Unlisted codes (I, Q, …) become "frame".
_AXIS_CODES = {"T": TIME, "Z": Z, "Y": Y, "X": X, "S": CHANNEL, "C": CHANNEL}


def _tifffile() -> Any:
    return import_optional("tifffile", "tiff", "pd-xray-io")


def _default_axes(series: Any) -> tuple[str, ...]:
    """Axes from tifffile's series description, or a neutral default."""
    axes = series.axes
    names = tuple(axes) if not isinstance(axes, str) else tuple(_AXIS_CODES.get(code, FRAME) for code in axes)
    try:
        return validate_axes(names, len(series.shape))
    except ValueError:
        pass
    shape = series.shape
    if len(shape) == 2:
        return (Y, X)
    if len(shape) == 3:
        return (Y, X, CHANNEL) if shape[-1] in (3, 4) and axes[-1:] in ("S", (CHANNEL,)) else (FRAME, Y, X)
    raise PdXrayError(f"Can't tell the axes of a {len(shape)}D TIFF series ({axes}). Pass `axes`.")


def _metadata(tif: Any) -> dict[str, Any]:
    """tifffile/ImageJ metadata plus TIFF tags that matter for interpretation."""
    metadata: dict[str, Any] = {}
    if tif.shaped_metadata:
        metadata.update({k: v for k, v in tif.shaped_metadata[0].items() if k not in ("shape", "axes")})
    if tif.imagej_metadata:
        metadata.update(tif.imagej_metadata)
    orientation = tif.pages[0].tags.get("Orientation")
    if orientation is not None:
        value = orientation.value
        value = value[0] if isinstance(value, bytes) else int(value)
        if value != 1:
            metadata["tiff_orientation"] = value  # recorded, not applied (ImageJ ignores it too)
    return metadata


class TIFFDataset(Dataset):
    """Lazy view of the first image series in a TIFF file.

    Uncompressed, contiguous files are memory-mapped. Otherwise pages are read on demand.
    Keeps the file open until `close()`.
    """

    def __init__(self, path: str | Path, axes: Sequence[str] | None = None) -> None:
        tifffile = _tifffile()
        self.path = Path(path)
        self._tif = tifffile.TiffFile(self.path)
        self._lock = threading.Lock()
        try:
            series = self._tif.series[0]
            super().__init__(series.shape, series.dtype, axes or _default_axes(series), _metadata(self._tif))
            self._memmap: NDArray[Any] | None = None
            if series.dataoffset is not None:
                self._memmap = tifffile.memmap(self.path, series=0, mode="r")
            self._pages = series.pages
            page_shape = tuple(series.keyframe.shape)
            if self._memmap is None and self.shape not in (page_shape, (len(self._pages), *page_shape)):
                raise PdXrayError(
                    f"{self.path.name}: compressed TIFF series of shape {self.shape} isn't supported yet."
                )
        except Exception:
            self._tif.close()
            raise

    def _page(self, index: int) -> NDArray[Any]:
        with self._lock:  # one file handle; tifffile page reads aren't safe to run concurrently
            return np.asarray(self._pages[index].asarray())

    def __getitem__(self, key: Any) -> NDArray[Any]:
        if self._memmap is not None:
            return np.array(self._memmap[key])
        if len(self._pages) == 1 and self.shape == tuple(self._pages[0].shape):
            return np.asarray(self._page(0)[key])
        first, rest = split_key(key, self.shape[0])
        if isinstance(first, int):
            return np.asarray(self._page(first)[rest])
        frames = [self._page(i) for i in first]
        stack = np.stack(frames) if frames else np.empty((0, *self.shape[1:]), self.dtype)
        return stack[(slice(None),) + rest]

    def close(self) -> None:
        self._memmap = None
        self._tif.close()


def open_tiff(path: Path, axes: Sequence[str] | None = None) -> TIFFDataset:
    """Open a TIFF file. Axes default to those stored by tifffile/ImageJ, else ("frame", "y", "x") / ("y", "x")."""
    return TIFFDataset(path, axes)


def read_tiff_frame(path: Path) -> tuple[NDArray[Any], dict[str, Any]]:
    """Read the first page of a TIFF file, plus its metadata."""
    with _tifffile().TiffFile(path) as tif:
        return np.asarray(tif.pages[0].asarray()), _metadata(tif)
