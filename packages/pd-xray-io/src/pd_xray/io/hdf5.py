"""HDF5 reader (extra `[hdf5]`: h5py + hdf5plugin)."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import PdXrayError, X, Y, import_optional
from pd_xray.io.dataset import FRAME, Dataset, split_key

logger = logging.getLogger(__name__)


class DatasetSelectionError(PdXrayError):
    """Raised when an HDF5 file holds several image datasets and none was chosen, or the chosen one is invalid."""


@dataclass(frozen=True)
class HDF5DatasetInfo:
    """One dataset inside an HDF5 file, as listed by `inspect_hdf5`.

    Attributes:
        name: Path inside the file, e.g. "entry/data/data".
        shape: Array shape, corrected to the written frames if the stored extent is wrong (see `warnings`).
        dtype: NumPy dtype string, e.g. "uint16".
        chunks: Chunk shape, or None if stored contiguously.
        compression: Filter names joined by "+", e.g. "blosc", or None.
        attrs: Dataset attributes as plain Python values.
        warnings: Problems found, e.g. a wrong extent or missing frames.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None
    compression: str | None
    attrs: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def is_image(self) -> bool:
        """Whether this looks like image data: at least 2D and numeric."""
        return len(self.shape) >= 2 and np.dtype(self.dtype).kind in "biuf"


def _h5py() -> Any:
    import_optional("hdf5plugin", "hdf5", "pd-xray-io")  # registers extra compression filters with h5py
    return import_optional("h5py", "hdf5", "pd-xray-io")


def _to_python(value: Any) -> Any:
    """Convert an HDF5 attribute value to a plain Python value."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        return [_to_python(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


# HDF5 dimension-scale bookkeeping, exposed as axes instead of metadata.
_INTERNAL_ATTRS = {"DIMENSION_LABELS", "DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME"}


def _attrs(obj: Any) -> dict[str, Any]:
    return {key: _to_python(value) for key, value in obj.attrs.items() if key not in _INTERNAL_ATTRS}


def _compression(obj: Any) -> str | None:
    plist = obj.id.get_create_plist()
    names = [plist.get_filter(i)[3].decode(errors="replace") for i in range(plist.get_nfilters())]
    return "+".join(names) or None


def _checked_extent(obj: Any) -> tuple[tuple[int, ...], list[int], list[str]]:
    """Check the first axis of a chunked, growable dataset against the chunks actually written.

    Writers that crash can leave a wrong extent (e.g. 1e19 frames) or unwritten frames, which h5py reads as the fill
    value. Only datasets with an unlimited first axis are checked; this reads the chunk index, not pixel data.

    Returns:
        (shape trimmed to the last written frame, missing frame indices, warnings).
    """
    shape = tuple(obj.shape)
    if obj.chunks is None or not shape or obj.maxshape[0] is not None:
        return shape, [], []
    offsets: set[int] = set()
    if hasattr(obj.id, "chunk_iter"):
        obj.id.chunk_iter(lambda info: offsets.add(info.chunk_offset[0]))
    else:
        offsets.update(obj.id.get_chunk_info(i).chunk_offset[0] for i in range(obj.id.get_num_chunks()))
    step = obj.chunks[0]
    end = min(shape[0], max(offsets) + step) if offsets else 0
    missing = [f for start in range(0, end, step) if start not in offsets for f in range(start, min(start + step, end))]
    warnings = []
    if end < shape[0]:
        warnings.append(f"The file claims {shape[0]} frames but only {end} are written; using {end}.")
    if missing:
        warnings.append(
            f"{len(missing)} frames were never written and read as {obj.fillvalue}: {missing[:10]}"
            + (" …" if len(missing) > 10 else "")
        )
    return (end, *shape[1:]), missing, warnings


def inspect_hdf5(path: str | Path) -> list[HDF5DatasetInfo]:
    """List every dataset in an HDF5 file without reading any pixel data.

    Args:
        path: Local file path.

    Returns:
        Datasets in file order. Use `HDF5DatasetInfo.is_image` to filter image candidates.
    """
    h5py = _h5py()
    found: list[HDF5DatasetInfo] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            shape, _, warnings = _checked_extent(obj)
            found.append(
                HDF5DatasetInfo(
                    name, shape, str(obj.dtype), obj.chunks, _compression(obj), _attrs(obj), tuple(warnings)
                )
            )

    with h5py.File(path, "r") as file:
        file.visititems(visit)
    return found


def _default_axes(h5_dataset: Any, name: str) -> tuple[str, ...]:
    """Axes from the dataset's dimension labels, or a neutral default for 2D/3D data."""
    labels = [dim.label.strip().lower() for dim in h5_dataset.dims]
    if all(labels):
        return tuple(labels)
    ndim = h5_dataset.ndim
    if ndim == 2:
        return (Y, X)
    if ndim == 3:
        return (FRAME, Y, X)
    raise DatasetSelectionError(
        f"Dataset '{name}' has {ndim} dimensions and no dimension labels. "
        "Pass `axes`, e.g. ('time', 'angle', 'y', 'x')."
    )


class HDF5Dataset(Dataset):
    """Lazy view of one dataset in an HDF5 file. Keeps the file open until `close()`.

    If the stored extent is wrong, `shape` covers only the written frames, and `metadata["missing_frames"]` lists
    frames that were never written.
    """

    def __init__(self, path: str | Path, name: str, axes: Sequence[str] | None = None) -> None:
        h5py = _h5py()
        self.path = Path(path)
        self.name = name
        self._file = h5py.File(self.path, "r")
        try:
            obj = self._file.get(name)
            if not isinstance(obj, h5py.Dataset):
                raise DatasetSelectionError(f"'{name}' is not a dataset in {self.path.name}.")
            self._dataset = obj
            shape, missing, warnings = _checked_extent(obj)
            metadata = _attrs(obj)
            if missing:
                metadata["missing_frames"] = missing
            super().__init__(shape, obj.dtype, axes or _default_axes(obj, name), metadata)
            self.warnings.extend(warnings)
            for warning in warnings:
                logger.warning("%s:%s: %s", self.path.name, name, warning)
        except Exception:
            self._file.close()
            raise

    def __getitem__(self, key: Any) -> NDArray[Any]:
        first, rest = split_key(key, self.shape[0])
        if isinstance(first, int):
            return np.asarray(self._dataset[(first,) + rest])
        if not first:
            return np.asarray(self._dataset[(slice(0, 0),) + rest])
        if isinstance(first, range) and first.step > 0:
            return np.asarray(self._dataset[(slice(first.start, first.stop, first.step),) + rest])
        # h5py needs increasing, unique index lists: read those, then restore the requested order.
        unique, inverse = np.unique(list(first), return_inverse=True)
        return np.asarray(self._dataset[(unique.tolist(),) + rest])[inverse]

    def close(self) -> None:
        self._file.close()


def open_hdf5(path: Path, axes: Sequence[str] | None = None, dataset: str | None = None) -> HDF5Dataset:
    """Open a dataset in an HDF5 file.

    Args:
        path: Local file path.
        axes: Axis names. Defaults to the dimension labels, or ("frame", "y", "x") / ("y", "x") if unlabelled.
        dataset: Path of the dataset inside the file. May be omitted if the file holds exactly one image dataset.

    Raises:
        DatasetSelectionError: If `dataset` is omitted and the file has zero or several image datasets.
    """
    if dataset is None:
        candidates = [info.name for info in inspect_hdf5(path) if info.is_image]
        if len(candidates) != 1:
            raise DatasetSelectionError(
                f"{path.name} has {len(candidates)} image datasets; choose one with `dataset=`: {candidates}"
            )
        dataset = candidates[0]
    return HDF5Dataset(path, dataset, axes)
