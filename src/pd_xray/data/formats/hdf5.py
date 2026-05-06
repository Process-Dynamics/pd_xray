import h5py  # type: ignore[import]
import hdf5plugin  # type: ignore[import]  # noqa: F401 — registers compression codecs as a side effect
import numpy as np
from collections.abc import Generator
from contextlib import contextmanager
from numpy.typing import DTypeLike, NDArray
from pathlib import Path
from typing import Any

from pd_xray.data.formats.base import FormatReader
from pd_xray.core.logging import get_logger
from pd_xray.core import T

logger = get_logger(__name__)

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_EXTENSIONS = {".hdf5", ".h5", ".nxs"}


def _find_hdf5_dataset(hf: Any, dataset_path: str | None) -> str:
    """Return dataset_path as-is, or auto-detect when there is exactly one dataset."""
    if dataset_path is not None:
        return dataset_path
    found: list[str] = []

    def _collect(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            found.append(f"/{name}")

    hf.visititems(_collect)
    if not found:
        raise ValueError("No datasets found in HDF5 file")
    if len(found) == 1:
        return found[0]
    raise ValueError(
        f"Multiple datasets found; specify dataset_path. Available: {found}"
    )


class LazyHDF5Array:
    """Lazy wrapper around an HDF5 dataset, no data is loaded until you slice it.

    Supports numpy-style indexing:  ``arr[0]``, ``arr[10:50]``, ``arr[::2, :, :]``

    Two modes:
    - Local  : created via __init__(path, ...). File is opened/closed per __getitem__.
    - Remote : created via _from_dataset(dataset, ...). Works with an already-open
               h5py.Dataset (e.g. one backed by a streaming S3 file). The caller is
               responsible for keeping the h5py.File open for the array's lifetime.
    """

    def __init__(self, path: Path, dataset_path: str, dtype: DTypeLike = np.float32) -> None:
        self._path: Path | None = path
        self._dataset_path = dataset_path
        self._dtype = np.dtype(dtype)
        self._dataset_ref: Any = None
        with h5py.File(str(path), "r") as f:
            ds = f[dataset_path]
            self._shape: tuple[int, ...] = ds.shape
            self._h5dtype: str = str(ds.dtype)

    @classmethod
    def _from_dataset(cls, dataset: Any, dtype: DTypeLike = np.float32) -> "LazyHDF5Array":
        """Create a LazyHDF5Array from an already-open h5py.Dataset (remote mode)."""
        obj = object.__new__(cls)
        obj._path = None
        obj._dataset_path = dataset.name
        obj._dtype = np.dtype(dtype)
        obj._shape = tuple(dataset.shape)
        obj._h5dtype = str(dataset.dtype)
        obj._dataset_ref = dataset
        return obj

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def __getitem__(self, key: Any) -> NDArray:
        if self._path is not None:
            with h5py.File(str(self._path), "r") as f:
                data = f[self._dataset_path][key]
        else:
            data = self._dataset_ref[key]
        return np.asarray(data, dtype=self._dtype)

    def __len__(self) -> int:
        return self._shape[0]

    def __repr__(self) -> str:
        source = self._path.name if self._path is not None else "<remote>"
        return (
            f"LazyHDF5Array(source={source!r}, "
            f"dataset={self._dataset_path!r}, "
            f"shape={self._shape}, dtype={self._dtype})"
        )


class HDF5Reader(FormatReader):
    """Reader for HDF5 (.hdf5, .h5, .nxs) files using h5py.

    HDF5 files may contain a hierarchy of groups and datasets. A single file
    can hold one or many frames of X-ray image data spread across multiple
    datasets. Use ``inspect()`` to explore the file structure before reading.
    """

    def inspect(self, path: str | Path) -> dict[str, Any]:
        """Return a summary of all groups and datasets without loading pixel data.

        Useful for understanding file structure before deciding which dataset to read.

        Args:
            path: Path to the HDF5 file.

        Returns:
            Dict with keys:
                ``"groups"``   - list of all group paths.
                ``"datasets"`` - dict mapping each dataset path to its shape, dtype, and attrs.
        """
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found or unsupported: {path}")

        groups: list[str] = []
        datasets: dict[str, dict[str, Any]] = {}

        def _visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group):
                groups.append(f"/{name}")
            elif isinstance(obj, h5py.Dataset):
                datasets[f"/{name}"] = {
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "attrs": dict(obj.attrs),
                }

        try:
            with h5py.File(str(path), "r") as f:
                groups.append("/")
                f.visititems(_visitor)
            return {"groups": groups, "datasets": datasets}
        except Exception as e:
            raise IOError(f"Error inspecting HDF5 file {path}: {e}")

    def read_header(self, path: str | Path, dataset_path: str | None = None) -> dict[str, Any]:
        """Read metadata from a dataset without loading pixel data.

        Args:
            path         : Path to the HDF5 file.
            dataset_path : Internal HDF5 path to the dataset (e.g. ``"/entry/data"``).
                           If ``None``, auto-selects the only dataset in the file or raises
                           if there are multiple.

        Returns:
            Dict with at least: ``{"shape": tuple, "dtype": str, "n_frames": int, "attrs": dict}``.
        """
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found or unsupported: {path}")

        dataset_path = self._resolve_dataset(path, dataset_path)

        try:
            with h5py.File(str(path), "r") as f:
                ds = f[dataset_path]
                shape = ds.shape
                dtype = str(ds.dtype)
                attrs = dict(ds.attrs)
                n_frames = shape[0] if len(shape) >= 3 else 1
            return {
                "shape": shape,
                "dtype": dtype,
                "n_frames": n_frames,
                "dataset_path": dataset_path,
                "attrs": attrs,
            }
        except (KeyError, IOError):
            raise
        except Exception as e:
            raise IOError(f"Error reading HDF5 header from {path}: {e}")

    def read(
        self,
        path: str | Path,
        dataset_path: str | None = None,
        start_frame: int = -1,
        end_frame: int = -1,
        dtype: DTypeLike = np.float32,
    ) -> NDArray[T]:
        """Read pixel data from an HDF5 dataset.

        Args:
            path         : Path to the HDF5 file.
            dataset_path : Internal HDF5 path to the dataset (e.g. ``"/entry/data"``).
                           If ``None``, auto-selects the only dataset in the file or raises
                           if there are multiple.
            start_frame  : First frame index (inclusive). Negative means read from the start.
                           Only applies when the dataset has 3 or more dimensions.
            end_frame    : Last frame index (exclusive). Negative means read until the end.

        Returns:
            2D array (Y, X) for single-frame datasets, or 3D array (T, Y, X) for stacks.
        """
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found or unsupported: {path}")

        dataset_path = self._resolve_dataset(path, dataset_path)

        try:
            with h5py.File(str(path), "r") as f:
                ds = f[dataset_path]
                shape = ds.shape

                if len(shape) < 2:
                    raise ValueError(
                        f"Dataset '{dataset_path}' has shape {shape}; expected at least 2D image data"
                    )

                if len(shape) == 2:
                    data = ds[()].astype(dtype)
                    logger.debug(f"Read single frame from '{dataset_path}' with shape {shape}")
                    return data

                n_frames = shape[0]
                start = 0 if start_frame < 0 else start_frame
                end = n_frames if end_frame < 0 else end_frame

                if start >= n_frames:
                    raise IndexError(
                        f"start_frame {start} is out of bounds for dataset with {n_frames} frames"
                    )
                if end > n_frames:
                    raise IndexError(
                        f"end_frame {end} is out of bounds for dataset with {n_frames} frames"
                    )

                data = ds[start:end].astype(dtype)

            if data.shape[0] == 1:
                return data[0]
            return data
        except (IndexError, ValueError, KeyError):
            raise
        except Exception as e:
            raise IOError(f"Error reading HDF5 file {path}: {e}")

    def lazy_read(
        self,
        path: str | Path,
        dataset_path: str | None = None,
        dtype: DTypeLike = np.float32,
    ) -> LazyHDF5Array:
        """Return a lazy array that loads data only when sliced.

        No pixel data is read until you index the result::

            arr = reader.lazy_read("scan.h5")
            frame = arr[0]          # loads one frame
            stack = arr[10:20]      # loads ten frames
            tile  = arr[0, :512, :512]  # loads a spatial tile of frame 0

        Args:
            path         : Path to the HDF5 file.
            dataset_path : Internal HDF5 path (e.g. ``"/entry/data"``).
                           Auto-detected when there is exactly one dataset.
            dtype        : NumPy dtype for the returned arrays (default float32).

        Returns:
            :class:`LazyHDF5Array` with ``.shape``, ``.dtype``, and numpy-style indexing.
        """
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found or unsupported: {path}")
        dataset_path = self._resolve_dataset(path, dataset_path)
        return LazyHDF5Array(path, dataset_path, dtype=dtype)

    @contextmanager
    def lazy_open(
        self,
        source: Any,
        dataset_path: str | None = None,
        dtype: DTypeLike = np.float32,
    ) -> Generator["LazyHDF5Array", None, None]:
        """Open an HDF5 dataset lazily from a local path or a seekable file-like object.

        Keeps h5py.File open for the duration of the with-block. Use this when
        source is a streaming file object (e.g. from backend.open_fileobj()) where
        the underlying connection must stay open while reading.

        Args:
            source       : A local path (str or Path) or any seekable binary
                           file-like object from backend.open_fileobj().
            dataset_path : Internal HDF5 path. Auto-detected when there is exactly
                           one dataset.
            dtype        : NumPy dtype for returned arrays (default float32).

        Usage with a remote backend::

            with backend.open_fileobj("scan.h5") as f:
                with reader.lazy_open(f) as arr:
                    frame = arr[0]

        Usage with a local file::

            with reader.lazy_open("/scratch/scan.h5") as arr:
                frame = arr[0]
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not self.can_read(path):
                raise FileNotFoundError(f"File not found or unsupported: {path}")
            with h5py.File(str(path), "r") as hf:
                resolved = _find_hdf5_dataset(hf, dataset_path)
                yield LazyHDF5Array._from_dataset(hf[resolved], dtype)
        else:
            with h5py.File(source, "r") as hf:
                resolved = _find_hdf5_dataset(hf, dataset_path)
                yield LazyHDF5Array._from_dataset(hf[resolved], dtype)

    @contextmanager
    def lazy_open_remote(
        self,
        backend: Any,
        path: str,
        dataset_path: str | None = None,
        dtype: DTypeLike = np.float32,
    ) -> Generator["LazyHDF5Array", None, None]:
        """Open a remote HDF5 dataset lazily via any connected StorageBackend.

        Shorthand for nesting backend.open_fileobj() and lazy_open() together.

        Usage::

            with backend:
                backend.connect(...)
                with reader.lazy_open_remote(backend, "scan.h5") as arr:
                    print(arr.shape)
                    frame = arr[0]

        Args:
            backend      : Any connected StorageBackend (S3Backend, LocalBackend, etc.).
            path         : Path relative to the backend root.
            dataset_path : Internal HDF5 path. Auto-detected when there is exactly one dataset.
            dtype        : NumPy dtype for returned arrays (default float32).
        """
        with backend.open_fileobj(path) as f:
            with self.lazy_open(f, dataset_path=dataset_path, dtype=dtype) as arr:
                yield arr

    def can_read(self, path: str | Path) -> bool:
        """Check if the file exists, has a recognised extension, and has valid HDF5 magic bytes."""
        path = Path(path)
        if not path.exists() or not path.is_file():
            return False
        if path.suffix.lower() not in _EXTENSIONS:
            return False
        try:
            with open(path, "rb") as f:
                return f.read(8) == _HDF5_MAGIC
        except OSError:
            return False

    def _resolve_dataset(self, path: Path, dataset_path: str | None) -> str:
        """Return ``dataset_path`` as-is, or auto-detect when it is ``None``."""
        if dataset_path is not None:
            return dataset_path

        info = self.inspect(path)
        datasets = list(info["datasets"].keys())

        if not datasets:
            raise ValueError(f"No datasets found in {path}")
        if len(datasets) == 1:
            return datasets[0]

        raise ValueError(
            f"Multiple datasets found in {path}; specify dataset_path. Available: {datasets}"
        )
