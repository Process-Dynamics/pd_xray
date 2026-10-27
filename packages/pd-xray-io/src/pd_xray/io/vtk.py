"""Legacy VTK reader (`.vtk`): binary STRUCTURED_POINTS volumes with one scalar field, e.g. phase-field simulation
output. NumPy only.

Not registered in `open_dataset` (so the web app doesn't list `.vtk` files): use `open_vtk` or `open_vtk_series`, and
`convert` to write HDF5.
"""

import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import TIME, PdXrayError, X, Y, Z
from pd_xray.io.dataset import FRAME, Dataset, split_key
from pd_xray.io.formats import FileStore
from pd_xray.io.sequence import find_files

_HEADER_BYTES = 4096
_TRAILING_BYTES = 64  # whitespace some writers add after the data; more than this means another field follows
_HEADER_KEYWORDS = {
    "DATASET",
    "DIMENSIONS",
    "SPACING",
    "ASPECT_RATIO",
    "ORIGIN",
    "POINT_DATA",
    "SCALARS",
    "LOOKUP_TABLE",
}
# VTK type name -> stored dtype. Legacy binary data is big-endian.
_TYPES = {
    "unsigned_char": ">u1",
    "char": ">i1",
    "unsigned_short": ">u2",
    "short": ">i2",
    "unsigned_int": ">u4",
    "int": ">i4",
    "unsigned_long": ">u8",
    "long": ">i8",
    "float": ">f4",
    "double": ">f8",
    **{f"vtktype{kind}{bits}": f">{kind[0]}{bits // 8}" for kind in ("int", "uint") for bits in (8, 16, 32, 64)},
    "vtktypefloat32": ">f4",
    "vtktypefloat64": ">f8",
}
_TIME_TITLE = re.compile(r"t\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*s")


@dataclass(frozen=True)
class _Header:
    shape: tuple[int, int, int]  # (nz, ny, nx)
    dtype: np.dtype[Any]  # as stored (big-endian)
    offset: int  # of the data
    title: str
    scalar_name: str
    spacing: list[float]  # (z, y, x)
    origin: list[float]  # (z, y, x)


def _read_header(file: BinaryIO, name: str) -> _Header:
    """Parse the text header and check the file size against it."""
    head = file.read(_HEADER_BYTES)
    size = file.seek(0, os.SEEK_END)
    if not head.startswith(b"# vtk DataFile"):
        raise PdXrayError(f"{name} isn't a legacy VTK file (no '# vtk DataFile' line).")
    lines: list[str] = []
    offset = 0
    while len(lines) < 3 or not lines[-1].upper().startswith("LOOKUP_TABLE"):
        end = head.find(b"\n", offset)
        if end < 0:
            raise PdXrayError(f"{name}: no LOOKUP_TABLE line in the first {_HEADER_BYTES} bytes of the header.")
        lines.append(head[offset:end].decode("latin-1").strip())
        offset = end + 1
    title, encoding = lines[1], lines[2].upper()
    if encoding != "BINARY":
        raise PdXrayError(f"{name} is {encoding or 'not'} encoded; only BINARY VTK files are supported.")
    fields = {line.split()[0].upper(): line.split()[1:] for line in lines[3:] if line}
    dataset = (fields.get("DATASET") or ["no DATASET"])[0]
    if dataset.upper() != "STRUCTURED_POINTS":
        raise PdXrayError(f"{name} holds {dataset}; only STRUCTURED_POINTS is supported.")
    unknown = sorted(set(fields) - _HEADER_KEYWORDS)
    if unknown:
        raise PdXrayError(f"{name}: unsupported header lines {unknown}; only one POINT_DATA SCALARS field works.")
    try:
        nx, ny, nz = (int(v) for v in fields["DIMENSIONS"])
        n_points = int(fields["POINT_DATA"][0])
        scalar_name, type_name, *components = fields["SCALARS"]
        spacing = [float(v) for v in fields.get("SPACING", fields.get("ASPECT_RATIO", ["1", "1", "1"]))][::-1]
        origin = [float(v) for v in fields.get("ORIGIN", ["0", "0", "0"])][::-1]
    except KeyError as error:
        raise PdXrayError(f"{name}: no {error.args[0]} line in the VTK header.") from error
    except ValueError as error:
        raise PdXrayError(f"{name}: malformed VTK header ({error}).") from error
    if len(spacing) != 3 or len(origin) != 3:
        raise PdXrayError(f"{name}: SPACING and ORIGIN need three values each.")
    if components not in ([], ["1"]):
        raise PdXrayError(f"{name}: field '{scalar_name}' has {components[0]} components; only 1 is supported.")
    if type_name.lower() not in _TYPES:
        raise PdXrayError(f"{name}: unsupported data type '{type_name}'.")
    if n_points != nx * ny * nz:
        raise PdXrayError(f"{name}: POINT_DATA {n_points} doesn't match DIMENSIONS {nx} {ny} {nz}.")
    dtype = np.dtype(_TYPES[type_name.lower()])
    expected = offset + n_points * dtype.itemsize
    if size < expected:
        raise PdXrayError(f"{name} is truncated: the header needs {expected} bytes, the file has {size}.")
    if size > expected + _TRAILING_BYTES:
        raise PdXrayError(
            f"{name} has {size - expected} bytes after the '{scalar_name}' data (another field or CELL_DATA?); "
            "only files with one field are supported."
        )
    return _Header((nz, ny, nx), dtype, offset, title, scalar_name, spacing, origin)


def _time(title: str) -> float | None:
    """Time in seconds from a title like "t = 0.45 s", else None."""
    match = _TIME_TITLE.fullmatch(title)
    return float(match.group(1)) if match else None


class VTKDataset(Dataset):
    """Lazy view of a legacy VTK volume, shape (nz, ny, nx), axes ("z", "y", "x") by default.

    Values come back in native byte order (VTK stores big-endian). Local files are memory-mapped; from a `storage`,
    z-slices are read on demand. Metadata: `scalar_name`, `vtk_title`, `spacing` and `origin` in (z, y, x) order
    (reversed from the header's x, y, z), and `time_s` if the title reads "t = <seconds> s".
    """

    def __init__(self, path: str | Path, axes: Sequence[str] | None = None, storage: FileStore | None = None) -> None:
        self.path = Path(path)
        self._handle = None if storage is None else storage.open(str(self.path))
        self._lock = threading.Lock()
        self._memmap: NDArray[Any] | None = None
        try:
            if self._handle is None:
                with self.path.open("rb") as file:
                    header = _read_header(file, self.path.name)
            else:
                header = _read_header(self._handle, self.path.name)
            metadata: dict[str, Any] = {
                "scalar_name": header.scalar_name,
                "vtk_title": header.title,
                "spacing": header.spacing,
                "origin": header.origin,
            }
            time = _time(header.title)
            if time is not None:
                metadata["time_s"] = time
            super().__init__(header.shape, header.dtype.newbyteorder("="), axes or (Z, Y, X), metadata)
            self._stored = header.dtype
            self._offset = header.offset
            if self._handle is None:
                self._memmap = np.memmap(self.path, header.dtype, "r", header.offset, header.shape)
        except Exception:
            self.close()
            raise

    def _slice(self, index: int) -> NDArray[Any]:
        assert self._handle is not None
        size = self.shape[1] * self.shape[2] * self.dtype.itemsize
        with self._lock:  # one file handle shared by all reads
            self._handle.seek(self._offset + index * size)
            data = self._handle.read(size)
        return np.frombuffer(data, self._stored).reshape(self.shape[1:]).astype(self.dtype)

    def __getitem__(self, key: Any) -> NDArray[Any]:
        if self._memmap is not None:
            return np.array(self._memmap[key], dtype=self.dtype)
        first, rest = split_key(key, self.shape[0])
        if isinstance(first, int):
            return np.asarray(self._slice(first)[rest])
        slices = [self._slice(i) for i in first]
        stack = np.stack(slices) if slices else np.empty((0, *self.shape[1:]), self.dtype)
        return stack[(slice(None),) + rest]

    def close(self) -> None:
        self._memmap = None
        if self._handle is not None:
            self._handle.close()


class VTKSeries(Dataset):
    """Several VTK volumes on one grid (e.g. time steps) as one dataset, shape (n_files, nz, ny, nx).

    Volume `i` is `files[i]`, read lazily. Axes default to ("time", "z", "y", "x") when every title gives a time,
    else ("frame", "z", "y", "x"). Metadata: `scalar_name`, `spacing`, `origin` (as in `VTKDataset`), `vtk_titles`,
    `source_files` (names) and `time_s` (one per file, if every title gives one).

    Attributes:
        files: The files, in order.
        volumes: One `VTKDataset` per file.
    """

    def __init__(
        self, files: Sequence[Path], axes: Sequence[str] | None = None, storage: FileStore | None = None
    ) -> None:
        if not files:
            raise PdXrayError("No VTK files to open as a series.")
        self.files = list(files)
        self.volumes: list[VTKDataset] = []
        try:
            self.volumes.extend(VTKDataset(path, storage=storage) for path in self.files)
            grids = [_grid(volume) for volume in self.volumes]
            for volume, grid in zip(self.volumes, grids, strict=True):
                if grid != grids[0]:
                    raise PdXrayError(
                        f"{volume.path.name} is {grid}, but {self.files[0].name} is {grids[0]}; a series needs "
                        "every file on the same grid."
                    )
            first = self.volumes[0]
            metadata: dict[str, Any] = {key: first.metadata[key] for key in ("scalar_name", "spacing", "origin")}
            metadata["vtk_titles"] = [volume.metadata["vtk_title"] for volume in self.volumes]
            metadata["source_files"] = [path.name for path in self.files]
            times = [volume.metadata.get("time_s") for volume in self.volumes]
            if None not in times:
                metadata["time_s"] = times
            default_axes = (TIME if "time_s" in metadata else FRAME, Z, Y, X)
            super().__init__((len(self.files), *first.shape), first.dtype, axes or default_axes, metadata)
            if "time_s" in metadata and any(b <= a for a, b in pairwise(metadata["time_s"])):
                self.warnings.append(f"Times {times} aren't increasing in file order.")
        except Exception:
            self.close()
            raise

    def __getitem__(self, key: Any) -> NDArray[Any]:
        first, rest = split_key(key, self.shape[0])
        if isinstance(first, int):
            return self.volumes[first][rest]
        parts = [self.volumes[i][rest] for i in first]
        if parts:
            return np.stack(parts)
        return np.empty((0, *self.shape[1:]), self.dtype)[(slice(None),) + rest]

    def close(self) -> None:
        for volume in self.volumes:
            volume.close()


def _grid(volume: VTKDataset) -> str:
    """What must match between files of a series, as text for error messages."""
    m = volume.metadata
    return f"{volume.shape} {volume.dtype} '{m['scalar_name']}' spacing {m['spacing']} origin {m['origin']}"


def open_vtk(path: str | Path, axes: Sequence[str] | None = None, storage: FileStore | None = None) -> VTKDataset:
    """Open a legacy VTK volume (binary STRUCTURED_POINTS, one scalar field) as a lazy (z, y, x) dataset.

    Args:
        path: Local file path, or a path in `storage`.
        axes: Axis names overriding ("z", "y", "x").
        storage: Read the file from here instead of the local disk (see `open_dataset`).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        PdXrayError: If the file isn't a supported VTK file (ASCII, other dataset types, several fields, truncated).

    Example:
        >>> with open_vtk("AlCu5_3D_V1_1.vtk") as volume:
        ...     convert(volume, "AlCu5_3D_V1_1.h5")  # (z, y, x) float32, one chunk per z-slice
    """
    return VTKDataset(path, axes, storage)


def open_vtk_series(
    source: str | Path | Sequence[str | Path],
    pattern: str = "*.vtk",
    axes: Sequence[str] | None = None,
    storage: FileStore | None = None,
) -> VTKSeries:
    """Open several VTK volumes on one grid (e.g. time steps) as one lazy (time, z, y, x) dataset.

    Args:
        source: A directory (files matching `pattern`, in natural order: _2 before _10) or a list of files (kept in
            the given order).
        pattern: Glob pattern within the directory.
        axes: Axis names. Defaults to ("time", "z", "y", "x"), or ("frame", …) if the titles don't give times.
        storage: Read the files from here instead of the local disk (see `open_dataset`); paths are in it.

    Raises:
        PdXrayError: If no file matches, a file isn't supported, or the files' grids differ.

    Example:
        >>> with open_vtk_series("VTKS_EQX_FRAC1") as series:
        ...     convert(series, "frac1.h5")  # (time, z, y, x), `time_s` as an attribute
    """
    files = find_files(source, pattern, storage) if isinstance(source, (str, Path)) else [Path(p) for p in source]
    return VTKSeries(files, axes, storage)
