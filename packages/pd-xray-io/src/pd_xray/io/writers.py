"""Frame-by-frame writers: TIFF (multi-page), HDF5 and PNG/JPG (one file per frame).

Writers consume an iterable of frames in C order over the frame axes, so memory use is about one frame regardless of
dataset size. Output is written into a hidden staging folder next to it and moved into place when complete, so a failed
or cancelled write leaves nothing behind (and doesn't touch an existing file of the same name).
"""

import json
import math
import os
import shutil
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import PdXrayError, frame_axes, import_optional, validate_axes
from pd_xray.io.formats import UnsupportedFormatError
from pd_xray.io.image import METADATA_KEY

Frames = Iterable[NDArray[Any]]
STAGING_SUFFIX = ".pd-xray-writing"


def jsonable(metadata: dict[str, Any]) -> dict[str, Any]:
    """Metadata as JSON-safe values (unknown types become strings)."""
    result: dict[str, Any] = json.loads(json.dumps(metadata, default=str))
    return result


def frame_layout(shape: Sequence[int], axes: Sequence[str]) -> tuple[tuple[int, ...], int]:
    """(frame shape, number of frames) for an array with these axes."""
    n_frame_axes = len(frame_axes(validate_axes(axes, len(shape))))
    return tuple(shape[n_frame_axes:]), math.prod(shape[:n_frame_axes])


def _checked(frames: Frames, frame_shape: tuple[int, ...], dtype: np.dtype[Any], n: int) -> Iterator[NDArray[Any]]:
    count = 0
    for frame in frames:
        if count == n:
            raise PdXrayError(f"Got more than the expected {n} frames.")
        if frame.shape != frame_shape or frame.dtype != dtype:
            raise PdXrayError(f"Expected a {frame_shape} {dtype} frame, got {frame.shape} {frame.dtype}.")
        count += 1
        yield frame
    if count != n:
        raise PdXrayError(f"Got {count} frames, expected {n}.")


def _is_jpeg(path: Path) -> bool:
    return path.suffix.lower() in (".jpg", ".jpeg")


def _any_dtype(path: Path, dtype: np.dtype[Any], frame_shape: tuple[int, ...]) -> bool:
    return True


def _image_supports(path: Path, dtype: np.dtype[Any], frame_shape: tuple[int, ...]) -> bool:
    """PNG: uint8/uint16 grey, uint8 RGB/RGBA. JPG: uint8 grey/RGB."""
    if len(frame_shape) == 3:
        return dtype == np.uint8 and frame_shape[-1] in ((3,) if _is_jpeg(path) else (3, 4))
    return len(frame_shape) == 2 and (dtype == np.uint8 or (dtype == np.uint16 and not _is_jpeg(path)))


def _write_tiff(
    path: Path,
    frames: Frames,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    axes: tuple[str, ...],
    metadata: dict[str, Any],
    compression: str | None = None,
) -> None:
    """Multi-page TIFF as one series (BigTIFF above ~4 GB). Axes and metadata go in tifffile's JSON description.

    Uncompressed by default, so the output can be memory-mapped. Options: `compression` (e.g. "zlib").
    """
    tifffile = import_optional("tifffile", "tiff", "pd-xray-io")
    frame_shape, _ = frame_layout(shape, axes)
    colour = len(frame_shape) == 3 and frame_shape[-1] in (3, 4)
    nbytes = math.prod(shape) * dtype.itemsize
    with tifffile.TiffWriter(path, bigtiff=nbytes > 2**32 - 2**27) as tif:
        tif.write(
            iter(frames),
            shape=shape,
            dtype=dtype,
            compression=compression,
            photometric="rgb" if colour else "minisblack",
            metadata={**metadata, "axes": list(axes)},
        )


def _write_hdf5(
    path: Path,
    frames: Frames,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    axes: tuple[str, ...],
    metadata: dict[str, Any],
    dataset: str = "data",
    compression: str | None = None,
    compression_opts: Any = None,
) -> None:
    """One dataset, chunked per frame, with axes as dimension labels and metadata as attributes.

    Options: `dataset` (default "data"), `compression` (e.g. "gzip"; default none), `compression_opts`.
    """
    import_optional("hdf5plugin", "hdf5", "pd-xray-io")
    h5py = import_optional("h5py", "hdf5", "pd-xray-io")
    frame_shape, _ = frame_layout(shape, axes)
    frame_dims = shape[: len(shape) - len(frame_shape)]
    with h5py.File(path, "w") as file:
        data = file.create_dataset(
            dataset,
            shape,
            dtype,
            chunks=(1,) * len(frame_dims) + frame_shape,
            compression=compression,
            compression_opts=compression_opts,
        )
        for dim, name in zip(data.dims, axes, strict=True):
            dim.label = name
        for key, value in metadata.items():
            try:
                data.attrs[key] = value
            except (TypeError, ValueError):
                data.attrs[key] = json.dumps(value)
        for index, frame in enumerate(frames):
            data[np.unravel_index(index, frame_dims)] = frame


def _write_images(
    path: Path,
    frames: Frames,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    axes: tuple[str, ...],
    metadata: dict[str, Any],
    quality: int = 95,
) -> None:
    """PNG/JPG, one file per frame: `scan.png` -> `scan_00000.png`, … (just `scan.png` if one frame).

    Metadata (with the frame index) goes in a PNG text chunk / JPEG comment as JSON. Options: `quality` (JPG).
    """
    image = import_optional("PIL.Image", "png", "pd-xray-io")
    png = import_optional("PIL.PngImagePlugin", "png", "pd-xray-io")
    _, n_frames = frame_layout(shape, axes)
    width = max(5, len(str(n_frames - 1)))
    for index, frame in enumerate(frames):
        out = path if n_frames == 1 else path.with_name(f"{path.stem}_{index:0{width}d}{path.suffix}")
        text = json.dumps({**metadata, "frame_index": index})
        if _is_jpeg(path):
            image.fromarray(frame).save(out, quality=quality, comment=text)
        else:
            info = png.PngInfo()
            info.add_text(METADATA_KEY, text)
            image.fromarray(frame).save(out, pnginfo=info)


Supports = Callable[[Path, np.dtype[Any], tuple[int, ...]], bool]
_FORMATS: dict[str, tuple[Callable[..., None], Supports]] = {
    ".tif": (_write_tiff, _any_dtype),
    ".tiff": (_write_tiff, _any_dtype),
    ".h5": (_write_hdf5, _any_dtype),
    ".hdf5": (_write_hdf5, _any_dtype),
    ".png": (_write_images, _image_supports),
    ".jpg": (_write_images, _image_supports),
    ".jpeg": (_write_images, _image_supports),
}


def _format(path: Path) -> tuple[Callable[..., None], Supports]:
    entry = _FORMATS.get(path.suffix.lower())
    if entry is None:
        raise UnsupportedFormatError(f"Can't write '{path.name}'. Supported: {', '.join(sorted(_FORMATS))}.")
    return entry


def supports(path: str | Path, dtype: np.dtype[Any], frame_shape: tuple[int, ...]) -> bool:
    """Whether the format chosen by `path`'s extension stores frames of this shape and dtype exactly."""
    return _format(Path(path))[1](Path(path), np.dtype(dtype), tuple(frame_shape))


def write_frames(
    path: str | Path,
    frames: Frames,
    shape: Sequence[int],
    dtype: np.dtype[Any],
    axes: Sequence[str],
    metadata: dict[str, Any] | None = None,
    **options: Any,
) -> Path:
    """Write frames to a file whose format is chosen by extension (.tif/.tiff, .h5/.hdf5, .png, .jpg/.jpeg).

    Args:
        path: Output path. Parent directories are created. An existing file is replaced once writing succeeds.
        frames: Frames in C order over the frame axes, each of shape `shape[n_frame_axes:]` and `dtype`. Consumed
            lazily, so a generator keeps memory use at one frame.
        shape, dtype, axes: Of the whole output, e.g. (100, 512, 512), uint16, ("angle", "y", "x").
        metadata: Stored with the output.
        **options: Format options (see `_write_tiff`, `_write_hdf5`, `_write_images`).

    Returns:
        `path`.

    Raises:
        PdXrayError: If the format can't store the dtype, or frames don't match `shape`/`dtype`.
        OperationCancelledError: If the frames iterable is cancelled; nothing is written then.
    """
    path = Path(path)
    shape, dtype = tuple(shape), np.dtype(dtype)
    write, can_store = _format(path)
    frame_shape, n_frames = frame_layout(shape, axes)
    if not can_store(path, dtype, frame_shape):
        raise PdXrayError(f"{path.suffix} can't store {frame_shape} {dtype} frames; convert first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}{STAGING_SUFFIX}")
    shutil.rmtree(staging, ignore_errors=True)  # left by a crash
    staging.mkdir()
    try:
        checked = _checked(frames, frame_shape, dtype, n_frames)
        write(staging / path.name, checked, shape, dtype, tuple(axes), jsonable(metadata or {}), **options)
        for written in sorted(staging.iterdir()):
            os.replace(written, path.parent / written.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return path
