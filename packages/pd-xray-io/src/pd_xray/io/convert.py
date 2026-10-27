"""Convert a dataset to another format, frame by frame, keeping values exact whenever the target format allows."""

import logging
import math
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import CancelToken, PdXrayError, ProgressCallback, frame_axes, track
from pd_xray.io.dataset import Dataset, split_key
from pd_xray.io.mapped import MappedDataset
from pd_xray.io.writers import frame_layout, supports, write_frames

logger = logging.getLogger(__name__)


def iter_frames(dataset: Dataset, first: Sequence[int] | None = None) -> Iterator[NDArray[Any]]:
    """Yield 2D (or 2D + channel) frames in C order over the frame axes, one read at a time.

    Args:
        dataset: Source dataset.
        first: Indices along the first axis to include. All by default. Ignored for single-frame datasets.
    """
    return dataset.iter_frames(first)


def compute_value_range(
    dataset: Dataset,
    first: Sequence[int] | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> tuple[float, float]:
    """Min and max over the selected frames, ignoring NaN. Reads every selected frame once."""
    n_frame_axes = len(frame_axes(dataset.axes))
    total = (len(first) if first is not None else dataset.shape[0]) if n_frame_axes else 1
    total *= math.prod(dataset.shape[1:n_frame_axes])
    lo, hi = math.inf, -math.inf
    for frame in track(iter_frames(dataset, first), total, "Computing value range", progress, cancel):
        if frame.dtype.kind == "f" and np.isnan(frame).all():
            continue
        lo, hi = min(lo, float(np.nanmin(frame))), max(hi, float(np.nanmax(frame)))
    return (lo, hi) if lo <= hi else (0.0, 1.0)


def convert(
    source: Dataset,
    destination: str | Path,
    frames: Any = None,
    value_range: tuple[float, float] | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
    dtype: Any = None,
    **options: Any,
) -> Path:
    """Write `source` (or some of its frames) to `destination`; the extension picks the format.

    Values are copied exactly when the format can store the dtype (TIFF/HDF5 always; PNG for uint8/uint16).
    Otherwise (e.g. float32 -> PNG, uint16 -> JPG) values are scaled linearly to the format's integer range using one
    range for all frames, so the histogram's shape is kept, and the mapping is stored in
    `metadata["value_scaling"]`: original ≈ offset + pixel * scale. JPG is lossy in any case. `dtype` chooses the
    output type instead: floats are exact casts, integers other than the source's are scaled as above.

    Args:
        source: Dataset to convert.
        destination: Output path (.tif/.tiff, .h5/.hdf5, .png, .jpg/.jpeg). PNG/JPG write one file per frame.
        frames: Index along the first axis, e.g. `slice(0, 100)`, `[0, 5, 9]` or `7`. All frames by default.
        value_range: (min, max) for scaling. Values outside are clipped. By default the data's own min/max, which
            needs an extra pass over the frames.
        progress: Optional progress callback.
        cancel: Optional cancel token.
        dtype: Output dtype, e.g. "float32" or "uint16". By default the source's, if the format can store it.
        **options: Writer options, e.g. `compression="zlib"`, `dataset="data"`, `quality=95`.

    Returns:
        The destination path.

    Raises:
        PdXrayError: If the format can't store the frames (or the chosen `dtype`).
    """
    destination = Path(destination)
    n_frame_axes = len(frame_axes(source.axes))
    metadata = dict(source.metadata)
    if "missing_frames" in metadata:  # indices refer to the source, and the frames are written as fill values
        metadata["source_missing_frames"] = metadata.pop("missing_frames")
    first: list[int] | None = None
    shape = source.shape
    if n_frame_axes and frames is not None:
        selected, _ = split_key(frames, source.shape[0])
        first = [selected] if isinstance(selected, int) else list(selected)
        shape = (len(first), *source.shape[1:])
        metadata["source_frames"] = first

    frame_shape, n_frames = frame_layout(shape, source.axes)
    if dtype is not None:
        dtype = np.dtype(dtype)
        if not supports(destination, dtype, frame_shape):
            raise PdXrayError(f"{destination.suffix} can't store {frame_shape} {dtype} frames.")
    elif supports(destination, source.dtype, frame_shape):
        dtype = source.dtype
    else:
        dtype = next(
            (np.dtype(t) for t in (np.uint16, np.uint8) if supports(destination, np.dtype(t), frame_shape)), None
        )
        if dtype is None:
            raise PdXrayError(f"{destination.suffix} can't store frames of shape {frame_shape}.")
    scaling: tuple[float, float, int] | None = None
    if dtype != source.dtype and dtype.kind in "ui":
        lo, hi = value_range or compute_value_range(source, first, progress, cancel)
        maxval = int(np.iinfo(dtype).max)
        scale = (hi - lo) / maxval if hi > lo else 1.0
        scaling = (lo, scale, maxval)
        metadata["value_scaling"] = {
            "source_dtype": str(source.dtype),
            "value_range": [lo, hi],
            "offset": lo,
            "scale": scale,
            "formula": "original ≈ offset + pixel * scale",
        }
        logger.warning("Scaling %s to %s for %s: precision is reduced.", source.dtype, dtype, destination.name)
    if destination.suffix.lower() in (".jpg", ".jpeg"):
        logger.warning("JPEG is lossy: pixel values in %s won't match the source exactly.", destination.name)

    clipped = threading.Event()
    if scaling is not None:
        lo, scale, maxval = scaling

        def scaled(frame: NDArray[Any]) -> NDArray[Any]:
            values = frame.astype(np.float64)
            values -= lo
            values /= scale
            np.nan_to_num(values, copy=False, nan=0.0)
            if values.min() < -0.5 or values.max() > maxval + 0.5:
                clipped.set()
            np.clip(values, 0, maxval, out=values)
            return np.rint(values, out=values).astype(dtype)

        # Scaling takes ~0.1 s per 8 MP frame: do it in a mapped source's processing threads.
        source = MappedDataset(source, scaled, source.workers if isinstance(source, MappedDataset) else 1)

    def output_frames() -> Iterator[NDArray[Any]]:
        for frame in track(iter_frames(source, first), n_frames, f"Writing {destination.name}", progress, cancel):
            yield frame if frame.dtype == dtype else frame.astype(dtype)

    write_frames(destination, output_frames(), shape, dtype, source.axes, metadata, **options)
    if clipped.is_set():
        logger.warning("Values outside the value range were clipped in %s.", destination.name)
    return destination
