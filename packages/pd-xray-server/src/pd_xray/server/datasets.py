"""Datasets opened by the UI, with cached frames, pyramid levels and rendered tiles."""

import hashlib
import json
import math
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import CHANNEL, CancelToken, PdXrayError, Progress, frame_axes
from pd_xray.io import Dataset, open_dataset, open_sequence
from pd_xray.processing import Pipeline
from pd_xray.processing.pipeline import Reference
from pd_xray.server.cache import LRUCache
from pd_xray.server.colormaps import COLORMAPS
from pd_xray.server.references import average_files, reference_files
from pd_xray.server.render import ImageFormat, encode, frame_stats, halve, histogram, level_count, to_uint8

_KEEP_PIPELINES = 32  # registered preview pipelines kept; reference frames are shared through the cache


class NotFoundError(PdXrayError):
    """Raised for an unknown dataset or job id, or a tile outside the frame."""


def jsonable(value: Any) -> Any:
    """Convert metadata values (NumPy scalars/arrays, bytes, paths, …) to JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class OpenDataset:
    """A dataset opened through the server.

    Attributes:
        id: Server-assigned id used in URLs.
        path: File, or directory for a file sequence.
        dataset: The lazy dataset.
        options: How it was opened (`dataset`, `pattern`), so the UI can show and repeat it.
    """

    id: str
    path: Path
    dataset: Dataset
    options: dict[str, Any]

    @property
    def frame_axes(self) -> tuple[str, ...]:
        return frame_axes(self.dataset.axes)

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """Lengths of the frame axes (the slider ranges)."""
        return self.dataset.shape[: len(self.frame_axes)]

    @property
    def height_width(self) -> tuple[int, int]:
        n = len(self.frame_axes)
        return self.dataset.shape[n], self.dataset.shape[n + 1]


class DatasetStore:
    """Open datasets by id, plus a shared size-bounded cache for display data. Thread-safe."""

    def __init__(self, cache_bytes: int, tile_size: int, roots: Sequence[Path]) -> None:
        self.tile_size = tile_size
        self.roots = tuple(roots)
        self.cache = LRUCache(cache_bytes)
        self._open: dict[str, OpenDataset] = {}
        # Pipelines registered for previews: id -> (dataset id, resolved pipeline), most recent last.
        self._pipelines: OrderedDict[str, tuple[str, Pipeline]] = OrderedDict()
        self._lock = threading.Lock()

    def open(
        self, path: Path, dataset: str | None = None, pattern: str | None = None, axes: Sequence[str] | None = None
    ) -> OpenDataset:
        """Open a file, or a directory as a file sequence.

        Args:
            path: Resolved file or directory.
            dataset: HDF5 dataset path inside the file.
            pattern: Glob pattern for a directory, e.g. "*_img.tif". Defaults to "*".
            axes: Axis names overriding the detected ones.

        Raises:
            PdXrayError: If the file can't be opened as given (reasons in the message).
        """
        if path.is_dir():
            if dataset is not None:
                raise PdXrayError("'dataset' applies to HDF5 files, not to folders.")
            opened: Dataset = open_sequence(path, pattern or "*", axes)
            options: dict[str, Any] = {"pattern": pattern or "*"}
        else:
            if pattern is not None:
                raise PdXrayError("'pattern' applies to folders (file sequences), not to single files.")
            options = {} if dataset is None else {"dataset": dataset}
            try:
                opened = open_dataset(path, axes, **options)
            except TypeError:
                raise PdXrayError(f"'dataset' applies to HDF5 files, not to '{path.name}'.") from None
        entry = OpenDataset(uuid.uuid4().hex[:12], path, opened, options)
        with self._lock:
            self._open[entry.id] = entry
        return entry

    def get(self, dataset_id: str) -> OpenDataset:
        """The open dataset with this id.

        Raises:
            NotFoundError: If no such dataset is open.
        """
        with self._lock:
            entry = self._open.get(dataset_id)
        if entry is None:
            raise NotFoundError(f"No open dataset with id '{dataset_id}'. It may have been closed.")
        return entry

    def list(self) -> list[OpenDataset]:
        with self._lock:
            return list(self._open.values())

    def close(self, dataset_id: str) -> None:
        """Close a dataset and drop its cached data.

        Raises:
            NotFoundError: If no such dataset is open.
        """
        with self._lock:
            entry = self._open.pop(dataset_id, None)
        if entry is None:
            raise NotFoundError(f"No open dataset with id '{dataset_id}'.")
        self.cache.discard(lambda key: isinstance(key, tuple) and key[1] == dataset_id)
        with self._lock:
            for pipeline_id in [p for p, (owner, _) in self._pipelines.items() if owner == dataset_id]:
                del self._pipelines[pipeline_id]
        entry.dataset.close()

    def close_all(self) -> None:
        for entry in self.list():
            self.close(entry.id)

    def info(self, entry: OpenDataset) -> dict[str, Any]:
        """JSON description of an open dataset for the UI."""
        ds = entry.dataset
        height, width = entry.height_width
        return {
            "id": entry.id,
            "path": str(entry.path),
            "name": entry.path.name,
            "options": entry.options,
            "shape": list(ds.shape),
            "dtype": str(ds.dtype),
            "axes": list(ds.axes),
            "frame_axes": list(entry.frame_axes),
            "frame_shape": list(entry.frame_shape),
            "height": height,
            "width": width,
            "channels": ds.shape[-1] if ds.axes[-1] == CHANNEL else 1,
            "levels": level_count(height, width, self.tile_size),
            "tile_size": self.tile_size,
            "nbytes": ds.nbytes,
            "metadata": jsonable(ds.metadata),
            "warnings": list(ds.warnings),
        }

    @staticmethod
    def parse_frame(entry: OpenDataset, frame: str) -> tuple[int, ...]:
        """Frame index from its URL form: comma-separated indices, one per frame axis ("12", "3,40"); "-" if none.

        Raises:
            PdXrayError: If the index is malformed or out of range.
        """
        shape = entry.frame_shape
        try:
            index = () if frame == "-" else tuple(int(part) for part in frame.split(","))
        except ValueError:
            raise PdXrayError(f"Frame index '{frame}' must be comma-separated integers.") from None
        if len(index) != len(shape) or any(not 0 <= i < n for i, n in zip(index, shape, strict=True)):
            axes = ", ".join(f"{a} < {n}" for a, n in zip(entry.frame_axes, shape, strict=True)) or "none ('-')"
            raise PdXrayError(f"Frame index '{frame}' doesn't fit the frame axes ({axes}).")
        return index

    def frame(self, entry: OpenDataset, index: tuple[int, ...], pipeline: str | None = None) -> NDArray[Any]:
        """One frame, (y, x) or (y, x, channel): native dtype, or float32 processed by a registered pipeline. Cached
        and read-only."""
        if pipeline is not None:
            return self.process(entry, index, pipeline)[0]

        def read() -> NDArray[Any]:
            array = np.asarray(entry.dataset[index] if index else entry.dataset[...])
            array.flags.writeable = False
            return array

        result: NDArray[Any] = self.cache.get_or_compute(("frame", entry.id, index), read)
        return result

    def register_pipeline(self, entry: OpenDataset, steps: Sequence[Any], cancel: CancelToken | None = None) -> str:
        """Validate pipeline steps, load their references, and return an id for previews of this dataset.

        Args:
            entry: Dataset; relative reference paths are relative to its folder.
            steps: Steps as in a pipeline file: [{"op", "params", "enabled"}].
            cancel: Optional cancel token for loading references.

        Raises:
            PdXrayError: For invalid steps or references (the message names the step).
        """
        pipeline = Pipeline.from_dict({"pd_xray_pipeline": 1, "steps": steps})
        text = json.dumps([entry.id, pipeline.to_dict()], sort_keys=True)
        pipeline_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        with self._lock:
            if pipeline_id in self._pipelines:
                self._pipelines.move_to_end(pipeline_id)
                return pipeline_id
        base = entry.path if entry.path.is_dir() else entry.path.parent
        resolved = pipeline.resolve(lambda reference: self.reference(reference, base, cancel))
        with self._lock:
            self._pipelines[pipeline_id] = (entry.id, resolved)
            while len(self._pipelines) > _KEEP_PIPELINES:
                self._pipelines.popitem(last=False)
        return pipeline_id

    def reference(self, reference: Reference, base: Path, cancel: CancelToken | None = None) -> NDArray[Any]:
        """Averaged reference frame (see `references.py`). Cached until a file changes."""
        files = reference_files(reference, base, self.roots)
        key = ("reference", tuple(files), tuple(f.stat().st_mtime_ns for f in files))

        def load() -> NDArray[Any]:
            array = average_files(files, cancel)
            array.flags.writeable = False
            return array

        result: NDArray[Any] = self.cache.get_or_compute(key, load)
        return result

    def process(
        self, entry: OpenDataset, index: tuple[int, ...], pipeline: str, cancel: CancelToken | None = None
    ) -> tuple[NDArray[Any], tuple[tuple[str, float], ...]]:
        """A frame processed by a registered pipeline, with the seconds each step took. Cached.

        Raises:
            NotFoundError: If the pipeline id is unknown (expired).
            OperationCancelledError: If `cancel` is cancelled between steps.
        """
        with self._lock:
            registered = self._pipelines.get(pipeline)
        if registered is None or registered[0] != entry.id:
            raise NotFoundError("This preview expired; show the processed view again to recompute it.")
        resolved = registered[1]

        def compute() -> tuple[NDArray[Any], tuple[tuple[str, float], ...]]:
            timings: list[tuple[str, float]] = []
            step, started = "", time.perf_counter()

            def progress(update: Progress) -> None:  # called before each step and at the end
                nonlocal step, started
                now = time.perf_counter()
                if step:
                    timings.append((step, now - started))
                step, started = update.message, now

            array = resolved(self.frame(entry, index), progress, cancel)
            array.flags.writeable = False
            return array, tuple(timings)

        result: tuple[NDArray[Any], tuple[tuple[str, float], ...]] = self.cache.get_or_compute(
            ("processed", entry.id, index, pipeline), compute
        )
        return result

    def preview(
        self, entry: OpenDataset, index: tuple[int, ...], steps: Sequence[Any], cancel: CancelToken | None = None
    ) -> dict[str, Any]:
        """Process one frame for display and describe the result, so the UI can request its tiles.

        Returns:
            {"pipeline" (id for tile/stats requests), "height", "width", "levels", "stats", "timings": [{"op",
            "seconds"}]}. Cached results come back with the timings of their first computation.
        """
        pipeline = self.register_pipeline(entry, steps, cancel)
        array, timings = self.process(entry, index, pipeline, cancel)
        return {
            "pipeline": pipeline,
            "height": array.shape[0],
            "width": array.shape[1],
            "levels": level_count(array.shape[0], array.shape[1], self.tile_size),
            "stats": self.stats(entry, index, pipeline),
            "timings": [{"op": op, "seconds": seconds} for op, seconds in timings],
        }

    def level(
        self, entry: OpenDataset, index: tuple[int, ...], level: int, pipeline: str | None = None
    ) -> NDArray[Any]:
        """Pyramid level of a frame: as `frame` at level 0, float32 above (each the halved level below). Cached and
        read-only."""
        if level == 0:
            return self.frame(entry, index, pipeline)

        def shrink() -> NDArray[Any]:
            array = halve(self.level(entry, index, level - 1, pipeline))
            array.flags.writeable = False
            return array

        result: NDArray[Any] = self.cache.get_or_compute(("level", entry.id, index, pipeline, level), shrink)
        return result

    def stats(self, entry: OpenDataset, index: tuple[int, ...], pipeline: str | None = None) -> dict[str, float]:
        """Display statistics of a full-resolution frame (see `render.frame_stats`). Cached."""
        result: dict[str, float] = self.cache.get_or_compute(
            ("stats", entry.id, index, pipeline), lambda: frame_stats(self.frame(entry, index, pipeline))
        )
        return result

    def histogram(
        self,
        entry: OpenDataset,
        index: tuple[int, ...],
        bins: int = 256,
        low: float | None = None,
        high: float | None = None,
        pipeline: str | None = None,
    ) -> dict[str, Any]:
        """Histogram of a full-resolution frame over [low, high] (default: the frame's min and max). Cached.

        Returns:
            {"low", "high", "counts"}: `bins` pixel counts in equal bins.

        Raises:
            PdXrayError: For fewer than 2 or more than 4096 bins.
        """
        if not 2 <= bins <= 4096:
            raise PdXrayError(f"Histogram bins must be 2 to 4096, got {bins}.")
        if low is None or high is None:
            stats = self.stats(entry, index, pipeline)
            low = stats["min"] if low is None else low
            high = stats["max"] if high is None else high
        lo, hi = low, high
        counts: list[int] = self.cache.get_or_compute(
            ("histogram", entry.id, index, pipeline, bins, lo, hi),
            lambda: histogram(self.frame(entry, index, pipeline), bins, lo, hi),
        )
        return {"low": lo, "high": hi, "counts": counts}

    def tile(
        self,
        entry: OpenDataset,
        index: tuple[int, ...],
        level: int,
        column: int,
        row: int,
        fmt: ImageFormat,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "grey",
        pipeline: str | None = None,
        tone: str | None = None,
    ) -> bytes:
        """Encoded 8-bit tile. Contrast limits default to the frame's automatic ones (`stats` low/high). Cached.

        `cmap` names a table in `COLORMAPS`. It applies to single-channel frames; colour frames are shown as they are.
        `pipeline` (from `preview`) shows the processed frame instead of the original. `tone` (from
        `render.parse_tone`) adjusts tones between vmin and vmax.

        Raises:
            NotFoundError: If the level or tile lies outside the frame, or the pipeline expired.
            PdXrayError: For an unknown colour map.
        """
        if cmap not in COLORMAPS:
            raise PdXrayError(f"Unknown colour map '{cmap}'. Known: {', '.join(COLORMAPS)}.")
        height, width = entry.height_width if pipeline is None else self.frame(entry, index, pipeline).shape[:2]
        levels = level_count(height, width, self.tile_size)
        if not 0 <= level < levels:
            raise NotFoundError(f"Level {level} doesn't exist; levels are 0 to {levels - 1}.")
        size, factor = self.tile_size, 2**level
        if not (0 <= row < math.ceil(height / factor / size) and 0 <= column < math.ceil(width / factor / size)):
            raise NotFoundError(f"Tile ({column}, {row}) is outside level {level}.")
        if vmin is None or vmax is None:
            stats = self.stats(entry, index, pipeline)
            vmin = stats["low"] if vmin is None else vmin
            vmax = stats["high"] if vmax is None else vmax

        def render() -> bytes:
            data = self.level(entry, index, level, pipeline)
            image = to_uint8(data[row * size : (row + 1) * size, column * size : (column + 1) * size], vmin, vmax, tone)
            if image.ndim == 2 and cmap != "grey":
                image = COLORMAPS[cmap][image]
            return encode(image, fmt)

        key = ("tile", entry.id, index, pipeline, level, column, row, fmt, vmin, vmax, cmap, tone)
        result: bytes = self.cache.get_or_compute(key, render)
        return result
