"""Pipeline files (save/load in the allowed folders) and runs: a pipeline applied to a whole dataset, into a file.

A run writes its output and, next to it, `<output>.pipeline.yaml`: the pipeline (references relative to the dataset's
folder where possible) plus a `run` section recording the source, output settings, date and versions. The same
record goes into the output's metadata (`pd_xray_run`). `Pipeline.load` reads the YAML file and ignores `run`.
"""

import datetime
import math
import os
import time
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pd_xray.core import CancelToken, PdXrayError, Progress, ProgressCallback, frame_axes
from pd_xray.io import convert, map_frames, supports
from pd_xray.processing import Pipeline
from pd_xray.server.datasets import DatasetStore, OpenDataset
from pd_xray.server.references import relative_references

RUN_WORKERS = min(32, os.cpu_count() or 1)
"""Frames processed at once in a run (filters scale almost linearly with threads)."""
OUTPUT_DTYPES = ("float32", "uint16", "uint8")
PIPELINE_EXTENSIONS = (".yaml", ".yml")
SIDECAR_SUFFIX = ".pipeline.yaml"
_PACKAGES = ("pd-xray-core", "pd-xray-io", "pd-xray-processing", "pd-xray-server")


def dataset_folder(entry: OpenDataset) -> Path:
    """Folder that relative references start from: the sequence's folder, or the file's."""
    return entry.path if entry.path.is_dir() else entry.path.parent


def save_pipeline(path: Path, steps: Sequence[Any], base: Path | None, overwrite: bool = False) -> Path:
    """Save steps as a pipeline file; references inside `base` (the dataset's folder) are stored relative to it.

    Raises:
        PdXrayError: For invalid steps or a name not ending in .yaml/.yml.
        FileExistsError: If the file exists and not `overwrite`.
    """
    if path.suffix.lower() not in PIPELINE_EXTENSIONS:
        raise PdXrayError(f"Pipeline files end in .yaml or .yml, got '{path.name}'.")
    pipeline = Pipeline.from_dict({"pd_xray_pipeline": 1, "steps": list(steps)})
    if base is not None:
        pipeline = relative_references(pipeline, base)
    if path.exists() and not overwrite:
        raise FileExistsError(f"'{path.name}' already exists.")
    return pipeline.save(path)


def output_files(destination: Path) -> list[Path]:
    """Existing files a run to `destination` would replace (PNG/JPG write `name_00000.png`, …)."""
    found = [p for p in (destination, destination.with_name(destination.name + SIDECAR_SUFFIX)) if p.exists()]
    if destination.suffix.lower() in (".png", ".jpg", ".jpeg") and destination.parent.is_dir():
        found += sorted(destination.parent.glob(f"{destination.stem}_[0-9]*{destination.suffix}"))
    return found


def check_run(entry: OpenDataset, destination: Path, dtype: str, overwrite: bool) -> None:
    """Checks a run can start, so mistakes show at once rather than when the job reaches them.

    Raises:
        PdXrayError: For an unknown dtype or one the output format can't store.
        FileExistsError: If the output exists and not `overwrite`.
    """
    if dtype not in OUTPUT_DTYPES:
        raise PdXrayError(f"Output type must be one of {', '.join(OUTPUT_DTYPES)}, got '{dtype}'.")
    channels = entry.dataset.shape[len(frame_axes(entry.dataset.axes)) + 2 :]
    if not supports(destination, np.dtype(dtype), (1, 1, *channels)):  # frame size may change with crop; channels don't
        raise PdXrayError(f"{destination.suffix} files can't store {dtype} frames; choose another type or format.")
    existing = output_files(destination)
    if existing and not overwrite:
        raise FileExistsError(f"'{existing[0].name}' already exists. Choose another name, or overwrite.")


def _versions() -> dict[str, str]:
    found = {}
    for package in _PACKAGES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "unknown"
    return found


def run_pipeline(
    store: DatasetStore,
    entry: OpenDataset,
    steps: Sequence[Any],
    destination: Path,
    dtype: str = "float32",
    value_range: tuple[float, float] | None = None,
    frames: slice | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> dict[str, Any]:
    """Process a dataset (or some frames along its first axis) into `destination`; the extension picks the format.

    Args:
        store: For loading (cached) reference frames, limited to the allowed folders.
        entry: The open dataset.
        steps: Steps as in a pipeline file. Relative references start from the dataset's folder.
        destination: Output file, checked by `check_run`.
        dtype: "float32" (exact), or "uint16"/"uint8" scaled linearly from `value_range` (clipped outside).
        value_range: (min, max) of processed values for integer output. By default the processed data's min and max,
            which processes every frame twice.
        frames: Frames along the first axis. All by default.
        progress: Optional progress callback (one update per frame written).
        cancel: Optional cancel token. A cancelled run leaves no output (D8).

    Returns:
        {"destination", "pipeline_file", "frames", "seconds"}.
    """
    started = time.monotonic()
    pipeline = Pipeline.from_dict({"pd_xray_pipeline": 1, "steps": list(steps)})
    base = dataset_folder(entry)
    if progress is not None:
        progress(Progress(0, None, "Loading references"))
    resolved = pipeline.resolve(lambda reference: store.reference(reference, base, cancel))
    record = {
        "source": str(entry.path),
        **entry.options,
        "frames": None if frames is None else [frames.start, frames.stop, frames.step],
        "output": destination.name,
        "dtype": dtype,
        "value_range": None if value_range is None else list(value_range),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
    }
    saved = relative_references(pipeline, base).to_dict()
    if progress is not None:
        progress(Progress(0, None, "Processing the first frame"))
    processed = map_frames(entry.dataset, resolved, RUN_WORKERS, {"pd_xray_run": {**record, "pipeline": saved}})
    convert(processed, destination, frames, value_range, progress, cancel, dtype=dtype)
    count = math.prod(entry.frame_shape)  # 1 for a single image
    if frames is not None and entry.frame_shape:
        count = count // entry.frame_shape[0] * len(range(*frames.indices(entry.frame_shape[0])))
    sidecar = destination.with_name(destination.name + SIDECAR_SUFFIX)
    sidecar.write_text(yaml.safe_dump({**saved, "run": record}, sort_keys=False), encoding="utf-8")
    return {
        "destination": str(destination),
        "pipeline_file": str(sidecar),
        "frames": count,
        "seconds": time.monotonic() - started,
    }
