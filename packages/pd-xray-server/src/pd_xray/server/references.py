"""Reference frames for pipelines (e.g. darks and flats), limited to the allowed folders, and their saved form."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from pd_xray.core import CancelToken
from pd_xray.io import reference_files as named_files
from pd_xray.io import reference_frames
from pd_xray.processing import OPERATIONS, Pipeline, Step, average_frames
from pd_xray.processing.pipeline import Reference
from pd_xray.server.paths import resolve_path


def reference_files(reference: Reference, base: Path, roots: Sequence[Path]) -> list[Path]:
    """The files a reference names (see `pd_xray.io.reference_files`), each entry checked to lie inside the roots.

    Raises:
        AccessDeniedError: For a path outside the roots.
        PdXrayError: If a path doesn't exist or a folder has no openable files.
    """
    entries = (reference,) if isinstance(reference, str) else reference
    return named_files([str(resolve_path(base / entry, roots)) for entry in entries])  # absolute entries replace base


def average_files(files: Sequence[Path], cancel: CancelToken | None = None) -> NDArray[Any]:
    """Mean of every frame in `files` (single- or multi-frame files), as float32.

    Raises:
        PdXrayError: If a file can't be opened or frames differ in shape.
    """
    return average_frames(reference_frames([str(f) for f in files]), cancel=cancel)


def relative_references(pipeline: Pipeline, base: Path) -> Pipeline:
    """A copy with references inside `base` (the dataset's folder) made relative to it, so the saved pipeline also
    works on other scans with the same layout. Others stay absolute (D10)."""

    def relative(path: str) -> str:
        absolute = base / path  # an absolute `path` replaces base
        return absolute.relative_to(base).as_posix() if absolute.is_relative_to(base) else path

    def swap(step: Step) -> Step:
        params = dict(step.params)
        for param in OPERATIONS[step.op].params:
            value = params[param.name]
            if param.kind == "reference" and value is not None:
                params[param.name] = relative(value) if isinstance(value, str) else [relative(v) for v in value]
        return Step(step.op, params, step.enabled)

    return Pipeline(swap(step) for step in pipeline.steps)
