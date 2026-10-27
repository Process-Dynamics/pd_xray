"""Pipelines: an ordered list of operation steps, saved as versioned YAML and applied frame by frame.

File format (version 1):

    pd_xray_pipeline: 1
    steps:
      - op: flat_field
        params: {dark: scan/dar, flat: [scan/flat_1.tif, scan/flat_2.tif]}
      - op: gaussian
        params: {sigma: 2.0}
        enabled: false
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from pd_xray.core import CancelToken, PdXrayError, Progress, ProgressCallback, track
from pd_xray.processing.operations import OPERATIONS, Frame

VERSION = 1
_VERSION_KEY = "pd_xray_pipeline"

Reference = str | tuple[str, ...]
"""An unresolved reference parameter: one string, or several (stored as a list in the file, a tuple here)."""


def _reference(value: Any) -> Reference | None:
    """The hashable reference in a parameter value, or None if it holds no unresolved reference."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return tuple(value)
    return None


@dataclass(frozen=True)
class Step:
    """One operation with its parameters. Disabled steps are kept in the file but skipped."""

    op: str
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True


class Pipeline:
    """Validated, immutable sequence of steps. Call it on a frame to process it.

    Raises:
        PdXrayError: On construction, for unknown operations or invalid parameters.
    """

    def __init__(self, steps: Iterable[Step] = ()) -> None:
        checked = []
        for index, step in enumerate(steps):
            operation = OPERATIONS.get(step.op)
            if operation is None:
                raise PdXrayError(f"Step {index + 1}: unknown operation '{step.op}'. Known: {sorted(OPERATIONS)}.")
            try:
                checked.append(Step(step.op, operation.validate(step.params), step.enabled))
            except PdXrayError as error:
                raise PdXrayError(f"Step {index + 1} ({step.op}): {error}") from None
        self.steps: tuple[Step, ...] = tuple(checked)

    def __call__(
        self, frame: NDArray[Any], progress: ProgressCallback | None = None, cancel: CancelToken | None = None
    ) -> Frame:
        """Process one frame (y, x) or (y, x, channel). Returns a new float32 frame.

        Args:
            frame: Input frame, any numeric dtype.
            progress: Optional callback, called before each enabled step (message = operation name) and at the end.
            cancel: Optional cancel token, checked before each step.

        Raises:
            PdXrayError: If references are unresolved (call `resolve` first).
            OperationCancelledError: If `cancel` is cancelled.
        """
        if self.references:
            raise PdXrayError(f"Unresolved references {sorted(self.references)}; call Pipeline.resolve() first.")
        result: Frame = np.array(frame, dtype=np.float32)
        enabled = [step for step in self.steps if step.enabled]
        for done, step in enumerate(enabled):
            if cancel is not None:
                cancel.raise_if_cancelled()
            if progress is not None:
                progress(Progress(done, len(enabled), step.op))
            result = OPERATIONS[step.op].function(result, **step.params)
        if progress is not None:
            progress(Progress(len(enabled), len(enabled), ""))
        return result

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Pipeline) and self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"Pipeline({list(self.steps)})"

    @property
    def references(self) -> set[Reference]:
        """References (e.g. dark/flat locations) that still need `resolve()`. Lists appear as tuples."""
        return {
            reference
            for step in self.steps
            for param in OPERATIONS[step.op].params
            if param.kind == "reference" and (reference := _reference(step.params[param.name])) is not None
        }

    def resolve(self, load: Callable[[Reference], NDArray[Any]]) -> "Pipeline":
        """A copy with references replaced by the frames `load(reference)` returns (once per reference)."""
        loaded = {reference: np.asarray(load(reference), dtype=np.float32) for reference in self.references}

        def swap(step: Step) -> Step:
            params = dict(step.params)
            for param in OPERATIONS[step.op].params:
                if param.kind == "reference" and (reference := _reference(params[param.name])) is not None:
                    params[param.name] = loaded[reference]
            return Step(step.op, params, step.enabled)

        return Pipeline(swap(step) for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Plain data for saving.

        Raises:
            PdXrayError: If a reference parameter holds an array instead of its reference string.
        """
        steps = []
        for step in self.steps:
            if any(isinstance(v, np.ndarray) for v in step.params.values()):
                raise PdXrayError(f"Step '{step.op}' holds a loaded array; save the unresolved pipeline instead.")
            steps.append({"op": step.op, "params": dict(step.params), "enabled": step.enabled})
        return {_VERSION_KEY: VERSION, "steps": steps}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Pipeline":
        """Pipeline from `to_dict()` output.

        Raises:
            PdXrayError: For a missing or newer version, or malformed steps.
        """
        version = data.get(_VERSION_KEY) if isinstance(data, Mapping) else None
        if version != VERSION:
            raise PdXrayError(f"Not a version {VERSION} pd_xray pipeline (found {_VERSION_KEY}: {version!r}).")
        try:
            return cls(Step(s["op"], s.get("params") or {}, s.get("enabled", True)) for s in data.get("steps") or [])
        except (KeyError, TypeError, AttributeError) as error:
            raise PdXrayError(f"Malformed pipeline step: {error!r}.") from None

    def to_yaml(self) -> str:
        text: str = yaml.safe_dump(self.to_dict(), sort_keys=False)
        return text

    @classmethod
    def from_yaml(cls, text: str) -> "Pipeline":
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise PdXrayError(f"Invalid pipeline YAML: {error}") from None
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))


def apply_pipeline(
    pipeline: Pipeline,
    frames: Iterable[NDArray[Any]],
    total: int | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> Iterator[Frame]:
    """Lazily process frames one at a time (constant memory). Pass the result to `pd_xray.io.write_frames`.

    To process a whole dataset into a file, `pd_xray.io.convert(pd_xray.io.map_frames(dataset, pipeline), path)` is
    simpler, and processes several frames at once with `workers`.

    Args:
        pipeline: Resolved pipeline (no reference strings left).
        frames: Input frames, e.g. `pd_xray.io.iter_frames(dataset)`.
        total: Number of frames, for progress (inferred if `frames` has a length).
        progress: Optional progress callback.
        cancel: Optional cancel token, checked before each frame.

    Yields:
        float32 frames. Their shape can differ from the input (crop, rotate), so get the output shape from the first
        frame or from `pipeline(first_input_frame)`.

    Raises:
        PdXrayError: If references are unresolved (raised immediately, before any frame is read).
    """
    if pipeline.references:
        raise PdXrayError(f"Unresolved references {sorted(pipeline.references)}; call Pipeline.resolve() first.")
    return (pipeline(frame) for frame in track(frames, total, "Processing", progress, cancel))
