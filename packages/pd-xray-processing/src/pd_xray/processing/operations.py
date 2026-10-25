"""Frame operations with typed parameters, so the UI can build controls from the registry.

Every operation takes a float32 frame (y, x) or (y, x, channel) plus keyword parameters and returns a new float32
frame. Pipelines cast input frames to float32 before the first step.
"""

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.measure import block_reduce
from skimage.restoration import rolling_ball as _rolling_ball
from skimage.transform import resize

from pd_xray.core import CancelToken, PdXrayError, ProgressCallback, track

Frame = NDArray[np.float32]
ParamKind = Literal["float", "int", "bool", "choice", "reference"]


@dataclass(frozen=True)
class Param:
    """One operation parameter.

    `kind="reference"` names a frame loaded by the caller (e.g. averaged dark frames): pipeline files store it as a
    string, and `Pipeline.resolve()` swaps in the array before running.
    """

    name: str
    kind: ParamKind
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    optional: bool = False
    description: str = ""

    def validate(self, value: Any) -> Any:
        """Return `value` coerced to this parameter's type.

        Raises:
            PdXrayError: If the value has the wrong type or is out of range.
        """
        if value is None:
            if self.optional:
                return None
            raise PdXrayError(f"Parameter '{self.name}' is required.")
        if self.kind == "reference":
            if isinstance(value, (str, np.ndarray)):
                return value
            raise PdXrayError(f"Parameter '{self.name}' must be a reference string or an array, got {value!r}.")
        if self.kind == "choice":
            if value not in self.choices:
                raise PdXrayError(f"Parameter '{self.name}' must be one of {list(self.choices)}, got {value!r}.")
            return value
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise PdXrayError(f"Parameter '{self.name}' must be true or false, got {value!r}.")
            return value
        number_types = (int, np.integer) if self.kind == "int" else (int, float, np.integer, np.floating)
        if isinstance(value, bool) or not isinstance(value, number_types):
            raise PdXrayError(f"Parameter '{self.name}' must be {'an integer' if self.kind == 'int' else 'a number'}.")
        number = int(value) if self.kind == "int" else float(value)
        if not math.isfinite(number):
            raise PdXrayError(f"Parameter '{self.name}' must be finite, got {value!r}.")
        if (self.minimum is not None and number < self.minimum) or (self.maximum is not None and number > self.maximum):
            raise PdXrayError(f"Parameter '{self.name}' must be in [{self.minimum}, {self.maximum}], got {value!r}.")
        return number


@dataclass(frozen=True)
class Operation:
    """A named frame function and its parameters."""

    name: str
    title: str
    description: str
    params: tuple[Param, ...]
    function: Callable[..., Frame]

    def validate(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Parameters with defaults filled in and values checked.

        Raises:
            PdXrayError: For unknown parameters or invalid values.
        """
        known = {p.name for p in self.params}
        unknown = set(params) - known
        if unknown:
            raise PdXrayError(f"Operation '{self.name}' has no parameter(s) {sorted(unknown)}. Known: {sorted(known)}.")
        return {p.name: p.validate(params.get(p.name, p.default)) for p in self.params}

    def __call__(self, frame: Frame, **params: Any) -> Frame:
        return self.function(frame, **self.validate(params))


def _spatial(frame: Frame, value: float) -> tuple[float, ...]:
    """Per-axis filter size that leaves the channel axis alone."""
    return (value, value, 0) if frame.ndim == 3 else (value, value)


def invert(frame: Frame, low: float, high: float) -> Frame:
    return np.asarray(high + low - frame, dtype=np.float32)


def gaussian(frame: Frame, sigma: float) -> Frame:
    return np.asarray(ndimage.gaussian_filter(frame, _spatial(frame, sigma)), dtype=np.float32)


def median(frame: Frame, size: int) -> Frame:
    size_per_axis = tuple(int(s) or 1 for s in _spatial(frame, size))
    return np.asarray(ndimage.median_filter(frame, size_per_axis), dtype=np.float32)


def normalise(frame: Frame, low_percentile: float, high_percentile: float) -> Frame:
    if low_percentile >= high_percentile:
        raise PdXrayError("low_percentile must be below high_percentile.")
    low, high = np.nanpercentile(frame, [low_percentile, high_percentile])
    if high == low:
        return np.zeros_like(frame)
    return np.asarray((frame - low) / (high - low), dtype=np.float32)


def clip(frame: Frame, low: float | None, high: float | None) -> Frame:
    if low is not None and high is not None and low > high:
        raise PdXrayError("low must not be above high.")
    if low is None and high is None:
        return frame.copy()
    return np.asarray(np.clip(frame, low, high), dtype=np.float32)


def threshold(frame: Frame, value: float, keep: str) -> Frame:
    mask = frame >= value if keep == "above" else frame < value
    return mask.astype(np.float32)


def crop(frame: Frame, top: int, left: int, height: int, width: int) -> Frame:
    bottom = frame.shape[0] if height == 0 else top + height
    right = frame.shape[1] if width == 0 else left + width
    if bottom > frame.shape[0] or right > frame.shape[1] or top >= bottom or left >= right:
        raise PdXrayError(f"Crop rows {top}:{bottom}, columns {left}:{right} is outside the {frame.shape[:2]} frame.")
    return frame[top:bottom, left:right].copy()


def rotate(frame: Frame, angle: str) -> Frame:
    return np.ascontiguousarray(np.rot90(frame, k=-int(angle) // 90, axes=(0, 1)))


def flip(frame: Frame, direction: str) -> Frame:
    return np.ascontiguousarray(np.flip(frame, axis=1 if direction == "horizontal" else 0))


def _rolling_ball_background(image: Frame, radius: float) -> Frame:
    """Like ImageJ, shrink the image (block minimum) for large radii, then interpolate back. Much faster, and the
    background is smooth anyway."""
    shrink = 1 if radius <= 10 else 2 if radius <= 30 else 4 if radius <= 100 else 8
    if shrink == 1:
        return np.asarray(_rolling_ball(image, radius=radius), dtype=np.float32)
    small = block_reduce(image, (shrink, shrink), np.min, cval=np.finfo(np.float32).max)
    background = _rolling_ball(small, radius=radius / shrink)
    return np.asarray(resize(background, image.shape, order=1, preserve_range=True), dtype=np.float32)


def rolling_ball(frame: Frame, radius: float, light_background: bool) -> Frame:
    planes = [frame] if frame.ndim == 2 else [frame[..., c] for c in range(frame.shape[-1])]
    result = []
    for plane in planes:
        if light_background:
            top = float(np.nanmax(plane))
            background = top - _rolling_ball_background(top - plane, radius)
        else:
            background = _rolling_ball_background(plane, radius)
        result.append(plane - background)
    return np.asarray(result[0] if frame.ndim == 2 else np.stack(result, axis=-1), dtype=np.float32)


def flat_field(frame: Frame, dark: Frame | None, flat: Frame | None) -> Frame:
    for name, reference in (("dark", dark), ("flat", flat)):
        if isinstance(reference, str):
            raise PdXrayError(f"Reference '{reference}' for '{name}' isn't loaded; call Pipeline.resolve() first.")
        if reference is not None and reference.shape != frame.shape:
            raise PdXrayError(f"The {name} frame is {reference.shape}, but frames are {frame.shape}.")
    if dark is None and flat is None:
        raise PdXrayError("flat_field needs a dark frame, a flat frame, or both.")
    signal = frame if dark is None else frame - dark
    if flat is None:
        return np.asarray(signal, dtype=np.float32)
    beam = flat if dark is None else flat - dark
    result = np.zeros_like(frame)
    np.divide(signal, beam, out=result, where=beam > 0)
    return result


def average_frames(
    frames: Iterable[NDArray[Any]],
    method: Literal["mean", "median"] = "mean",
    total: int | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> Frame:
    """Combine frames (e.g. darks or flats) into one float32 reference frame.

    "mean" uses constant memory. "median" is robust to outliers (e.g. zingers) but holds all frames in memory.
    """
    tracked = track(frames, total, f"Averaging frames ({method})", progress, cancel)
    if method == "median":
        stack = np.stack([np.asarray(f, dtype=np.float32) for f in tracked])
        return np.asarray(np.median(stack, axis=0), dtype=np.float32)
    total_sum: NDArray[np.float64] | None = None
    count = 0
    for frame in tracked:
        total_sum = frame.astype(np.float64) if total_sum is None else total_sum + frame
        count += 1
    if total_sum is None:
        raise PdXrayError("No frames to average.")
    return np.asarray(total_sum / count, dtype=np.float32)


OPERATIONS: Mapping[str, Operation] = MappingProxyType(
    {
        op.name: op
        for op in (
            Operation(
                "flat_field",
                "Flat/dark correction",
                "(frame - dark) / (flat - dark). Pixels where flat <= dark become 0. Either reference may be omitted.",
                (
                    Param("dark", "reference", optional=True, description="Averaged dark frame (no beam)."),
                    Param("flat", "reference", optional=True, description="Averaged flat frame (beam, no sample)."),
                ),
                flat_field,
            ),
            Operation(
                "invert",
                "Invert",
                "high + low - value.",
                (Param("low", "float", 0.0), Param("high", "float", 65535.0, description="e.g. 4095 for 12-bit data.")),
                invert,
            ),
            Operation(
                "gaussian",
                "Gaussian blur",
                "Gaussian filter (reflected edges).",
                (Param("sigma", "float", 1.0, 0.1, 100.0, description="Standard deviation in pixels."),),
                gaussian,
            ),
            Operation(
                "median",
                "Median filter",
                "Median over a square window. Removes outliers (hot pixels, zingers).",
                (Param("size", "int", 3, 1, 51, description="Window width in pixels."),),
                median,
            ),
            Operation(
                "normalise",
                "Normalise",
                "Map this frame's percentiles to 0..1. Computed per frame, so frames are scaled differently.",
                (
                    Param("low_percentile", "float", 0.0, 0.0, 100.0),
                    Param("high_percentile", "float", 100.0, 0.0, 100.0),
                ),
                normalise,
            ),
            Operation(
                "clip",
                "Clip",
                "Limit values to [low, high]. Leave a bound empty to skip it.",
                (Param("low", "float", optional=True), Param("high", "float", optional=True)),
                clip,
            ),
            Operation(
                "threshold",
                "Threshold",
                "1 where the value is kept, 0 elsewhere.",
                (Param("value", "float", 0.0), Param("keep", "choice", "above", choices=("above", "below"))),
                threshold,
            ),
            Operation(
                "crop",
                "Crop",
                "Keep a rectangle. Height or width 0 means up to the edge.",
                (
                    Param("top", "int", 0, 0),
                    Param("left", "int", 0, 0),
                    Param("height", "int", 0, 0),
                    Param("width", "int", 0, 0),
                ),
                crop,
            ),
            Operation(
                "rotate",
                "Rotate",
                "Rotate clockwise by a multiple of 90°.",
                (Param("angle", "choice", "90", choices=("90", "180", "270")),),
                rotate,
            ),
            Operation(
                "flip",
                "Flip",
                "Mirror the frame.",
                (Param("direction", "choice", "horizontal", choices=("horizontal", "vertical")),),
                flip,
            ),
            Operation(
                "rolling_ball",
                "Subtract background (rolling ball)",
                "Subtract a rolling-ball background, as in ImageJ (image shrunk for radii > 10).",
                (
                    Param("radius", "float", 50.0, 1.0, 1000.0, description="Ball radius in pixels."),
                    Param("light_background", "bool", False, description="Background brighter than the objects."),
                ),
                rolling_ball,
            ),
        )
    }
)
