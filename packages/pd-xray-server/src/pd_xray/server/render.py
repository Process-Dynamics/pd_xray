"""Display rendering: pyramid levels, contrast stretching and 8-bit image encoding. Never changes the data itself.

Level 0 of a frame is full resolution, and level `n` is level `n - 1` halved with a 2x2 block mean (so shrunk by
`2**n`). The last level fits in one tile. Tiles are `tile_size` squares of a level, addressed by (column, row); edge
tiles are smaller.
"""

import io
import json
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import PdXrayError, import_optional
from pd_xray.processing import TONE_PARAMS, tone_map

ImageFormat = Literal["png", "jpeg"]
MEDIA_TYPES: dict[str, str] = {"png": "image/png", "jpeg": "image/jpeg"}


def level_count(height: int, width: int, tile_size: int) -> int:
    """Number of pyramid levels for a frame, down to the first level that fits in one tile."""
    levels = 1
    while max(height, width) > tile_size * 2 ** (levels - 1):
        levels += 1
    return levels


def halve(frame: NDArray[Any]) -> NDArray[np.float32]:
    """2x2 block mean. An odd last row or column is averaged on its own.

    Args:
        frame: (y, x) or (y, x, channel), any numeric dtype. Sums of four uint16 values are exact in float32.

    Returns:
        float32 array of shape (ceil(y / 2), ceil(x / 2)[, channel]).
    """
    data = frame.astype(np.float32, copy=False)
    if data.shape[0] % 2:  # repeating the last row/column makes its blocks the mean of the pixels that exist
        data = np.concatenate([data, data[-1:]], axis=0)
    if data.shape[1] % 2:
        data = np.concatenate([data, data[:, -1:]], axis=1)
    result: NDArray[np.float32] = (data[0::2, 0::2] + data[1::2, 0::2] + data[0::2, 1::2] + data[1::2, 1::2]) * 0.25
    return result


def frame_stats(frame: NDArray[Any]) -> dict[str, float]:
    """Display statistics of a frame, ignoring NaN and inf.

    Returns:
        {"min", "max"} over all pixels, and {"low", "high"}: the 0.1 and 99.9 percentiles of every 4th pixel in y and x,
        used as automatic contrast limits.
    """
    finite = frame if frame.dtype.kind in "biu" else frame[np.isfinite(frame)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "low": 0.0, "high": 0.0}
    sample = frame[::4, ::4]
    sample = sample if sample.dtype.kind in "biu" else sample[np.isfinite(sample)]
    low, high = np.percentile(sample, [0.1, 99.9]) if sample.size else (finite.min(), finite.max())
    return {"min": float(finite.min()), "max": float(finite.max()), "low": float(low), "high": float(high)}


def parse_tone(text: str | None) -> str | None:
    """Check display tone settings (JSON of `TONE_PARAMS` values; missing ones take defaults).

    Returns:
        Canonical JSON with every setting (usable as a cache key), or None for no change (missing or all defaults).

    Raises:
        PdXrayError: For invalid JSON, unknown settings or invalid values.
    """
    if not text:
        return None
    try:
        settings = json.loads(text)
    except json.JSONDecodeError:
        raise PdXrayError(f"Tone settings must be JSON, got {text!r}.") from None
    known = {p.name: p for p in TONE_PARAMS}
    if not isinstance(settings, dict) or set(settings) - set(known):
        raise PdXrayError(f"Tone settings must be an object with keys from {sorted(known)}, got {text!r}.")
    checked = {name: param.validate(settings.get(name, param.default)) for name, param in known.items()}
    defaults = {name: param.validate(param.default) for name, param in known.items()}
    return None if checked == defaults else json.dumps(checked, sort_keys=True)


def histogram(frame: NDArray[Any], bins: int, low: float, high: float) -> list[int]:
    """Pixel counts in `bins` equal bins over [low, high] (values outside are not counted; NaN and inf ignored).

    Colour frames count every channel.
    """
    if high <= low:
        high = low + 1  # a constant frame: one bin holds everything
    counts, _ = np.histogram(frame, bins=bins, range=(low, high))
    return [int(c) for c in counts]


def to_uint8(data: NDArray[Any], vmin: float, vmax: float, tone: str | None = None) -> NDArray[np.uint8]:
    """Map [vmin, vmax] linearly onto [0, 255], clipping outside values. NaN becomes 0.

    `tone` (from `parse_tone`) applies photo-style tone adjustments in between (see `pd_xray.processing.tone_map`).
    """
    scale = 255.0 / (vmax - vmin) if vmax > vmin else 0.0
    if tone is not None:
        scale /= 255.0
    scaled = (data.astype(np.float32) - np.float32(vmin)) * np.float32(scale)
    if tone is not None:
        scaled = tone_map(scaled, **json.loads(tone)) * np.float32(255)
    np.clip(scaled, 0, 255, out=scaled)
    np.nan_to_num(scaled, copy=False, nan=0.0)
    np.rint(scaled, out=scaled)
    return scaled.astype(np.uint8)


def encode(image: NDArray[np.uint8], fmt: ImageFormat) -> bytes:
    """Encode a (y, x) grey or (y, x, 3|4) colour 8-bit image. PNG is lossless; JPEG (quality 90) drops alpha.

    Raises:
        PdXrayError: For a channel count other than 3 or 4.
    """
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise PdXrayError(f"Can't display a frame with {image.shape[2]} channels; expected 1, 3 or 4.")
    pil = import_optional("PIL.Image", "png", "pd-xray-io").fromarray(image)
    buffer = io.BytesIO()
    if fmt == "png":
        pil.save(buffer, format="PNG", compress_level=1)  # fast; size matters less than latency on a LAN
    else:
        pil.convert("RGB" if image.ndim == 3 else "L").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
