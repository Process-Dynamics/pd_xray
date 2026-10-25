"""Named axes describing array structure, e.g. ("angle", "y", "x").

Axis names are free lowercase strings. The constants below are the names other packages recognise. Arrays always end
with the spatial axes ("y", "x"), optionally followed by "channel" for colour images. Every other axis is a frame axis
the viewer can slide through.
"""

from collections.abc import Sequence

TIME = "time"
ANGLE = "angle"
Z = "z"
Y = "y"
X = "x"
CHANNEL = "channel"

Axes = tuple[str, ...]


def validate_axes(axes: Sequence[str], ndim: int) -> Axes:
    """Check that `axes` correctly describes an array with `ndim` dimensions.

    Returns:
        The axes as a tuple.

    Raises:
        ValueError: If the length is wrong, names are empty or repeated, or the axes don't end with ("y", "x") or
            ("y", "x", "channel").
    """
    result = tuple(axes)
    if len(result) != ndim:
        raise ValueError(f"Got {len(result)} axis names {result} for an array with {ndim} dimensions.")
    if any(not name or name != name.strip().lower() for name in result):
        raise ValueError(f"Axis names must be non-empty lowercase strings, got {result}.")
    if len(set(result)) != len(result):
        raise ValueError(f"Axis names must be unique, got {result}.")
    spatial = result[-3:-1] if result[-1:] == (CHANNEL,) else result[-2:]
    if spatial != (Y, X) or CHANNEL in result[:-1]:
        raise ValueError(f"Axes must end with ('y', 'x') or ('y', 'x', 'channel'), got {result}.")
    return result


def frame_axes(axes: Axes) -> Axes:
    """Axes that index frames: everything except "y", "x" and "channel"."""
    return tuple(name for name in axes if name not in (Y, X, CHANNEL))
