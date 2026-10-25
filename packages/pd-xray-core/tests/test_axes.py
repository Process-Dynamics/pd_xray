import pytest

from pd_xray.core import frame_axes, validate_axes


@pytest.mark.parametrize(
    "axes",
    [
        ("y", "x"),
        ("angle", "y", "x"),
        ("time", "angle", "y", "x"),
        ("y", "x", "channel"),
        ("time", "y", "x", "channel"),
    ],
)
def test_valid_axes(axes: tuple[str, ...]) -> None:
    assert validate_axes(list(axes), len(axes)) == axes


@pytest.mark.parametrize(
    ("axes", "ndim"),
    [
        (("y", "x"), 3),  # wrong length
        (("x", "y"), 2),  # wrong spatial order
        (("y", "y", "x"), 3),  # repeated
        (("Time", "y", "x"), 3),  # not lowercase
        (("", "y", "x"), 3),  # empty
        (("y", "x", "time"), 3),  # frame axis after spatial axes
        (("channel", "y", "x"), 3),  # channel only allowed last
    ],
)
def test_invalid_axes(axes: tuple[str, ...], ndim: int) -> None:
    with pytest.raises(ValueError):
        validate_axes(axes, ndim)


def test_frame_axes() -> None:
    assert frame_axes(("time", "angle", "y", "x", "channel")) == ("time", "angle")
    assert frame_axes(("y", "x")) == ()
