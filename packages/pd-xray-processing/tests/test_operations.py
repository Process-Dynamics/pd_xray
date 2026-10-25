import numpy as np
import pytest
from scipy import ndimage

from pd_xray.core import PdXrayError
from pd_xray.processing import OPERATIONS, Pipeline, Step, average_frames, tone_map

RNG = np.random.default_rng(0)
FRAME = RNG.uniform(0, 100, (40, 60)).astype(np.float32)


def test_every_operation_has_valid_defaults_and_keeps_float32() -> None:
    for name, op in OPERATIONS.items():
        if name == "flat_field":
            continue
        result = op(FRAME)
        assert result.dtype == np.float32, name
        assert result is not FRAME and np.shares_memory(result, FRAME) is False, name


def test_param_validation() -> None:
    with pytest.raises(PdXrayError, match="sigma"):
        OPERATIONS["gaussian"](FRAME, sigma=-1)
    with pytest.raises(PdXrayError, match="integer"):
        OPERATIONS["median"](FRAME, size=2.5)
    with pytest.raises(PdXrayError, match="no parameter"):
        OPERATIONS["gaussian"](FRAME, radius=1)
    with pytest.raises(PdXrayError, match="one of"):
        OPERATIONS["rotate"](FRAME, angle="45")


def test_simple_ops_values() -> None:
    assert OPERATIONS["invert"](FRAME, low=0, high=100)[3, 4] == pytest.approx(100 - FRAME[3, 4])
    clipped = OPERATIONS["clip"](FRAME, low=20, high=None)
    assert clipped.min() == 20 and clipped.max() == FRAME.max()
    mask = OPERATIONS["threshold"](FRAME, value=50)
    np.testing.assert_array_equal(mask, (FRAME >= 50).astype(np.float32))
    normalised = OPERATIONS["normalise"](FRAME)
    assert (normalised.min(), normalised.max()) == (0, 1)
    np.testing.assert_array_equal(OPERATIONS["crop"](FRAME, top=5, left=10, height=3), FRAME[5:8, 10:])
    with pytest.raises(PdXrayError, match="outside"):
        OPERATIONS["crop"](FRAME, top=39, height=5)
    np.testing.assert_array_equal(OPERATIONS["rotate"](FRAME, angle="90"), np.rot90(FRAME, -1))
    np.testing.assert_array_equal(OPERATIONS["flip"](FRAME, direction="vertical"), FRAME[::-1])


def test_filters_leave_channels_alone() -> None:
    colour = np.stack([FRAME, FRAME * 2], axis=-1)
    blurred = OPERATIONS["gaussian"](colour, sigma=2)
    np.testing.assert_allclose(blurred[..., 1], 2 * ndimage.gaussian_filter(FRAME, 2), rtol=1e-5)
    np.testing.assert_array_equal(OPERATIONS["median"](colour, size=3)[..., 0], ndimage.median_filter(FRAME, 3))


def test_rolling_ball_removes_smooth_background() -> None:
    y, x = np.mgrid[:200, :300]
    background = (1000 + 0.05 * x + 0.02 * y).astype(np.float32)
    spot = np.zeros_like(background)
    spot[95:105, 145:155] = 500
    for radius in (8, 50):  # 50 uses the shrunk path
        result = OPERATIONS["rolling_ball"](background + spot, radius=radius)
        assert np.abs(result[40:160, 40:260][spot[40:160, 40:260] == 0]).max() < 10
        assert result[100, 150] == pytest.approx(500, abs=10)
    light = OPERATIONS["rolling_ball"](background - spot, radius=50, light_background=True)
    assert light[100, 150] == pytest.approx(-500, abs=10)


def test_flat_field_and_average() -> None:
    darks = [np.full((4, 5), v, np.uint16) for v in (9, 10, 11)]
    dark = average_frames(darks)
    assert dark.dtype == np.float32 and dark[0, 0] == 10
    assert average_frames(darks + [np.full((4, 5), 1000, np.uint16)], method="median")[0, 0] == 10.5
    flat = np.full((4, 5), 110, np.float32)
    flat[0, 0] = 10  # flat == dark: no beam, result 0
    frame = np.full((4, 5), 60, np.float32)
    result = OPERATIONS["flat_field"](frame, dark=dark, flat=flat)
    assert result[1, 1] == pytest.approx(0.5) and result[0, 0] == 0
    assert OPERATIONS["flat_field"](frame, dark=dark)[1, 1] == 50
    with pytest.raises(PdXrayError, match="resolve"):
        OPERATIONS["flat_field"](frame, dark="darks/*.tif")
    with pytest.raises(PdXrayError, match="dark frame is"):
        OPERATIONS["flat_field"](frame, dark=dark[:2])


def test_tone_map_adjustments() -> None:
    x = np.linspace(0, 1, 1001, dtype=np.float32)
    np.testing.assert_allclose(tone_map(x), x, atol=1e-6)
    np.testing.assert_allclose(tone_map(x, exposure=1), np.minimum(2 * x, 1), atol=1e-6)
    for name in ("contrast", "highlights", "shadows", "whites", "blacks"):
        for value in (-100, 100):  # every slider keeps the order of values and stays in [0, 1]
            y = tone_map(x, **{name: value})
            assert np.all(np.diff(y) >= -1e-6) and y.min() >= 0 and y.max() <= 1, (name, value)
    assert tone_map(x, contrast=100)[250] < 0.25 < 0.75 < tone_map(x, contrast=100)[750]
    assert tone_map(x, shadows=50)[333] > x[333] and tone_map(x, highlights=-50)[667] < x[667]
    assert tone_map(x, blacks=-100)[200] == 0 and tone_map(x, whites=100)[800] == 1
    np.testing.assert_allclose(tone_map(x, curve=[[0, 1], [1, 0]]), 1 - x, atol=1e-6)  # a curve may invert
    y = tone_map(x, curve=[[0.2, 0.1], [0.5, 0.6], [1, 1]])
    np.testing.assert_allclose(y[[0, 100, 200, 500]], [0.1, 0.1, 0.1, 0.6], atol=1e-6)  # flat before the first point
    assert np.isnan(tone_map(np.array([np.nan], dtype=np.float32))[0])


def test_tone_step_maps_black_white_and_back() -> None:
    frame = np.array([[100.0, 550.0, 1000.0, 2000.0]], dtype=np.float32)
    out = OPERATIONS["tone"](frame, black=100, white=1000, curve=[[0, 0], [0.5, 0.25], [1, 1]])
    np.testing.assert_allclose(out, [[100, 325, 1000, 1000]], rtol=1e-6)  # 0.5 -> 0.25 of the span; above clips
    with pytest.raises(PdXrayError, match="white point"):
        OPERATIONS["tone"](frame, black=5, white=5)
    with pytest.raises(PdXrayError, match="increasing x"):
        Pipeline([Step("tone", {"curve": [[0.5, 0], [0.2, 1]]})])
    with pytest.raises(PdXrayError, match="\\[x, y\\] points"):
        Pipeline([Step("tone", {"curve": "linear"})])
