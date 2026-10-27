import numpy as np
import pytest

from pd_xray.server.cache import LRUCache
from pd_xray.server.render import frame_stats, halve, level_count, to_uint8


def test_level_count() -> None:
    assert level_count(512, 512, 512) == 1
    assert level_count(513, 100, 512) == 2
    assert level_count(2304, 4096, 512) == 4  # 4096 / 2**3 = 512


def test_halve_edges_average_only_existing_pixels() -> None:
    frame = np.arange(5 * 7, dtype=np.uint16).reshape(5, 7)
    result = halve(frame)
    assert result.shape == (3, 4) and result.dtype == np.float32
    assert result[0, 0] == frame[:2, :2].mean()
    assert result[2, 3] == frame[4, 6]  # 1-pixel corner block
    assert result[1, 3] == frame[2:4, 6].mean()
    rgb = np.stack([frame, frame * 2, frame * 3], axis=-1)
    np.testing.assert_allclose(halve(rgb)[..., 2], result * 3, rtol=1e-6)


def test_to_uint8_maps_clips_and_zeroes_nan() -> None:
    data = np.array([-10.0, 0.0, 50.0, 100.0, 200.0, np.nan])
    assert to_uint8(data, 0, 100).tolist() == [0, 0, 128, 255, 255, 0]
    assert to_uint8(np.array([5, 5], dtype=np.uint16), 5, 5).tolist() == [0, 0]


def test_frame_stats_ignores_non_finite() -> None:
    frame = np.linspace(0, 1000, 64 * 64, dtype=np.float32).reshape(64, 64)
    frame[0, 0], frame[0, 4] = np.nan, np.inf
    stats = frame_stats(frame)
    assert stats["max"] == 1000 and 0 < stats["min"] < 1
    assert stats["min"] <= stats["low"] < stats["high"] <= stats["max"]


def test_cache_evicts_least_recent_and_skips_errors() -> None:
    cache = LRUCache(max_bytes=2000)
    calls: list[str] = []

    def make(name: str) -> np.ndarray:
        calls.append(name)
        return np.zeros(100, np.float64)  # 800 bytes

    for name in ["a", "b", "a", "c"]:  # "b" is least recent when "c" arrives
        cache.get_or_compute(name, lambda name=name: make(name))
    cache.get_or_compute("a", lambda: make("a"))
    cache.get_or_compute("b", lambda: make("b"))
    assert calls == ["a", "b", "c", "b"]
    assert cache.nbytes <= 2000
    with pytest.raises(ZeroDivisionError):
        cache.get_or_compute("bad", lambda: 1 / 0)
    assert cache.get_or_compute("bad", lambda: 7) == 7
