from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("tifffile")

from pd_xray.core import CancelToken, OperationCancelledError, PdXrayError, Progress
from pd_xray.io import ArrayDataset, convert, iter_frames, map_frames, open_dataset, reference_frames

DATA = np.arange(4 * 3 * 6 * 5, dtype=np.uint16).reshape(4, 3, 6, 5)  # (angle, time, y, x)


def crop_double(frame: NDArray[Any]) -> NDArray[Any]:
    return frame[1:, :3].astype(np.float32) * 2


def test_mapped_indexing_and_parallel_order() -> None:
    calls = []
    source = ArrayDataset(DATA, ("angle", "time", "y", "x"), {"energy": 25})

    def function(frame: NDArray[Any]) -> NDArray[Any]:
        calls.append(1)
        return crop_double(frame)

    mapped = map_frames(source, function, workers=3, metadata={"note": "x"})
    assert (mapped.shape, mapped.dtype, mapped.axes) == ((4, 3, 5, 3), np.float32, source.axes)
    assert mapped.metadata == {"energy": 25, "note": "x"}
    expected = DATA[:, :, 1:, :3].astype(np.float32) * 2
    np.testing.assert_array_equal(mapped[0, 0], expected[0, 0])
    assert len(calls) == 1  # the first frame is kept from finding the shape
    np.testing.assert_array_equal(mapped[1:3, -1, 2], expected[1:3, -1, 2])
    np.testing.assert_array_equal(mapped[..., 0], expected[..., 0])
    np.testing.assert_array_equal(np.stack(list(iter_frames(mapped))), expected.reshape(12, 5, 3))
    np.testing.assert_array_equal(np.stack(list(iter_frames(mapped, [3, 1]))), expected[[3, 1]].reshape(6, 5, 3))
    with pytest.raises(PdXrayError, match="2D frames into 1D"):
        map_frames(source, lambda f: f[0])


def test_convert_dtype_choice(tmp_path: Path) -> None:
    mapped = map_frames(ArrayDataset(DATA[:, 0], ("angle", "y", "x")), crop_double, workers=2)
    convert(mapped, tmp_path / "f.tif")
    with open_dataset(tmp_path / "f.tif") as tif:
        assert tif.dtype == np.float32
        np.testing.assert_array_equal(tif[...], DATA[:, 0, 1:, :3] * 2.0)
    convert(mapped, tmp_path / "u.tif", value_range=(0, 100), dtype="uint8", frames=slice(0, 2))
    with open_dataset(tmp_path / "u.tif") as tif:
        assert (tif.dtype, tif.shape) == (np.uint8, (2, 5, 3))
        assert tif.metadata["value_scaling"]["value_range"] == [0, 100]
        expected = np.rint(np.clip(DATA[:2, 0, 1:, :3] * 2.0, 0, 100) * 255 / 100)
        np.testing.assert_array_equal(tif[...], expected)
    with pytest.raises(PdXrayError, match="can't store"):
        convert(mapped, tmp_path / "f.png", dtype="float32")


def test_cancelled_write_leaves_nothing_and_keeps_the_old_file(tmp_path: Path) -> None:
    token = CancelToken()
    source = ArrayDataset(DATA[:, 0], ("angle", "y", "x"))
    convert(source, tmp_path / "out.tif")
    before = (tmp_path / "out.tif").read_bytes()

    def cancel_after_two(update: Progress) -> None:
        if update.current == 2:
            token.cancel()

    for name in ("out.tif", "new.h5", "new.png"):
        with pytest.raises(OperationCancelledError):
            convert(
                map_frames(source, crop_double, workers=2), tmp_path / name, progress=cancel_after_two, cancel=token
            )
        token = CancelToken()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.tif"]  # no staging folders or partial files
    assert (tmp_path / "out.tif").read_bytes() == before


def test_reference_frames(tmp_path: Path) -> None:
    (tmp_path / "dark").mkdir()
    for i in (10, 2):  # natural order: d2 before d10
        convert(ArrayDataset(np.full((2, 3), i, np.uint16), ("y", "x")), tmp_path / "dark" / f"d{i}.tif")
    convert(ArrayDataset(np.zeros((2, 2, 3), np.uint16), ("frame", "y", "x")), tmp_path / "two.tif")
    frames: Iterator[NDArray[Any]] = reference_frames(["dark", str(tmp_path / "two.tif")], base=tmp_path)
    assert [int(f[0, 0]) for f in frames] == [2, 10, 0, 0]
    with pytest.raises(PdXrayError, match="doesn't exist"):
        list(reference_frames("missing", tmp_path))
    convert(ArrayDataset(np.zeros((4, 4), np.uint16), ("y", "x")), tmp_path / "big.tif")
    with pytest.raises(PdXrayError, match="earlier files"):
        list(reference_frames(["two.tif", "big.tif"], tmp_path))
