from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("PIL")

from pd_xray.io import ArrayDataset, convert, open_dataset
from pd_xray.io.dataset import split_key

FLOATS = np.linspace(-1.0, 3.0, 2 * 6 * 7, dtype=np.float32).reshape(2, 6, 7)


def test_split_key() -> None:
    assert split_key(-1, 5) == (4, ())
    assert split_key((slice(None, None, -2), 0), 5) == (range(4, -1, -2), (0,))
    assert split_key(..., 5) == (range(5), (...,))
    assert split_key([True, False, True], 3) == ([0, 2], ())
    with pytest.raises(IndexError):
        split_key(5, 5)


def test_hdf5_round_trip_keeps_dtype_axes_and_metadata(tmp_path: Path) -> None:
    source = ArrayDataset(FLOATS, ("time", "y", "x"), {"energy_kev": 25.0, "nested": {"a": 1}})
    convert(source, tmp_path / "f.h5", compression="gzip")
    with open_dataset(tmp_path / "f.h5") as ds:
        assert (ds.axes, ds.dtype) == (("time", "y", "x"), np.float32)
        assert ds.metadata == {"energy_kev": 25.0, "nested": '{"a": 1}'}  # no h5py DIMENSION_LABELS
        np.testing.assert_array_equal(ds[...], FLOATS)


def test_float_to_png_scales_with_one_recoverable_range(tmp_path: Path) -> None:
    convert(ArrayDataset(FLOATS, ("frame", "y", "x")), tmp_path / "f.png")
    for i in range(2):
        with open_dataset(tmp_path / f"f_0000{i}.png") as png:
            assert png.dtype == np.uint16
            scaling = png.metadata["value_scaling"]
            assert scaling["value_range"] == [-1.0, 3.0]
            recovered = scaling["offset"] + png[...] * scaling["scale"]
            np.testing.assert_allclose(recovered, FLOATS[i], atol=scaling["scale"] / 2 + 1e-9)


def test_uint16_png_is_exact_and_frames_select(tmp_path: Path) -> None:
    data = (np.arange(3 * 4 * 5, dtype=np.uint16) * 1000).reshape(3, 4, 5)
    convert(ArrayDataset(data, ("frame", "y", "x")), tmp_path / "u.png", frames=2)
    with open_dataset(tmp_path / "u.png") as png:
        assert "value_scaling" not in png.metadata
        assert png.metadata["source_frames"] == [2]
        np.testing.assert_array_equal(png[...], data[2])


def test_jpeg_is_uint8(tmp_path: Path) -> None:
    convert(ArrayDataset(FLOATS[0], ("y", "x")), tmp_path / "f.jpg")
    with open_dataset(tmp_path / "f.jpg") as jpg:
        assert jpg.dtype == np.uint8
        assert jpg.metadata["value_scaling"]["source_dtype"] == "float32"
