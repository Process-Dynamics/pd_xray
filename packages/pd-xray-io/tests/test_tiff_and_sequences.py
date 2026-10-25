from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("PIL")

from pd_xray.core import PdXrayError  # noqa: E402
from pd_xray.io import ArrayDataset, convert, open_dataset, open_sequence  # noqa: E402

STACK = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5) * 1000


def test_multipage_round_trip_is_memmapped(tmp_path: Path) -> None:
    convert(ArrayDataset(STACK, ("angle", "y", "x"), {"units": "counts"}), tmp_path / "s.tif")
    with open_dataset(tmp_path / "s.tif") as ds:
        assert ds.axes == ("angle", "y", "x")
        assert ds.metadata == {"units": "counts"}
        assert ds._memmap is not None  # type: ignore[attr-defined]
        np.testing.assert_array_equal(ds[1:, 2], STACK[1:, 2])


def test_compressed_multipage_reads_pages(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "c.tif", STACK, compression="zlib", photometric="minisblack")
    with open_dataset(tmp_path / "c.tif") as ds:
        assert ds._memmap is None  # type: ignore[attr-defined]
        assert ds.axes == ("frame", "y", "x")
        np.testing.assert_array_equal(ds[[2, 0], :, -1], STACK[[2, 0], :, -1])
        np.testing.assert_array_equal(ds[-1], STACK[-1])


def test_compressed_write_is_one_series_with_metadata(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    convert(ArrayDataset(STACK, ("angle", "y", "x"), {"units": "counts"}), tmp_path / "z.tif", compression="zlib")
    with open_dataset(tmp_path / "z.tif") as ds:
        assert (ds.axes, ds.metadata) == (("angle", "y", "x"), {"units": "counts"})
        np.testing.assert_array_equal(ds[...], STACK)
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_orientation_tag_recorded_not_applied(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "o.tif", STACK[0], extratags=[(274, "H", 1, 3)])
    with open_dataset(tmp_path / "o.tif") as ds:
        assert ds.metadata["tiff_orientation"] == 3
        np.testing.assert_array_equal(ds[...], STACK[0])


def test_sequence_natural_order_and_pattern(tmp_path: Path) -> None:
    for i in (10, 2, 1):
        tifffile.imwrite(tmp_path / f"s_{i}_img.tif", np.full((4, 5), i, np.uint16))
    tifffile.imwrite(tmp_path / "s_0_dar.tif", np.zeros((4, 5), np.uint16))
    ds = open_sequence(tmp_path, "*_img.tif")
    assert [f.name for f in ds.files] == ["s_1_img.tif", "s_2_img.tif", "s_10_img.tif"]
    assert ds.shape == (3, 4, 5)
    assert ds[::-1, 0, 0].tolist() == [10, 2, 1]


def test_sequence_rejects_mismatched_frame(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "a1.tif", np.zeros((4, 5), np.uint16))
    tifffile.imwrite(tmp_path / "a2.tif", np.zeros((4, 6), np.uint16))
    ds = open_sequence(tmp_path)
    with pytest.raises(PdXrayError, match=r"a2\.tif"):
        ds[1]
