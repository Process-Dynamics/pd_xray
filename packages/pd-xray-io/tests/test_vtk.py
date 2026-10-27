from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pytest

from pd_xray.core import PdXrayError
from pd_xray.io import convert
from pd_xray.io.vtk import open_vtk, open_vtk_series

# Uneven (nz, ny, nx) so that a mixed-up axis order fails.
VOLUME = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) / 7


def write_vtk(
    path: Path,
    volume: np.ndarray[Any, Any],
    title: str = "t = 0.45 s",
    type_name: str = "float",
    encoding: str = "BINARY",
    after: bytes = b"\n",
) -> Path:
    """A legacy VTK STRUCTURED_POINTS file as written by the simulation (big-endian data after the header)."""
    nz, ny, nx = volume.shape
    header = (
        f"# vtk DataFile Version 3.0\n{title}\n{encoding}\nDATASET STRUCTURED_POINTS\nDIMENSIONS {nx} {ny} {nz}\n"
        f"ORIGIN 0 0 10\nSPACING 1 2 3\n\nPOINT_DATA {volume.size}\nSCALARS frac1 {type_name}\nLOOKUP_TABLE default\n"
    )
    path.write_bytes(header.encode() + volume.astype(volume.dtype.newbyteorder(">")).tobytes() + after)
    return path


def test_volume_is_lazy_native_and_zyx(tmp_path: Path) -> None:
    with open_vtk(write_vtk(tmp_path / "a.vtk", VOLUME)) as ds:
        assert ds.shape == (3, 4, 5) and ds.axes == ("z", "y", "x")
        assert ds.dtype == np.float32 and ds.dtype.isnative
        assert ds.metadata == {
            "scalar_name": "frac1",
            "vtk_title": "t = 0.45 s",
            "spacing": [3.0, 2.0, 1.0],
            "origin": [10.0, 0.0, 0.0],
            "time_s": 0.45,
        }
        np.testing.assert_array_equal(ds[...], VOLUME)
        np.testing.assert_array_equal(ds[1:, 2, ::2], VOLUME[1:, 2, ::2])
        assert ds[-1].dtype.isnative


def test_integer_types_and_titles_without_time(tmp_path: Path) -> None:
    data = np.arange(-30, 30, dtype=np.int16).reshape(3, 4, 5)
    with open_vtk(write_vtk(tmp_path / "s.vtk", data, title="phase field", type_name="short")) as ds:
        assert ds.dtype == np.int16 and "time_s" not in ds.metadata
        np.testing.assert_array_equal(ds[[2, 0]], data[[2, 0]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"encoding": "ASCII"}, "only BINARY"),
        ({"type_name": "float 3"}, "3 components"),
        ({"type_name": "bit"}, "unsupported data type"),
        ({"after": b""}, None),  # one trailing newline is optional
        ({"after": b"\nSCALARS conc1 float\nLOOKUP_TABLE default\n" + bytes(240)}, "only files with one field"),
    ],
)
def test_unsupported_files_are_refused(tmp_path: Path, kwargs: dict[str, Any], message: str | None) -> None:
    path = write_vtk(tmp_path / "x.vtk", VOLUME, **kwargs)
    if message is None:
        open_vtk(path).close()
        return
    with pytest.raises(PdXrayError, match=message):
        open_vtk(path)


def test_truncated_and_other_dataset_types(tmp_path: Path) -> None:
    path = write_vtk(tmp_path / "t.vtk", VOLUME)
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(PdXrayError, match="truncated"):
        open_vtk(path)
    path.write_bytes(path.read_bytes().replace(b"STRUCTURED_POINTS", b"RECTILINEAR_GRID"))
    with pytest.raises(PdXrayError, match="only STRUCTURED_POINTS"):
        open_vtk(path)


def test_series_to_4d_hdf5(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    for i, t in ((1, 0), (2, 0.5), (10, 1)):  # natural order: _2 before _10
        write_vtk(tmp_path / f"v_{i}.vtk", VOLUME + i, title=f"t = {t} s")
    with open_vtk_series(tmp_path) as series:
        assert series.shape == (3, 3, 4, 5) and series.axes == ("time", "z", "y", "x")
        assert series.metadata["time_s"] == [0, 0.5, 1]
        assert series.metadata["source_files"] == ["v_1.vtk", "v_2.vtk", "v_10.vtk"]
        np.testing.assert_array_equal(series[2, 1], VOLUME[1] + 10)
        convert(series, tmp_path / "out.h5")
    with h5py.File(tmp_path / "out.h5") as file:
        data = file["data"]
        assert data.dtype == np.float32 and data.chunks == (1, 1, 4, 5)
        assert [dim.label for dim in data.dims] == ["time", "z", "y", "x"]
        np.testing.assert_array_equal(data.attrs["time_s"], [0, 0.5, 1])
        np.testing.assert_array_equal(data[1], VOLUME + 2)


def test_series_needs_one_grid(tmp_path: Path) -> None:
    write_vtk(tmp_path / "a.vtk", VOLUME)
    write_vtk(tmp_path / "b.vtk", VOLUME[:, :3])
    with pytest.raises(PdXrayError, match="same grid"):
        open_vtk_series(tmp_path)
    with open_vtk_series([tmp_path / "a.vtk", tmp_path / "a.vtk"]) as series:
        assert series.warnings  # equal times aren't increasing


def test_reads_through_a_store(tmp_path: Path) -> None:
    class Store:  # `storage=` stand-in: open() only, paths relative to tmp_path
        def open(self, path: str) -> BinaryIO:
            return (tmp_path / path).open("rb")

    write_vtk(tmp_path / "a.vtk", VOLUME)
    with open_vtk("a.vtk", storage=Store()) as ds:  # type: ignore[arg-type]
        assert ds._memmap is None  # type: ignore[attr-defined]
        np.testing.assert_array_equal(ds[...], VOLUME)
        np.testing.assert_array_equal(ds[-1, 1:3], VOLUME[-1, 1:3])
    with open_vtk_series(["a.vtk", "a.vtk"], storage=Store()) as series:  # type: ignore[arg-type]
        np.testing.assert_array_equal(series[:, 2], np.stack([VOLUME[2]] * 2))
