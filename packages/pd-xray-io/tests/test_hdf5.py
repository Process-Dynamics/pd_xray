from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pd_xray.io import open_dataset  # noqa: E402
from pd_xray.io.hdf5 import DatasetSelectionError, inspect_hdf5  # noqa: E402


@pytest.fixture
def h5_file(tmp_path: Path) -> Path:
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        d = f.create_dataset(
            "entry/proj", data=np.arange(4 * 8 * 6, dtype=np.uint16).reshape(4, 8, 6), chunks=(1, 8, 6)
        )
        d.attrs["units"] = b"counts"
        d.attrs["n_angles"] = np.int64(4)
        d.attrs["angles"] = np.array([0.0, 45.0])
        f.create_dataset("entry/angles", data=np.arange(4.0))
    return path


def test_inspect_lists_datasets(h5_file: Path) -> None:
    infos = {i.name: i for i in inspect_hdf5(h5_file)}
    proj = infos["entry/proj"]
    assert (proj.shape, proj.dtype, proj.chunks, proj.is_image) == ((4, 8, 6), "uint16", (1, 8, 6), True)
    assert proj.attrs == {"units": "counts", "n_angles": 4, "angles": [0.0, 45.0]}
    assert not infos["entry/angles"].is_image


def test_open_single_image_dataset_lazily(h5_file: Path) -> None:
    with open_dataset(h5_file) as ds:
        assert ds.axes == ("frame", "y", "x")
        assert ds.dtype == np.uint16
        assert ds[2].dtype == np.uint16
        np.testing.assert_array_equal(ds[1:3, 2:4, 0], np.arange(4 * 8 * 6).reshape(4, 8, 6)[1:3, 2:4, 0])


def test_axes_override_and_dimension_labels(h5_file: Path) -> None:
    with open_dataset(h5_file, axes=("angle", "y", "x")) as ds:
        assert ds.axes == ("angle", "y", "x")
    with h5py.File(h5_file, "a") as f:
        for dim, label in zip(f["entry/proj"].dims, ("Time", "y", "x"), strict=True):
            dim.label = label
    with open_dataset(h5_file) as ds:
        assert ds.axes == ("time", "y", "x")


def test_dataset_choice_required_when_ambiguous(h5_file: Path) -> None:
    with h5py.File(h5_file, "a") as f:
        f.create_dataset("entry/flat", data=np.zeros((8, 6), dtype=np.uint16))
    with pytest.raises(DatasetSelectionError, match="entry/proj"):
        open_dataset(h5_file)
    with open_dataset(h5_file, dataset="entry/flat") as ds:
        assert ds.axes == ("y", "x")
    with pytest.raises(DatasetSelectionError):
        open_dataset(h5_file, dataset="entry")
