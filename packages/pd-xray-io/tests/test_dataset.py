from pathlib import Path

import numpy as np
import pytest

from pd_xray.io import ArrayDataset, UnsupportedFormatError, open_dataset


def test_array_dataset_keeps_dtype_and_slices() -> None:
    array = np.arange(4 * 3 * 2, dtype=np.uint16).reshape(4, 3, 2)
    with ArrayDataset(array, ("angle", "y", "x"), {"units": "counts"}) as ds:
        assert (ds.shape, ds.ndim, len(ds), ds.nbytes) == ((4, 3, 2), 3, 4, array.nbytes)
        assert ds[1].dtype == np.uint16
        np.testing.assert_array_equal(ds[1:3, :, 0], array[1:3, :, 0])
        assert ds.metadata == {"units": "counts"}


def test_dataset_rejects_wrong_axes() -> None:
    with pytest.raises(ValueError):
        ArrayDataset(np.zeros((2, 3)), ("angle", "y", "x"))


def test_open_dataset_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_dataset(tmp_path / "missing.h5")
    unknown = tmp_path / "file.xyz"
    unknown.touch()
    with pytest.raises(UnsupportedFormatError):
        open_dataset(unknown)
