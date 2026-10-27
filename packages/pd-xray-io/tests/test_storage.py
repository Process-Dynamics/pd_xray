"""Reading through a `storage=` object instead of local paths (the NAS/S3 case), with a fake store over a folder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("tifffile")
pytest.importorskip("PIL")

from pd_xray.io import ArrayDataset, convert, iter_frames, open_dataset, open_sequence, reference_frames
from pd_xray.io.hdf5 import inspect_hdf5

DATA = np.arange(3 * 6 * 5, dtype=np.uint16).reshape(3, 6, 5)


@dataclass(frozen=True)
class FakeEntry:
    name: str
    path: str
    is_dir: bool


class FakeStore:
    """Files under `root`, addressed by relative paths; counts how often files are opened."""

    def __init__(self, root: Path) -> None:
        self.root, self.opened = root, 0

    def open(self, path: str) -> BinaryIO:
        self.opened += 1
        return (self.root / path).open("rb")

    def stat(self, path: str) -> Any:
        full = self.root / path
        if not full.exists():
            raise FileNotFoundError(path)
        return FakeEntry(full.name, path, full.is_dir())

    def list(self, path: str) -> list[Any]:
        return [FakeEntry(p.name, str(Path(path) / p.name), p.is_dir()) for p in (self.root / path).iterdir()]


def test_formats_through_a_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remote = tmp_path / "remote"
    (remote / "scan" / "dar").mkdir(parents=True)
    convert(ArrayDataset(DATA, ("angle", "y", "x")), remote / "scan" / "a.h5")
    convert(ArrayDataset(DATA, ("angle", "y", "x")), remote / "scan" / "a.tif")
    convert(ArrayDataset(DATA, ("frame", "y", "x")), remote / "scan" / "f.png")  # f_00000.png, …
    for i in (10, 2):
        convert(ArrayDataset(np.full((6, 5), i, np.uint16), ("y", "x")), remote / "scan" / "dar" / f"d{i}.tif")
    store = FakeStore(remote)
    monkeypatch.chdir(tmp_path)  # relative paths must not resolve against the working directory

    assert [d.name for d in inspect_hdf5("scan/a.h5", store)] == ["data"]
    for name in ("scan/a.h5", "scan/a.tif"):
        with open_dataset(name, storage=store) as ds:
            assert ds.axes == ("angle", "y", "x")
            np.testing.assert_array_equal(ds[1:], DATA[1:])
    with open_sequence("scan", "f_*.png", storage=store) as seq:
        assert [f.name for f in seq.files] == ["f_00000.png", "f_00001.png", "f_00002.png"]
        np.testing.assert_array_equal(np.stack(list(iter_frames(seq))), DATA)
    frames = reference_frames(["dar", "/scan/f_00002.png"], base="scan", storage=store)
    assert [int(f[0, 0]) for f in frames] == [2, 10, DATA[2, 0, 0]]
    opened = store.opened
    with pytest.raises(FileNotFoundError):
        open_dataset("scan/missing.h5", storage=store)
    assert store.opened == opened + 1
