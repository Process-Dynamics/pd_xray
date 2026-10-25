import os
from pathlib import Path

import pytest

from pd_xray.core import CancelToken, OperationCancelledError, Progress
from pd_xray.storage import IntegrityError, LocalStorage, copy, transfer

DATA = bytes(range(256)) * 40  # 10 240 bytes: 11 chunks of 1000


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transfer, "CHUNK_BYTES", 1000)


@pytest.fixture
def source(tmp_path: Path) -> LocalStorage:
    root = tmp_path / "src"
    (root / "scan" / "dark").mkdir(parents=True)
    (root / "scan" / "a.bin").write_bytes(DATA)
    (root / "scan" / "dark" / "d.bin").write_bytes(DATA[:1500])
    (root / "scan" / "empty.bin").write_bytes(b"")
    os.utime(root / "scan" / "a.bin", (1_700_000_000, 1_700_000_000))
    return LocalStorage(root)


def test_folder_copy_skip_and_overwrite(source: LocalStorage, tmp_path: Path) -> None:
    (tmp_path / "dst").mkdir()
    destination = LocalStorage(tmp_path / "dst")
    updates: list[Progress] = []
    result = copy(source, "scan", destination, "out/scan", progress=updates.append)
    assert (result.copied, result.skipped, result.bytes_copied) == (3, 0, len(DATA) + 1500)
    copied = tmp_path / "dst" / "out" / "scan"
    assert (copied / "a.bin").read_bytes() == DATA and (copied / "dark" / "d.bin").read_bytes() == DATA[:1500]
    assert (copied / "empty.bin").stat().st_size == 0
    assert int((copied / "a.bin").stat().st_mtime) == 1_700_000_000
    assert updates[-1].current == updates[-1].total == len(DATA) + 1500
    assert not list(copied.rglob("*.pd-xray-part"))

    assert copy(source, "scan", destination, "out/scan").skipped == 3  # nothing new

    (copied / "a.bin").write_bytes(b"changed")
    with pytest.raises(FileExistsError, match="1 file"):
        copy(source, "scan", destination, "out/scan")
    assert (copied / "a.bin").read_bytes() == b"changed"  # nothing copied before the check
    result = copy(source, "scan", destination, "out/scan", overwrite=True, verify=True)
    assert (result.copied, result.skipped, result.checksummed) == (1, 2, 1)
    assert (copied / "a.bin").read_bytes() == DATA

    assert copy(source, "scan/a.bin", destination, "single.bin").copied == 1  # a single file
    assert (tmp_path / "dst" / "single.bin").read_bytes() == DATA


def test_cancel_and_resume(source: LocalStorage, tmp_path: Path) -> None:
    (tmp_path / "dst").mkdir()
    destination = LocalStorage(tmp_path / "dst")
    stop = CancelToken()

    def cancel_after_some(update: Progress) -> None:
        if update.current >= 3000:
            stop.cancel()

    with pytest.raises(OperationCancelledError):
        copy(source, "scan/a.bin", destination, "a.bin", progress=cancel_after_some, cancel=stop)
    [part] = (tmp_path / "dst").glob(".a.bin.*.pd-xray-part")
    kept = part.stat().st_size
    assert 3000 <= kept < len(DATA) and not (tmp_path / "dst" / "a.bin").exists()
    (tmp_path / "dst" / ".a.bin.999-1.pd-xray-part").write_bytes(b"stale")  # from an older source version

    result = copy(source, "scan/a.bin", destination, "a.bin")
    assert result.bytes_copied == len(DATA) - kept  # resumed, not restarted
    assert (tmp_path / "dst" / "a.bin").read_bytes() == DATA
    assert not list((tmp_path / "dst").glob("*.pd-xray-part"))


def test_integrity(source: LocalStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "dst").mkdir()
    destination = LocalStorage(tmp_path / "dst")
    monkeypatch.setattr(source, "md5", lambda path: "0" * 32)  # as if S3 reported a different MD5
    with pytest.raises(IntegrityError, match="MD5"):
        copy(source, "scan/a.bin", destination, "a.bin")
    assert not list((tmp_path / "dst").iterdir())  # bad copy deleted, part included
