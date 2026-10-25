from pathlib import Path

import pytest

from pd_xray.storage import InvalidPathError, LocalStorage, join_path, normalise_path


def test_normalise_path() -> None:
    assert normalise_path("/a//b/./c/") == "a/b/c"
    assert normalise_path("") == normalise_path("/") == ""
    assert join_path("a", "b/c") == "a/b/c"
    with pytest.raises(InvalidPathError):
        normalise_path("a/../../etc")


def test_local_storage(tmp_path: Path) -> None:
    (tmp_path / "scan").mkdir()
    (tmp_path / "scan" / "a.bin").write_bytes(bytes(range(256)))
    (tmp_path / "link").symlink_to("/etc")
    storage = LocalStorage(tmp_path)

    assert {(e.name, e.is_dir) for e in storage.list()} == {("scan", True), ("link", True)}
    [entry] = storage.list("scan")
    assert (entry.path, entry.size) == ("scan/a.bin", 256)
    assert storage.stat("/scan/a.bin/") == entry
    assert storage.exists("scan") and not storage.exists("nope")
    with storage.open("scan/a.bin") as f:
        f.seek(100)
        assert f.read(3) == bytes([100, 101, 102])
    with pytest.raises(InvalidPathError):
        storage.list("link")  # symlink leaving the root
    with pytest.raises(FileNotFoundError):
        storage.stat("nope")
    storage.check()
