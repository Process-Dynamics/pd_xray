"""S3Storage against moto's in-memory S3."""

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from pd_xray.storage import AuthenticationError, S3Storage  # noqa: E402

KEYS = {"access_key": "testing", "secret_key": "testing", "region": "us-east-1"}


@pytest.fixture
def bucket() -> Iterator[None]:
    with moto.mock_aws():
        client = boto3.client(
            "s3", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
        )
        client.create_bucket(Bucket="lab")
        client.put_object(Bucket="lab", Key="scans/0042/a.bin", Body=bytes(range(256)) * 4096)
        client.put_object(Bucket="lab", Key="scans/0042/b.bin", Body=b"xy")
        client.put_object(Bucket="lab", Key="scans/empty/", Body=b"")  # console-made folder marker
        client.put_object(Bucket="lab", Key="notes.txt", Body=b"hi")
        client.create_bucket(Bucket="empty")
        yield


def test_list_stat_open(bucket: None) -> None:
    with S3Storage("lab", **KEYS) as storage:
        storage.check()
        assert {(e.name, e.is_dir, e.size) for e in storage.list()} == {("scans", True, None), ("notes.txt", False, 2)}
        assert {(e.path, e.is_dir) for e in storage.list("scans")} == {("scans/0042", True), ("scans/empty", True)}
        assert storage.list("scans/empty") == []
        entry = storage.stat("/scans/0042/a.bin")
        assert (entry.name, entry.size, entry.is_dir) == ("a.bin", 1 << 20, False)
        assert storage.stat("scans").is_dir
        with storage.open("scans/0042/a.bin") as f:
            f.seek(1000)
            assert f.read(4) == bytes([232, 233, 234, 235])
            f.seek(-2, 2)
            assert f.read() == bytes([254, 255])
        with pytest.raises(FileNotFoundError):
            storage.stat("scans/nope")
        with pytest.raises(FileNotFoundError):
            storage.list("nope")
        with pytest.raises(NotADirectoryError):
            storage.list("notes.txt")
        with pytest.raises(IsADirectoryError):
            storage.open("scans")


def test_prefix_and_empty_bucket(bucket: None) -> None:
    with S3Storage("lab", prefix="scans/0042", **KEYS) as storage:
        assert sorted(e.path for e in storage.list()) == ["a.bin", "b.bin"]
        assert storage.stat("b.bin").size == 2
        storage.check()
    with S3Storage("empty", **KEYS) as storage:
        assert storage.list() == []
        storage.check()
    with pytest.raises(FileNotFoundError, match="doesn't exist"), S3Storage("missing", **KEYS) as storage:
        storage.check()


def test_errors(bucket: None) -> None:
    with pytest.raises(AuthenticationError, match="both"):
        S3Storage("lab", access_key="x")
    with pytest.raises(AuthenticationError, match="No AWS profile"):
        S3Storage("lab", profile="no-such-profile")


def test_md5_and_copy(bucket: None, tmp_path: Path) -> None:
    from pd_xray.storage import LocalStorage, StorageError, copy

    with S3Storage("lab", **KEYS) as storage:
        assert storage.md5("scans/0042/b.bin") == hashlib.md5(b"xy").hexdigest()
        result = copy(storage, "scans", LocalStorage(tmp_path), "scans")
        assert (result.copied, result.checksummed) == (2, 2)  # the folder marker isn't a file
        assert (tmp_path / "scans" / "0042" / "a.bin").read_bytes() == bytes(range(256)) * 4096
        assert copy(storage, "scans", LocalStorage(tmp_path), "scans").skipped == 2
        with pytest.raises(StorageError, match="isn't supported yet"):
            copy(LocalStorage(tmp_path), "scans", storage, "back")
