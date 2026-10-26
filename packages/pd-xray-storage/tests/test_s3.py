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


def test_md5_and_download(bucket: None, tmp_path: Path) -> None:
    from pd_xray.storage import LocalStorage, copy

    with S3Storage("lab", **KEYS) as storage:
        assert storage.md5("scans/0042/b.bin") == hashlib.md5(b"xy").hexdigest()
        result = copy(storage, "scans", LocalStorage(tmp_path), "scans")
        assert (result.copied, result.checksummed) == (2, 2)  # the folder marker isn't a file
        assert (tmp_path / "scans" / "0042" / "a.bin").read_bytes() == bytes(range(256)) * 4096
        assert copy(storage, "scans", LocalStorage(tmp_path), "scans").skipped == 2


def test_upload_cancel_resume(bucket: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from pd_xray.core import CancelToken, OperationCancelledError, Progress
    from pd_xray.storage import LocalStorage, copy, s3, transfer

    monkeypatch.setattr(transfer, "CHUNK_BYTES", 1000)
    monkeypatch.setattr(s3, "MIN_PART_BYTES", 2000)  # 10 240 bytes -> 6 parts
    monkeypatch.setattr(moto.s3.models, "S3_UPLOAD_PART_MIN_SIZE", 1)
    data = bytes(range(256)) * 40
    (tmp_path / "scan").mkdir()
    (tmp_path / "scan" / "a.bin").write_bytes(data)
    (tmp_path / "scan" / "small.bin").write_bytes(b"small")  # one request, no multipart
    os.utime(tmp_path / "scan" / "a.bin", (1_700_000_000, 1_700_000_000))
    local = LocalStorage(tmp_path)
    stop = CancelToken()

    def cancel_after_some(update: Progress) -> None:
        if update.current >= 7000:
            stop.cancel()

    with S3Storage("lab", **KEYS) as storage:
        client = storage._client
        storage.upload("up/scan/.a.bin.99-1.pd-xray-part", 99, 1).commit()  # placeholder object
        old = client.create_multipart_upload(Bucket="lab", Key="up/scan/a.bin")["UploadId"]
        client.put_object(Bucket="lab", Key="up/scan/.a.bin.99-1.pd-xray-part", Body=old.encode())  # older version
        with pytest.raises(OperationCancelledError):
            copy(local, "scan/a.bin", storage, "up/scan/a.bin", progress=cancel_after_some, cancel=stop)
        assert not storage.exists("up/scan/a.bin") and not storage.exists("up/scan/.a.bin.99-1.pd-xray-part")
        [upload] = client.list_multipart_uploads(Bucket="lab")["Uploads"]  # the older one was ended
        assert upload["UploadId"] != old

        result = copy(local, "scan", storage, "up/scan")
        assert result.copied == 2 and result.bytes_copied < len(data) + 5  # resumed
        assert client.get_object(Bucket="lab", Key="up/scan/a.bin")["Body"].read() == data
        assert storage.stat("up/scan/a.bin").modified == 1_700_000_000
        assert "Uploads" not in client.list_multipart_uploads(Bucket="lab")
        assert sorted(e.name for e in storage.list("up/scan")) == ["a.bin", "small.bin"]  # marker gone
        assert copy(local, "scan", storage, "up/scan").skipped == 2

        result = copy(storage, "up", storage, "copied", verify=True)  # S3 -> S3
        assert (result.copied, result.checksummed) == (2, 2)
        assert copy(storage, "copied/scan", local, "back", verify=True).copied == 2
        assert (tmp_path / "back" / "a.bin").read_bytes() == data
        assert int((tmp_path / "back" / "a.bin").stat().st_mtime) == 1_700_000_000
