import fnmatch
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import boto3
import botocore.config
import botocore.exceptions

from pd_xray.data.backends.base import StorageBackend, FileInfo
from pd_xray.core.logging import get_logger

logger = get_logger(__name__)

_BLOCK_SIZE = 2 * 1024 * 1024  # 2 MB read-ahead blocks


class _S3RangeFile:
    """Seekable, read-only file-like object backed by boto3 range requests.

    Reads are served from 2 MB cached blocks so that the many small sequential
    reads a format reader makes when opening a file result in a small number of
    S3 API calls.
    """

    def __init__(self, client: Any, bucket: str, key: str) -> None:
        self._client = client
        self._bucket = bucket
        self._key = key
        self._pos = 0
        response = client.head_object(Bucket=bucket, Key=key)
        self._size: int = response["ContentLength"]
        self._cache: dict[int, bytes] = {}

    def _fetch_block(self, block_idx: int) -> bytes:
        cached = self._cache.get(block_idx)
        if cached is not None:
            return cached
        start = block_idx * _BLOCK_SIZE
        end = min(start + _BLOCK_SIZE - 1, self._size - 1)
        response = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
            Range=f"bytes={start}-{end}",
        )
        data: bytes = response["Body"].read()
        self._cache[block_idx] = data
        return data

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        end = self._size if size == -1 else min(self._pos + size, self._size)
        parts: list[bytes] = []
        pos = self._pos
        while pos < end:
            block_idx = pos // _BLOCK_SIZE
            offset = pos % _BLOCK_SIZE
            block = self._fetch_block(block_idx)
            chunk = block[offset : offset + (end - pos)]
            parts.append(chunk)
            pos += len(chunk)
        self._pos = pos
        return b"".join(parts)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:
        self._cache.clear()

    def __enter__(self) -> "_S3RangeFile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class S3Backend(StorageBackend):
    """Storage backend for S3-compatible object stores (AWS, MinIO, Ceph, etc.).

    Constructor args:
        bucket           : S3 bucket name.
        endpoint_url     : Custom endpoint for non-AWS stores (e.g. "http://localhost:9000").
                           Pass None to use the default AWS endpoint.
        region_name      : AWS region (e.g. "us-east-1"). Optional for custom endpoints.
        prefix           : Optional key prefix treated as a root. All paths passed to other
                           methods are relative to this prefix.
        addressing_style : "path" (default) or "virtual". Custom endpoints almost always
                           require "path". AWS S3 uses "virtual" by default, but also
                           accepts "path". With "virtual", boto3 rewrites the URL to
                           bucket.host.com which breaks most non-AWS endpoints.

    Credentials are passed at connect() time via keyword arguments:
        aws_access_key_id     : Access key.
        aws_secret_access_key : Secret key.
        aws_session_token     : Session token (optional, for temporary credentials).
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        prefix: str = "",
        addressing_style: str = "path",
    ) -> None:
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._prefix = prefix.strip("/")
        self._addressing_style = addressing_style
        self._client: Any = None
        self._credentials: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, **credentials: Any) -> None:
        """Configure the S3 client. No network call is made here.

        Keyword args:
            aws_access_key_id     : Access key.
            aws_secret_access_key : Secret key.
            aws_session_token     : Session token (optional).
        """
        # Only pass credentials that were actually supplied so boto3 does not
        # fall through to slow credential chain lookups (IMDS, etc.) for keys
        # the caller never intended to provide.
        self._credentials = {
            k: v for k, v in {
                "aws_access_key_id": credentials.get("aws_access_key_id"),
                "aws_secret_access_key": credentials.get("aws_secret_access_key"),
                "aws_session_token": credentials.get("aws_session_token"),
            }.items()
            if v is not None
        }
        self._client = boto3.client(
            "s3",
            endpoint_url=self._endpoint_url,
            region_name=self._region_name,
            config=botocore.config.Config(
                s3={"addressing_style": self._addressing_style},
                signature_version="s3v4",
                connect_timeout=10,
                read_timeout=180,  # three minutes
                retries={"max_attempts": 3, "mode": "standard"},
            ),
            **self._credentials,
        )
        logger.debug(f"S3 client ready for bucket '{self._bucket}' at {self._endpoint_url!r}")

    def disconnect(self) -> None:
        self._client = None
        self._credentials = {}

    @property
    def isconnected(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _full_key(self, path: str) -> str:
        path = path.lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{path}"
        return path

    def _strip_prefix(self, key: str) -> str:
        if self._prefix and key.startswith(self._prefix + "/"):
            return key[len(self._prefix) + 1 :]
        return key

    def _require_connected(self) -> None:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # File listing
    # ------------------------------------------------------------------

    def list_files(
        self,
        prefix: str = "",
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[FileInfo]:
        """List objects under prefix matching a glob pattern.

        Returns only key names and sizes from the S3 listing API.
        No file content is downloaded.

        Args:
            prefix    : Key prefix to search under (relative to the backend root).
            pattern   : Glob pattern applied to the filename part of each key.
            recursive : If False, only return objects at the immediate prefix level.
        """
        self._require_connected()

        search_prefix = self._full_key(prefix)
        if search_prefix and not search_prefix.endswith("/"):
            search_prefix += "/"

        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self._bucket,
            Prefix=search_prefix,
            Delimiter="" if recursive else "/",
        )

        results: list[FileInfo] = []
        for page in pages:
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                relative = self._strip_prefix(key)
                name = relative.split("/")[-1]
                if not name or name.startswith("."):
                    continue
                if fnmatch.fnmatch(name, pattern):
                    results.append(FileInfo(
                        path=relative,
                        size_bytes=obj["Size"],
                        is_directory=False,
                    ))
            for cp in page.get("CommonPrefixes", []):
                key = cp["Prefix"].rstrip("/")
                relative = self._strip_prefix(key)
                name = relative.split("/")[-1]
                if not name.startswith("."):
                    results.append(FileInfo(
                        path=relative + "/",
                        size_bytes=-1,
                        is_directory=True,
                    ))

        return sorted(results, key=lambda f: f.path)

    # ------------------------------------------------------------------
    # Streaming file access
    # ------------------------------------------------------------------

    @contextmanager
    def open_fileobj(self, path: str):
        """Return a seekable, read-only file-like object for the given remote path.

        Reads are served from 2 MB cached blocks via boto3 range requests, so the
        object can be passed directly to format readers that accept file-like objects
        (e.g. h5py.File, tifffile.TiffFile) without downloading the whole file first.

        Usage::

            with backend.open_fileobj("scan.h5") as f:
                with reader.lazy_open(f) as arr:
                    frame = arr[0]
        """
        self._require_connected()
        key = self._full_key(path)
        rf = _S3RangeFile(self._client, self._bucket, key)
        try:
            yield rf
        finally:
            rf.close()

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def read_bytes(self, path: str) -> bytes:
        """Download an entire object as bytes. Use only for small files."""
        self._require_connected()
        key = self._full_key(path)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()
        except botocore.exceptions.ClientError as e:
            raise FileNotFoundError(f"Object not found: s3://{self._bucket}/{key}") from e

    def read_file_to_local(self, remote_path: str, local_path: str) -> None:
        """Download an object to local disk.

        Args:
            remote_path : Key relative to the backend root.
            local_path  : Destination path on local filesystem.
        """
        self._require_connected()
        key = self._full_key(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self._bucket, key, local_path)
            logger.debug(f"Downloaded s3://{self._bucket}/{key} -> {local_path}")
        except botocore.exceptions.ClientError as e:
            raise FileNotFoundError(f"Object not found: s3://{self._bucket}/{key}") from e

    def write_file_from_local(self, local_path: str, remote_path: str) -> None:
        """Upload a local file to S3.

        Args:
            local_path  : Source path on local filesystem.
            remote_path : Destination key relative to the backend root.
        """
        self._require_connected()
        if not Path(local_path).is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        key = self._full_key(remote_path)
        self._client.upload_file(local_path, self._bucket, key)
        logger.debug(f"Uploaded {local_path} -> s3://{self._bucket}/{key}")

    # ------------------------------------------------------------------
    # Existence and size
    # ------------------------------------------------------------------

    def exists(self, path: str) -> bool:
        self._require_connected()
        key = self._full_key(path)
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False

    def file_size(self, path: str) -> int:
        self._require_connected()
        key = self._full_key(path)
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
            return response["ContentLength"]
        except botocore.exceptions.ClientError as e:
            raise FileNotFoundError(f"Object not found: s3://{self._bucket}/{key}") from e
