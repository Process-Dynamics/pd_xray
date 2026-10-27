"""S3 buckets: AWS or S3-compatible services (MinIO, Synology C2, Ceph, …). Needs the `[s3]` extra (boto3).

Folders are key prefixes ending in "/". boto3 clients are thread-safe, so one client serves all threads.
"""

import base64
import contextlib
import hashlib
import logging
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

from pd_xray.core import import_optional
from pd_xray.storage.base import PART_SUFFIX, Entry, Storage, Upload, name_of, normalise_path, part_name
from pd_xray.storage.errors import AuthenticationError, StorageConnectionError, StorageError

logger = logging.getLogger(__name__)
T = TypeVar("T")

_AUTH_CODES = {"InvalidAccessKeyId", "SignatureDoesNotMatch", "AccessDenied", "403", "ExpiredToken", "InvalidToken"}
_MISSING_CODES = {"NoSuchKey", "NoSuchBucket", "404", "NotFound", "NoSuchUpload"}

MIN_PART_BYTES = 8 << 20
"""Smallest multipart upload part; smaller files are uploaded in one request. S3 allows at most 10 000 parts, so
parts of files over 80 GB are bigger (10 MB for 100 GB)."""
MAX_PARTS = 10_000
PARALLEL_PARTS = 4
"""Parts uploaded at once per file."""


class S3Storage(Storage):
    """A bucket, or a prefix inside one.

    Credentials: give `access_key` and `secret_key` (plus `session_token` for temporary keys). When they're given, only
    they are used. Otherwise `profile` from ~/.aws, or boto3's usual chain (environment variables, ~/.aws, instance
    role) is used.

    Uploaded files keep their original modification time in the object metadata field "mtime" (as rclone does), which
    `stat` reports; `list` reports S3's upload time.

    Args:
        bucket: Bucket name.
        prefix: Folder inside the bucket that paths are relative to.
        endpoint_url: Service URL for S3-compatible services, e.g. "https://minio.lab:9000". None for AWS.
        region: Region, e.g. "eu-west-2". None uses the profile's or the service default.
        access_key: Access key id.
        secret_key: Secret access key.
        session_token: Session token of temporary credentials.
        profile: Profile in ~/.aws/config, used when no keys are given.
        timeout: Seconds to wait for a connection.
    """

    list_dates_exact = False

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        profile: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        boto3 = import_optional("boto3", "s3", "pd-xray-storage")
        config_module = import_optional("botocore.config", "s3", "pd-xray-storage")
        self._exceptions = import_optional("botocore.exceptions", "s3", "pd-xray-storage")

        self.bucket, self.prefix, self.endpoint_url = bucket, normalise_path(prefix), endpoint_url
        if (access_key is None) != (secret_key is None):
            raise AuthenticationError("Give both access_key and secret_key, or neither.")
        try:
            if access_key is not None:
                session = boto3.session.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    aws_session_token=session_token,
                    region_name=region,
                )
            else:
                session = boto3.session.Session(profile_name=profile, region_name=region)
        except self._exceptions.ProfileNotFound:
            raise AuthenticationError(f"No AWS profile named '{profile}' in ~/.aws.") from None
        config = config_module.Config(
            connect_timeout=timeout,
            read_timeout=60,
            retries={"max_attempts": 3, "mode": "standard"},
            max_pool_connections=32,
            s3={"addressing_style": "path"} if endpoint_url else None,  # most S3-compatible services need it
            request_checksum_calculation="when_required",  # uploads send Content-MD5, which every service accepts
        )
        self._client = session.client("s3", endpoint_url=endpoint_url, config=config)

    @property
    def url(self) -> str:
        where = f" at {self.endpoint_url}" if self.endpoint_url else ""
        return f"s3://{self.bucket}/{self.prefix}{where}".replace("/ at", " at")

    def _key(self, path: str) -> str:
        return "/".join(part for part in (self.prefix, path) if part)

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        """Call a client method on the bucket, with errors translated."""
        return self._guard(lambda: getattr(self._client, method)(Bucket=self.bucket, **kwargs), path)

    def _guard(self, work: Callable[[], T], path: str) -> T:
        """Run `work`, translating botocore errors about `path` into storage errors."""
        exceptions = self._exceptions
        try:
            return work()
        except exceptions.NoCredentialsError:
            raise AuthenticationError(
                "No S3 credentials found. Give access_key and secret_key, or a profile."
            ) from None
        except (exceptions.EndpointConnectionError, exceptions.ConnectTimeoutError) as exc:
            raise StorageConnectionError(f"Can't reach the S3 service: {exc}.") from None
        except exceptions.ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            message = exc.response.get("Error", {}).get("Message") or code
            if code in _AUTH_CODES:
                raise AuthenticationError(
                    f"S3 refused access to {self.url}: {message}. Check the keys and the bucket's permissions."
                ) from None
            if code == "NoSuchBucket":
                raise FileNotFoundError(f"Bucket '{self.bucket}' doesn't exist.") from None
            if code in _MISSING_CODES:
                raise FileNotFoundError(f"No such file or folder: {path!r} in {self.url}") from None
            raise StorageError(f"S3 error for {path!r} in {self.url}: {message}") from None

    def _head_bucket(self) -> None:
        try:
            self._call("head_bucket", "")
        except FileNotFoundError:
            raise FileNotFoundError(f"Bucket '{self.bucket}' doesn't exist.") from None

    def _folder(self, path: str) -> str:
        """Key prefix of the folder `path` ("" for the bucket root)."""
        key = self._key(path)
        return f"{key}/" if key else ""

    def stat(self, path: str = "") -> Entry:
        path = normalise_path(path)
        if path:
            try:
                head = self._call("head_object", path, Key=self._key(path))
                return Entry(name_of(path), path, False, head["ContentLength"], _modified(head))
            except FileNotFoundError:
                pass
        elif not self.prefix:
            self._head_bucket()
            return Entry("", "", True)
        if self._call("list_objects_v2", path, Prefix=self._folder(path), MaxKeys=1).get("KeyCount", 0):
            return Entry(name_of(path), path, True)
        raise FileNotFoundError(f"No such file or folder: {path!r} in {self.url}")

    def list(self, path: str = "") -> list[Entry]:
        path = normalise_path(path)
        paginator = self._client.get_paginator("list_objects_v2")
        pages = self._guard(
            lambda: list(paginator.paginate(Bucket=self.bucket, Prefix=self._folder(path), Delimiter="/")), path
        )
        entries, found = [], False
        for page in pages:
            found = found or bool(page.get("KeyCount", 0))
            for common in page.get("CommonPrefixes", []):
                name = common["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                entries.append(Entry(name, f"{path}/{name}".lstrip("/"), True))
            for item in page.get("Contents", []):
                name = item["Key"].rsplit("/", 1)[-1]
                if name:  # "folder/" marker objects made by consoles
                    entries.append(
                        Entry(name, f"{path}/{name}".lstrip("/"), False, item["Size"], item["LastModified"].timestamp())
                    )
        if not found:
            if path or self.prefix:
                if not self.stat(path).is_dir:  # FileNotFoundError if missing
                    raise NotADirectoryError(f"Not a folder: {path!r} in {self.url}")
            else:
                self._head_bucket()  # an empty bucket is fine; a missing one raises
        return entries

    def read_range(self, path: str, offset: int, size: int) -> bytes:
        path = normalise_path(path)
        key = self._key(path)
        response = self._call("get_object", path, Key=key, Range=f"bytes={offset}-{offset + size - 1}")
        return self._guard(lambda: bytes(response["Body"].read()), path)

    def md5(self, path: str) -> str | None:
        """The object's ETag when it is a plain MD5: single-part uploads without KMS or customer-key encryption."""
        path = normalise_path(path)
        head = self._call("head_object", path, Key=self._key(path))
        etag = str(head.get("ETag", "")).strip('"')
        plain = head.get("ServerSideEncryption") in (None, "AES256") and "SSECustomerAlgorithm" not in head
        return etag if plain and len(etag) == 32 and "-" not in etag else None

    def upload(self, path: str, size: int, modified: float) -> Upload:
        return _S3Upload(self, normalise_path(path), size, modified)

    def remove(self, path: str) -> None:
        path = normalise_path(path)
        self.stat(path)  # deleting a missing key isn't an error in S3
        if path.endswith(PART_SUFFIX):  # marker of an unfinished upload: end the upload too
            parent, _, marker = path.rpartition("/")
            name = marker[1 : -len(PART_SUFFIX)].rsplit(".", 1)[0]  # ".<name>.<size>-<mtime>.pd-xray-part"
            upload_id = self._read(path)
            with contextlib.suppress(FileNotFoundError):
                self._call(
                    "abort_multipart_upload", path, Key=self._key(f"{parent}/{name}".lstrip("/")), UploadId=upload_id
                )
        self._call("delete_object", path, Key=self._key(path))

    def _read(self, path: str) -> str:
        """A small object's content as text."""
        response = self._call("get_object", path, Key=self._key(path))
        return self._guard(lambda: bytes(response["Body"].read()), path).decode().strip()

    def close(self) -> None:
        self._client.close()


def _modified(head: dict[str, Any]) -> float:
    """The "mtime" metadata of an object if valid, else its upload time."""
    try:
        return float(head.get("Metadata", {})["mtime"])
    except (KeyError, ValueError):
        return float(head["LastModified"].timestamp())


def _content_md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


class _S3Upload(Upload):
    """Files up to one part: a single request on commit. Larger: a multipart upload with `PARALLEL_PARTS` parts in
    flight. Its id is kept in the marker object, so an interrupted upload resumes after its completed leading parts.
    Every request carries Content-MD5, so S3 rejects parts corrupted on the way."""

    def __init__(self, storage: S3Storage, path: str, size: int, modified: float) -> None:
        self._storage, self._path, self._key = storage, path, storage._key(path)
        self._metadata = {"mtime": f"{modified:.6f}"}
        self._part_bytes = max(MIN_PART_BYTES, -(-size // MAX_PARTS))
        self._buffer = bytearray()
        self._parts: dict[int, str] = {}  # part number -> ETag
        self._pending: deque[Future[tuple[int, str]]] = deque()
        self._pool: ThreadPoolExecutor | None = None
        self.offset, self.part_path, self._upload_id = 0, None, None
        if size <= self._part_bytes:
            return
        parent, _, name = path.rpartition("/")
        self._marker = f"{parent}/{part_name(name, size, modified)}".lstrip("/")
        self._upload_id = self._resume() or self._create()
        self._pool = ThreadPoolExecutor(PARALLEL_PARTS, thread_name_prefix="s3-part")

    def _resume(self) -> str | None:
        """Id of the interrupted upload, with `offset` and `_parts` set to its completed leading parts."""
        storage = self._storage
        try:
            upload_id = storage._read(self._marker)
            paginator = storage._client.get_paginator("list_parts")
            pages = storage._guard(
                lambda: list(paginator.paginate(Bucket=storage.bucket, Key=self._key, UploadId=upload_id)), self._path
            )
        except FileNotFoundError:  # no marker, or the upload was ended (e.g. by a bucket lifecycle rule)
            return None
        done = {part["PartNumber"]: part for page in pages for part in page.get("Parts", [])}
        while (number := len(self._parts) + 1) in done and done[number]["Size"] == self._part_bytes:
            self._parts[number] = done[number]["ETag"]
        self.offset = len(self._parts) * self._part_bytes
        logger.info("Resuming upload of %s after %d parts", self._path, len(self._parts))
        return upload_id

    def _create(self) -> str:
        storage = self._storage
        response = storage._call("create_multipart_upload", self._path, Key=self._key, Metadata=self._metadata)
        upload_id = str(response["UploadId"])
        storage._call("put_object", self._marker, Key=storage._key(self._marker), Body=upload_id.encode())
        return upload_id

    def write(self, data: bytes) -> None:
        self._buffer += data
        self.offset += len(data)
        while self._upload_id is not None and len(self._buffer) >= self._part_bytes:
            self._send(bytes(self._buffer[: self._part_bytes]))
            del self._buffer[: self._part_bytes]

    def _send(self, body: bytes) -> None:
        assert self._pool is not None
        while len(self._pending) >= PARALLEL_PARTS:
            self._collect()
        number = len(self._parts) + len(self._pending) + 1
        self._pending.append(self._pool.submit(self._upload_part, number, body))

    def _upload_part(self, number: int, body: bytes) -> tuple[int, str]:
        response = self._storage._call(
            "upload_part",
            self._path,
            Key=self._key,
            UploadId=self._upload_id,
            PartNumber=number,
            Body=body,
            ContentMD5=_content_md5(body),
        )
        return number, str(response["ETag"])

    def _collect(self) -> None:
        number, etag = self._pending.popleft().result()
        self._parts[number] = etag

    def commit(self) -> None:
        storage, body = self._storage, bytes(self._buffer)
        if self._upload_id is None:
            md5 = _content_md5(body)
            storage._call("put_object", self._path, Key=self._key, Body=body, ContentMD5=md5, Metadata=self._metadata)
            return
        if body:
            self._send(body)
        while self._pending:
            self._collect()
        parts = [{"PartNumber": number, "ETag": etag} for number, etag in sorted(self._parts.items())]
        storage._call(
            "complete_multipart_upload",
            self._path,
            Key=self._key,
            UploadId=self._upload_id,
            MultipartUpload={"Parts": parts},
        )
        storage._call("delete_object", self._marker, Key=storage._key(self._marker))
        self.close()

    def discard(self) -> None:
        self.close()
        if self._upload_id is not None:
            with contextlib.suppress(FileNotFoundError):
                self._storage.remove(self._marker)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)  # parts being sent finish; they count on resume
            self._pool = None
