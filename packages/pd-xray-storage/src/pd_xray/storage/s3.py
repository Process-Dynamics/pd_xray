"""S3 buckets: AWS or S3-compatible services (MinIO, Synology C2, Ceph, …). Needs the `[s3]` extra (boto3).

Folders are key prefixes ending in "/". boto3 clients are thread-safe, so one client serves all threads.
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pd_xray.core import import_optional
from pd_xray.storage.base import Entry, Storage, name_of, normalise_path
from pd_xray.storage.errors import AuthenticationError, StorageConnectionError, StorageError

logger = logging.getLogger(__name__)
T = TypeVar("T")

_AUTH_CODES = {"InvalidAccessKeyId", "SignatureDoesNotMatch", "AccessDenied", "403", "ExpiredToken", "InvalidToken"}
_MISSING_CODES = {"NoSuchKey", "NoSuchBucket", "404", "NotFound"}


class S3Storage(Storage):
    """A bucket, or a prefix inside one.

    Credentials: give `access_key` and `secret_key` (plus `session_token` for temporary keys). When they're given, only
    they are used. Otherwise `profile` from ~/.aws, or boto3's usual chain (environment variables, ~/.aws, instance
    role) is used.

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
                return Entry(name_of(path), path, False, head["ContentLength"], head["LastModified"].timestamp())
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

    def close(self) -> None:
        self._client.close()
