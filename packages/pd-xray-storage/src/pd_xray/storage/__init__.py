"""Storage backends (local/NFS, SFTP, S3) and file transfers for pd_xray.

`LocalStorage` is always available; `SFTPStorage` needs the `[sftp]` extra and `S3Storage` the `[s3]` extra. `copy`
copies files and folders between them.
"""

from pd_xray.storage.base import Entry, Storage, join_path, normalise_path
from pd_xray.storage.errors import (
    AuthenticationError,
    HostKeyError,
    InvalidPathError,
    StorageConnectionError,
    StorageError,
)
from pd_xray.storage.local import LocalStorage
from pd_xray.storage.s3 import S3Storage
from pd_xray.storage.sftp import SFTPStorage
from pd_xray.storage.transfer import CopyResult, IntegrityError, copy

__all__ = [
    "AuthenticationError",
    "CopyResult",
    "Entry",
    "HostKeyError",
    "IntegrityError",
    "InvalidPathError",
    "LocalStorage",
    "S3Storage",
    "SFTPStorage",
    "Storage",
    "StorageConnectionError",
    "StorageError",
    "copy",
    "join_path",
    "normalise_path",
]
