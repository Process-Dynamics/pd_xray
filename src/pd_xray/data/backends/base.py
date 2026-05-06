from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import IO, Self


@dataclass(frozen=True)
class FileInfo:
    """One entry returned by list_files().
    
    Attributes:
        path         : Full path relative to the backend root.
        size_bytes   : File size. -1 if the backend can't determine it.
        is_directory : True for directories/prefixes.
    """
    path: str
    size_bytes: int
    is_directory: bool


class StorageBackend(ABC):
    """Abstract base class for storage systems.
    
    Lifecycle:
        1. Create     : backend = SFTPBackend(host="...", port=22)
        2. Connect    : backend.connect(username="user", password="pass")
        3. Use        : files = backend.list_files("/data/raw/", "*.edf")
        4. Disconnect : backend.disconnect()
    
    Or use as context manager (preferred):
        with SFTPBackend(host="...", port=22) as backend:
            backend.connect(username="user", password="pass")
            files = backend.list_files(...)
    
    The constructor takes connection parameters. The connect() method takes credentials. This split exists because
    parameters come from YAML config but credentials come from runtime prompts or environment variables.
    """

    @abstractmethod
    def connect(
        self,
        **credentials
    ) -> None:
        """Open the connection. Credentials vary by backend.

        LocalBackend.connect() takes nothing (or validates root exists.)
        SFTPBackend.connect(username=, password=) opens SSH transport.
        SMBBackend.connect(username=, password=) opens SMB session.
        """
        ...
    
    @abstractmethod
    def disconnect(self) -> None:
        """Release all resources."""
        ...

    @property
    @abstractmethod
    def isconnected(self) -> bool:
        """Whether this backend is currently connected and usable."""
        ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    @abstractmethod
    def list_files(
        self,
        prefix: str = "",
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[FileInfo]:
        """List files matching a glob pattern under prefix.

        Args:
            prefix    : Directory or S3 prefix to search under.
            pattern   : Glob pattern (e.g., "proj_*.edf"). Applied to filenames only.
            recursive : Whether to search subdirectories.
        
        Returns:
            Sorted list of full paths/keys matching the pattern.
        """
        ...

    @abstractmethod
    def read_bytes(
        self,
        path: str,
    ) -> bytes:
        """Read entire file contents as bytes.

        Use only for small files (metadata, configs). For large files, use read_file_to_local() and let the format
        reader handle streaming.
        """
        ...

    @abstractmethod
    def read_file_to_local(
        self,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Download a single file to local filesystem.

        Args:
            remote_path : Path/key on the remote storage.
            local_path  : Destination path on local filesystem.
        """
        ...

    @abstractmethod
    def write_file_from_local(
        self,
        local_path: str,
        remote_path: str,
    ) -> None:
        """Upload a single file from local filesystem.

        Args:
            local_path  : Source path on local filesystem.
            remote_path : Destination path/key on the remote storage.
        """
        ...
    
    @abstractmethod
    def exists(
        self,
        path: str,
    ) -> bool:
        """Check if a file or prefix exists."""
        ...
    
    @abstractmethod
    def file_size(
        self,
        path: str
    ) -> int:
        """Return file size in bytes."""
        ...

    @abstractmethod
    def open_fileobj(self, path: str) -> AbstractContextManager[IO[bytes]]:
        """Return a seekable, read-only binary file-like object for the given path.

        Must be used as a context manager. The object is valid only inside the
        with-block; behaviour outside it is undefined.

        Usage::

            with backend.open_fileobj("scan.h5") as f:
                with reader.lazy_open(f) as arr:
                    frame = arr[0]
        """
        ...
