"""The `Storage` interface shared by the local, SFTP and S3 backends.

Paths are POSIX strings relative to the storage root: "" is the root, "scans/0042/a.tif" a file below it. Leading and
trailing "/" are ignored, and ".." is refused.
"""

import io
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import BinaryIO

from pd_xray.storage.errors import InvalidPathError


@dataclass(frozen=True)
class Entry:
    """A file or folder.

    Attributes:
        name: Last path component ("" for the root).
        path: Path relative to the storage root.
        is_dir: Whether it's a folder (an S3 prefix counts as a folder).
        size: Size in bytes, None for folders.
        modified: Last modification as a POSIX timestamp, None if unknown.
    """

    name: str
    path: str
    is_dir: bool
    size: int | None = None
    modified: float | None = None


def normalise_path(path: str) -> str:
    """`path` without leading/trailing "/", empty and "." components.

    Raises:
        InvalidPathError: For ".." components.
    """
    parts = [part for part in path.replace("\\", "/").split("/") if part not in ("", ".")]
    if ".." in parts:
        raise InvalidPathError(f"'..' isn't allowed in storage paths, got '{path}'.")
    return "/".join(parts)


def join_path(*parts: str) -> str:
    """Join relative storage paths, e.g. join_path("a", "b/c") -> "a/b/c"."""
    return normalise_path("/".join(parts))


def name_of(path: str) -> str:
    """Last component of a normalised path ("" for the root)."""
    return path.rsplit("/", 1)[-1]


class Storage(ABC):
    """A place that holds files: a local folder, an SFTP server or an S3 bucket.

    Connections open on first use and are reopened once if dropped. Methods are thread-safe. Close the storage (or use
    it as a context manager) to release connections.
    """

    @property
    @abstractmethod
    def url(self) -> str:
        """Where the root is, without credentials, e.g. "sftp://user@nas:22/volume1/data" or "s3://bucket/prefix"."""

    @abstractmethod
    def stat(self, path: str = "") -> Entry:
        """The file or folder at `path`.

        Raises:
            FileNotFoundError: If nothing is there.
            StorageConnectionError, AuthenticationError, HostKeyError: If the connection fails.
        """

    @abstractmethod
    def list(self, path: str = "") -> list[Entry]:
        """Entries directly inside the folder at `path`, in no particular order.

        Raises:
            FileNotFoundError: If the folder doesn't exist.
            NotADirectoryError: If `path` is a file.
        """

    @abstractmethod
    def read_range(self, path: str, offset: int, size: int) -> bytes:
        """Up to `size` bytes (`size` > 0) of the file at `path` from byte `offset`, which must lie inside the file.

        Thread-safe, so several ranges can be read in parallel.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """

    def open(self, path: str) -> BinaryIO:
        """The file at `path`, opened for reading. Seekable: reads fetch only the requested byte range.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            IsADirectoryError: If `path` is a folder.
        """
        entry = self.stat(path)
        if entry.is_dir:
            raise IsADirectoryError(f"Is a folder: {path!r} in {self.url}")
        reader = RangeReader(entry.size or 0, lambda offset, size: self.read_range(entry.path, offset, size), path)
        return io.BufferedReader(reader, 1 << 20)

    def md5(self, path: str) -> str | None:
        """MD5 hex digest of the file at `path` if the server already knows it (no reading), else None."""
        return None

    def exists(self, path: str) -> bool:
        """Whether a file or folder exists at `path`."""
        try:
            self.stat(path)
        except FileNotFoundError:
            return False
        return True

    def check(self) -> None:
        """Connect and read the root, raising the specific error if that fails. For "Test connection" buttons."""
        if not self.stat("").is_dir:
            raise NotADirectoryError(f"The root of {self.url} isn't a folder.")

    def close(self) -> None:  # noqa: B027  (optional for backends without connections)
        """Release connections. The storage can't be used afterwards."""

    def __enter__(self) -> "Storage":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.url!r})"


class RangeReader(io.RawIOBase):
    """Read-only, seekable raw file built on a `read_range(offset, size) -> bytes` function. Wrap it in
    `io.BufferedReader` so small reads don't each cost a round trip."""

    def __init__(self, size: int, read_range: Callable[[int, int], bytes], name: str) -> None:
        super().__init__()
        self._size, self._read_range, self.name, self._position = size, read_range, name, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}[whence]
        if base + offset < 0:
            raise ValueError(f"Negative seek position {base + offset}.")
        self._position = base + offset
        return self._position

    def readinto(self, buffer: memoryview | bytearray) -> int:  # type: ignore[override]
        size = min(len(buffer), self._size - self._position)
        if size <= 0:
            return 0
        data = self._read_range(self._position, size)
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)
