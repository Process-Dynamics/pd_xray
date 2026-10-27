"""The `Storage` interface shared by the local, SFTP and S3 backends.

Paths are POSIX strings relative to the storage root: "" is the root, "scans/0042/a.tif" a file below it. Leading and
trailing "/" are ignored, and ".." is refused.
"""

import io
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import BinaryIO, cast

from pd_xray.storage.errors import InvalidPathError

PART_SUFFIX = ".pd-xray-part"
BLOCK_BYTES = 256 * 1024
"""Block size of the read cache of remote files opened with `Storage.open`."""
CACHE_BLOCKS = 64
"""Blocks cached per open remote file (16 MB)."""


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

    Attributes:
        list_dates_exact: False when `list` reports a different `modified` than `stat` (S3: the upload time instead of
            the original time kept in metadata).
    """

    list_dates_exact = True

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
        return cast(BinaryIO, reader)

    @abstractmethod
    def upload(self, path: str, size: int, modified: float) -> "Upload":
        """Start writing the file `path` of `size` bytes, or resume an interrupted upload of the same source.

        Partial data is kept under a hidden ".<name>.<size>-<modified>.pd-xray-part" name next to `path` (S3: an
        unfinished multipart upload marked by an object of that name). Missing parent folders are created. Use
        `transfer.copy` rather than this directly; it also deletes partial data of older source versions.

        Args:
            path: The file to write. Replaced on `Upload.commit` if it exists.
            size: Final size in bytes.
            modified: Modification time (POSIX timestamp) the finished file gets.
        """

    @abstractmethod
    def remove(self, path: str) -> None:
        """Delete the file at `path`. Deleting partial data (see `upload`) also ends its unfinished S3 upload.

        Raises:
            FileNotFoundError: If it doesn't exist.
        """

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


def part_name(name: str, size: int, modified: float) -> str:
    """Hidden name of the partial data while uploading the file `name` (see `Storage.upload`)."""
    return f".{name}.{size}-{int(modified)}{PART_SUFFIX}"


def is_stale_part(item: str, name: str, current: str) -> bool:
    """Whether `item` (a file name) is partial data of `name` other than `current`, from an older source version."""
    return item != current and item.startswith(f".{name}.") and item.endswith(PART_SUFFIX)


class Upload(ABC):
    """A file being written by `Storage.upload`. Not thread-safe: one thread writes.

    Nothing appears at the destination path until `commit`. Close without committing to keep the partial data for
    resuming, or `discard` it.

    Attributes:
        offset: Bytes already written, by an interrupted earlier upload at first. `write` continues from here.
        part_path: Storage path of the partial file, readable with `read_range` up to `offset`. None when the partial
            data can't be read back (S3).
    """

    offset: int
    part_path: str | None

    @abstractmethod
    def write(self, data: bytes) -> None:
        """Append `data` at `offset`."""

    @abstractmethod
    def commit(self) -> None:
        """Finish: the file appears at its path (replacing what was there) with its modification time."""

    @abstractmethod
    def discard(self) -> None:
        """Delete the partial data."""

    def close(self) -> None:  # noqa: B027  (optional for backends without open handles)
        """Release handles, keeping the partial data."""

    def __enter__(self) -> "Upload":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()


class RangeReader(io.RawIOBase):
    """Read-only, seekable raw file built on a `read_range(offset, size) -> bytes` function, for remote files.

    Small reads come from a cache of aligned blocks (least recently used dropped first), so the many small, scattered
    reads of format libraries (HDF5 chunk indexes, TIFF headers) cost few round trips. Reads of 4 blocks or more go
    straight to `read_range` (e.g. frame data). Thread-safe.
    """

    def __init__(
        self,
        size: int,
        read_range: Callable[[int, int], bytes],
        name: str,
        block_bytes: int = BLOCK_BYTES,
        cache_blocks: int = CACHE_BLOCKS,
    ) -> None:
        super().__init__()
        self._size, self._read_range, self.name, self._position = size, read_range, name, 0
        self._block, self._keep = block_bytes, cache_blocks
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        with self._lock:
            base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}[whence]
            if base + offset < 0:
                raise ValueError(f"Negative seek position {base + offset}.")
            self._position = base + offset
            return self._position

    def readinto(self, buffer: memoryview | bytearray) -> int:  # type: ignore[override]
        with self._lock:
            size = min(memoryview(buffer).nbytes, self._size - self._position)
            if size <= 0:
                return 0
            start = self._position
            data = self._read_range(start, size) if size >= 4 * self._block else self._cached(start, size)
            memoryview(buffer).cast("B")[: len(data)] = data  # any buffer, e.g. a NumPy array (tifffile)
            self._position += len(data)
            return len(data)

    def _cached(self, start: int, size: int) -> bytes:
        block, cache = self._block, self._cache
        first, last = start // block, (start + size - 1) // block
        missing = [i for i in range(first, last + 1) if i not in cache]
        if missing:  # one request from the first to the last missing block
            low, high = missing[0] * block, min((missing[-1] + 1) * block, self._size)
            data = self._read_range(low, high - low)
            for i in range(missing[0], missing[-1] + 1):
                cache[i] = data[i * block - low : (i + 1) * block - low]
        for i in range(first, last + 1):
            cache.move_to_end(i)
        joined = b"".join(cache[i] for i in range(first, last + 1))
        while len(cache) > self._keep:
            cache.popitem(last=False)
        return joined[start - first * block : start - first * block + size]
