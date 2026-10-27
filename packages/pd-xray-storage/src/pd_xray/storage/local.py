"""Local folders, including NFS/SMB mounts."""

import os
from pathlib import Path
from typing import BinaryIO

from pd_xray.storage.base import Entry, Storage, Upload, name_of, normalise_path, part_name
from pd_xray.storage.errors import InvalidPathError


class LocalStorage(Storage):
    """A local folder (or a mounted network share).

    Args:
        root: The folder. Paths can't leave it, including through symlinks.

    Raises:
        NotADirectoryError: If `root` isn't an existing folder.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a folder: {self.root}")

    @property
    def url(self) -> str:
        return self.root.as_uri()

    def full_path(self, path: str) -> Path:
        """Absolute local path of `path`, checked to stay inside the root (also through symlinks)."""
        full = (self.root / normalise_path(path)).resolve()
        if not full.is_relative_to(self.root):
            raise InvalidPathError(f"'{path}' leads outside {self.root} (symlink).")
        return full

    def stat(self, path: str = "") -> Entry:
        path = normalise_path(path)
        full = self.full_path(path)
        result = full.stat()  # FileNotFoundError if missing
        is_dir = full.is_dir()
        return Entry(name_of(path), path, is_dir, None if is_dir else result.st_size, result.st_mtime)

    def list(self, path: str = "") -> list[Entry]:
        path = normalise_path(path)
        entries = []
        with os.scandir(self.full_path(path)) as scan:
            for item in scan:
                try:
                    is_dir = item.is_dir()
                    result = item.stat()
                except OSError:  # broken symlink or vanished file
                    continue
                entries.append(
                    Entry(
                        item.name,
                        f"{path}/{item.name}".lstrip("/"),
                        is_dir,
                        None if is_dir else result.st_size,
                        result.st_mtime,
                    )
                )
        return entries

    def open(self, path: str) -> BinaryIO:
        return self.full_path(path).open("rb")

    def read_range(self, path: str, offset: int, size: int) -> bytes:
        with self.full_path(path).open("rb") as file:
            return os.pread(file.fileno(), size, offset)

    def upload(self, path: str, size: int, modified: float) -> Upload:
        return _LocalUpload(self, normalise_path(path), size, modified)

    def remove(self, path: str) -> None:
        self.full_path(path).unlink()


class _LocalUpload(Upload):
    def __init__(self, storage: LocalStorage, path: str, size: int, modified: float) -> None:
        self.target, self.modified = storage.full_path(path), modified
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.part = self.target.with_name(part_name(self.target.name, size, modified))
        self.part_path = self.part.relative_to(storage.root).as_posix()
        self.offset = self.part.stat().st_size if self.part.exists() else 0
        if self.offset > size:
            self.offset = 0
        self._file = self.part.open("r+b" if self.offset else "wb", buffering=0)
        self._file.truncate(self.offset)

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.pwrite(self._file.fileno(), view, self.offset)
            self.offset += written
            view = view[written:]

    def commit(self) -> None:
        os.fsync(self._file.fileno())
        self._file.close()
        os.utime(self.part, (self.modified, self.modified))
        os.replace(self.part, self.target)

    def discard(self) -> None:
        self._file.close()
        self.part.unlink(missing_ok=True)

    def close(self) -> None:
        self._file.close()
