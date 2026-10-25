"""Local folders, including NFS/SMB mounts."""

import os
from pathlib import Path
from typing import BinaryIO

from pd_xray.storage.base import Entry, Storage, name_of, normalise_path
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
