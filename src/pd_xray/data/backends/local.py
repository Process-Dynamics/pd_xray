from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pd_xray.data.backends.base import StorageBackend, FileInfo
from pd_xray.core.logging import get_logger

logger = get_logger(__name__)


class LocalBackend(StorageBackend):
    """Storage backend for local filesystem.
    
    Constructor args:
        root : The root directory. All paths passed to other methods are relative to this root.
        
    """

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._connected = False
    
    @property
    def root(self) -> Path:
        return self._root

    def connect(self, **credentials: Any) -> None:
        # No credentials needed for local, just check the root exists and is a directory.
        if not self._root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self._root}")
        if not self._root.is_dir():
            raise NotADirectoryError(f"Root is not a directory: {self._root}")
        self._connected = True

    def disconnect(self):
        self._connected = False
    
    @property
    def isconnected(self) -> bool:
        return self._connected
    
    def list_files(
        self,
        prefix: str = "",
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[FileInfo]:
        full_path = self._root / prefix
        if not full_path.exists() or not full_path.is_dir():
            raise FileNotFoundError(f"Prefix path does not exist or is not a directory: {full_path}")
        matching_files = sorted(full_path.glob(pattern))
        return [
            FileInfo(
                path=str(m.relative_to(self._root)),
                size_bytes=m.stat().st_size if m.is_file() else -1,
                is_directory=m.is_dir(),
            )
            for m in matching_files
            if not m.name.startswith(".")  # Skip hidden files
        ]
    
    def exists(self, path: str) -> bool:
        return (self._root / path).exists()
    
    def file_size(self, path: str) -> int:
        full_path = self._root / path
        if not full_path.exists():
            raise FileNotFoundError(f"File does not exist: {full_path}")
        if not full_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {full_path}")
        return full_path.stat().st_size
    
    def read_bytes(self, path: str) -> bytes:
        full_path = self._root / path
        if not full_path.exists():
            raise FileNotFoundError(f"File does not exist: {full_path}")
        if not full_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {full_path}")
        return full_path.read_bytes()
    
    def read_file_to_local(self, remote_path: str, local_path: str) -> None:
        # For local backend, this is just a file copy from one local path to another.
        full_remote_path = self._root / remote_path
        if not full_remote_path.exists():
            raise FileNotFoundError(f"File does not exist: {full_remote_path}")
        if not full_remote_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {full_remote_path}")
        Path(local_path).write_bytes(full_remote_path.read_bytes())
    
    def write_file_from_local(self, local_path: str, remote_path: str) -> None:
        # For local backend, this is just a file copy from one local path to another.
        self.read_file_to_local(remote_path=remote_path, local_path=local_path)

    @contextmanager
    def open_fileobj(self, path: str):
        full_path = self._root / path
        if not full_path.exists():
            raise FileNotFoundError(f"File does not exist: {full_path}")
        if not full_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {full_path}")
        with open(full_path, "rb") as f:
            yield f