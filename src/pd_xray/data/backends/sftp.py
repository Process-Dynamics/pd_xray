import fnmatch
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import paramiko

from pd_xray.data.backends.base import FileInfo, StorageBackend
from pd_xray.core.logging import get_logger

logger = get_logger(__name__)


class SFTPBackend(StorageBackend):
    """Storage backend for SFTP servers (Synology NAS, etc.).

    Constructor args:
        host : Hostname or IP of the SFTP server.
        port : SSH port (default 22).
        root : Remote directory treated as the backend root. All paths passed to
               other methods are relative to this directory.

    Credentials are passed at connect() time:
        username : SSH username.
        password : SSH password.

    Usage::

        backend = SFTPBackend(host="192.168.1.10", root="/volume1/xray")
        backend.connect(username="admin", password="secret")
        files = backend.list_files("raw/", "*.edf")
        backend.disconnect()

    Or as a context manager::

        with SFTPBackend(host="192.168.1.10", root="/volume1/xray") as backend:
            backend.connect(username="admin", password="secret")
            files = backend.list_files(...)

    Or from a YAML config file::

        backend = SFTPBackend.from_config("config.yaml")
        backend.connect(username="admin", password="secret")

    YAML config keys:
        sftp.host : Remote hostname or IP (required).
        sftp.port : SSH port (optional, default 22).
        sftp.root : Remote root directory (optional, default "/").
    """

    def __init__(
        self,
        host: str,
        port: int = 22,
        root: str = "/",
    ) -> None:
        self._host = host
        self._port = port
        self._root = PurePosixPath(root)
        self._transport: paramiko.Transport | None = None

    @classmethod
    def from_config(cls, config: Any) -> "SFTPBackend":
        """Create an SFTPBackend from a Config instance or path to a YAML file.

        Args:
            config : Config instance or path to a YAML file.

        Returns:
            Configured SFTPBackend, not yet connected.
        """
        from pd_xray.core.config import Config
        if not isinstance(config, Config):
            config = Config(config)
        return cls(
            host=config.require("sftp.host"),
            port=config.get("sftp.port", 22),
            root=config.get("sftp.root", "/"),
        )

    def connect(self, **credentials: Any) -> None:
        """Open SSH transport and authenticate.

        Keyword args:
            username : SSH username.
            password : SSH password.
        """
        username = credentials.get("username")
        password = credentials.get("password")
        if not username or not password:
            raise ValueError("connect() requires 'username' and 'password'.")
        transport = paramiko.Transport((self._host, self._port))
        transport.connect(username=username, password=password)
        self._transport = transport
        logger.debug(f"Connected to sftp://{username}@{self._host}:{self._port}{self._root}")

    def disconnect(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    @property
    def isconnected(self) -> bool:
        return self._transport is not None and self._transport.is_active()

    def _require_connected(self) -> None:
        if not self.isconnected:
            raise RuntimeError("Not connected. Call connect() first.")

    def _full_path(self, path: str) -> str:
        return str(self._root / path.lstrip("/"))

    def _new_sftp(self) -> paramiko.SFTPClient:
        self._require_connected()
        sftp = paramiko.SFTPClient.from_transport(self._transport)  # type: ignore[arg-type]
        if sftp is None:
            raise RuntimeError("Failed to open SFTP channel.")
        return sftp

    def list_files(
        self,
        prefix: str = "",
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[FileInfo]:
        """List files matching a glob pattern under prefix.

        Args:
            prefix    : Subdirectory relative to the backend root.
            pattern   : Glob pattern applied to filenames only (e.g. "proj_*.edf").
            recursive : Whether to descend into subdirectories.

        Returns:
            Sorted list of FileInfo entries with paths relative to the backend root.
        """
        search_dir = self._full_path(prefix)
        sftp = self._new_sftp()
        results: list[FileInfo] = []
        try:
            self._collect(sftp, search_dir, pattern, recursive, results)
        finally:
            sftp.close()
        return sorted(results, key=lambda f: f.path)

    def _collect(
        self,
        sftp: paramiko.SFTPClient,
        directory: str,
        pattern: str,
        recursive: bool,
        results: list[FileInfo],
    ) -> None:
        try:
            entries = sftp.listdir_attr(directory)
        except IOError as e:
            raise FileNotFoundError(f"Remote directory not found: {directory}") from e
        root_str = str(self._root).rstrip("/")
        for entry in entries:
            if entry.filename.startswith("."):
                continue
            full = directory.rstrip("/") + "/" + entry.filename
            rel = full[len(root_str):].lstrip("/")
            is_dir = stat.S_ISDIR(entry.st_mode or 0)
            if is_dir:
                results.append(FileInfo(path=rel + "/", size_bytes=-1, is_directory=True))
                if recursive:
                    self._collect(sftp, full, pattern, recursive, results)
            elif fnmatch.fnmatch(entry.filename, pattern):
                results.append(FileInfo(path=rel, size_bytes=entry.st_size or -1, is_directory=False))

    @contextmanager
    def open_fileobj(self, path: str):
        """Return a seekable, read-only file-like object for the given remote path.

        Each call opens a dedicated SFTP channel so concurrent calls from
        different threads do not block each other.

        For sequential reads of large files, call f.prefetch() inside the with-block
        to enable read-ahead. Avoid prefetch for random-access formats like HDF5.

        Usage::

            with backend.open_fileobj("scan.h5") as f:
                with h5py.File(f, "r") as hf:
                    data = hf["entry/data"][:]
        """
        remote = self._full_path(path)
        sftp = self._new_sftp()
        try:
            try:
                f = sftp.open(remote, "rb")
            except IOError as e:
                raise FileNotFoundError(f"Remote file not found: {remote}") from e
            try:
                yield f
            finally:
                f.close()
        finally:
            sftp.close()

    def read_bytes(self, path: str) -> bytes:
        remote = self._full_path(path)
        sftp = self._new_sftp()
        try:
            with sftp.open(remote, "rb") as f:
                return f.read()
        except IOError as e:
            raise FileNotFoundError(f"Remote file not found: {remote}") from e
        finally:
            sftp.close()

    def read_file_to_local(self, remote_path: str, local_path: str) -> None:
        remote = self._full_path(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sftp = self._new_sftp()
        try:
            sftp.get(remote, local_path)
            logger.debug(f"Downloaded sftp://{self._host}{remote} -> {local_path}")
        except IOError as e:
            raise FileNotFoundError(f"Remote file not found: {remote}") from e
        finally:
            sftp.close()

    def write_file_from_local(self, local_path: str, remote_path: str) -> None:
        if not Path(local_path).is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        remote = self._full_path(remote_path)
        sftp = self._new_sftp()
        try:
            sftp.put(local_path, remote)
            logger.debug(f"Uploaded {local_path} -> sftp://{self._host}{remote}")
        finally:
            sftp.close()

    def exists(self, path: str) -> bool:
        remote = self._full_path(path)
        sftp = self._new_sftp()
        try:
            sftp.stat(remote)
            return True
        except IOError:
            return False
        finally:
            sftp.close()

    def file_size(self, path: str) -> int:
        remote = self._full_path(path)
        sftp = self._new_sftp()
        try:
            return sftp.stat(remote).st_size
        except IOError as e:
            raise FileNotFoundError(f"Remote file not found: {remote}") from e
        finally:
            sftp.close()
