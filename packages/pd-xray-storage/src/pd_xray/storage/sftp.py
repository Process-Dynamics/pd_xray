"""SFTP servers, e.g. a Synology NAS (enable SFTP under Control Panel > File Services). Needs the `[sftp]` extra.

asyncssh does the work (it keeps many requests in flight, so it is faster than paramiko). It runs on a private event
loop in a background thread, so the methods here are ordinary blocking, thread-safe calls.
"""

import asyncio
import logging
import posixpath
import threading
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pd_xray.core import import_optional
from pd_xray.storage.base import Entry, Storage, name_of, normalise_path
from pd_xray.storage.errors import AuthenticationError, HostKeyError, StorageConnectionError, StorageError

if TYPE_CHECKING:
    import asyncssh

logger = logging.getLogger(__name__)
T = TypeVar("T")

_MAX_REQUESTS = 128
"""Parallel read requests per file read (asyncssh default)."""


def _asyncssh() -> Any:
    return import_optional("asyncssh", "sftp", "pd-xray-storage")


class SFTPStorage(Storage):
    """A folder on an SFTP server.

    Credentials: give `password`, or a private key (`key_file` or `key_data`, with `passphrase` if encrypted). When any
    is given, only those are used. With none, the SSH agent and the default keys in ~/.ssh are tried.

    Host keys are checked against ~/.ssh/known_hosts. For a host not listed there, check the fingerprint in the
    `HostKeyError` and pass it as `host_key`.

    Args:
        host: Server name or address.
        username: Login name.
        port: SSH port.
        root: Folder that paths are relative to, e.g. "/volume1/xray". Default: the login folder.
        password: Password.
        key_file: Private key file.
        key_data: Private key text (OpenSSH or PEM).
        passphrase: Passphrase of the private key.
        host_key: Trusted host key fingerprint, e.g. "SHA256:…", for a host not in known_hosts.
        timeout: Seconds to wait for the connection.
    """

    def __init__(
        self,
        host: str,
        username: str,
        *,
        port: int = 22,
        root: str = "",
        password: str | None = None,
        key_file: str | Path | None = None,
        key_data: str | None = None,
        passphrase: str | None = None,
        host_key: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._asyncssh = _asyncssh()
        self.host, self.username, self.port, self.root, self.host_key = host, username, port, root, host_key
        self._timeout = timeout
        self._auth = self._auth_options(password, key_file, key_data, passphrase)
        self._conn: asyncssh.SSHClientConnection | None = None
        self._sftp: asyncssh.SFTPClient | None = None
        self._connect_lock = asyncio.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=f"sftp-{host}", daemon=True)
        self._thread.start()
        self._closed = False

    def _auth_options(
        self, password: str | None, key_file: str | Path | None, key_data: str | None, passphrase: str | None
    ) -> dict[str, Any]:
        if password is None and key_file is None and key_data is None:
            return {}  # asyncssh defaults: agent and ~/.ssh keys
        keys = []
        try:
            if key_file is not None:
                keys.append(self._asyncssh.read_private_key(str(key_file), passphrase))
            if key_data is not None:
                keys.append(self._asyncssh.import_private_key(key_data, passphrase))
        except (self._asyncssh.KeyImportError, self._asyncssh.KeyEncryptionError) as exc:
            raise AuthenticationError(f"Can't read the private key: {exc}. Check the key and its passphrase.") from None
        return {"password": password, "client_keys": keys, "agent_path": None}  # nothing else is tried

    @property
    def url(self) -> str:
        return f"sftp://{self.username}@{self.host}:{self.port}/{self.root.lstrip('/')}"

    # Event loop plumbing

    def _run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        if self._closed:
            coroutine.close()
            raise StorageError(f"{self.url} is closed.")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    async def _client(self) -> "asyncssh.SFTPClient":
        async with self._connect_lock:
            if self._sftp is None:
                self._conn = await self._connect()
                self._sftp = await self._conn.start_sftp_client()
            return self._sftp

    async def _connect(self) -> "asyncssh.SSHClientConnection":
        lib = self._asyncssh
        presented: list[str] = []
        trusted = self.host_key

        class Client(lib.SSHClient):  # type: ignore[misc, name-defined]
            def validate_host_public_key(self, host: str, addr: str, port: int, key: Any) -> bool:
                presented.append(key.get_fingerprint())  # only called for keys not in known_hosts
                return trusted is not None and presented[-1] == trusted

        options: dict[str, Any] = {"server_host_key_algs": "default"} if trusted else {}
        try:
            conn: asyncssh.SSHClientConnection
            conn, _ = await lib.create_connection(
                Client,
                self.host,
                self.port,
                username=self.username,
                connect_timeout=self._timeout,
                **options,
                **self._auth,
            )
        except lib.HostKeyNotVerifiable:
            fingerprint = presented[-1] if presented else "unknown"
            problem = "doesn't match the trusted key" if trusted else "isn't in ~/.ssh/known_hosts"
            raise HostKeyError(
                f"The host key of {self.host}:{self.port} ({fingerprint}) {problem}. If you trust it (compare it with "
                "the server's fingerprint), pass it as host_key.",
                fingerprint,
            ) from None
        except lib.PermissionDenied:
            raise AuthenticationError(
                f"{self.host} refused the login for '{self.username}'. Check the user name, password or key."
            ) from None
        except (OSError, asyncio.TimeoutError, lib.DisconnectError) as exc:
            raise StorageConnectionError(
                f"Can't connect to {self.host}:{self.port}: {str(exc) or 'timed out'}."
            ) from None
        logger.info("Connected to %s", self.url)
        return conn

    async def _call(self, operation: Callable[["asyncssh.SFTPClient"], Awaitable[T]], path: str) -> T:
        """Run `operation` on the SFTP client, reconnecting once if the connection dropped. Maps SFTP errors."""
        lib = self._asyncssh
        for attempt in (1, 2):
            sftp = await self._client()
            try:
                return await operation(sftp)
            except (lib.ConnectionLost, lib.DisconnectError, ConnectionError, BrokenPipeError):
                await self._reset()
                if attempt == 2:
                    raise StorageConnectionError(f"Lost the connection to {self.host}.") from None
                logger.info("Connection to %s dropped; reconnecting", self.host)
            except lib.SFTPNoSuchFile:
                raise FileNotFoundError(f"No such file or folder: {path!r} on {self.url}") from None
            except lib.SFTPPermissionDenied:
                raise PermissionError(f"Permission denied: {path!r} on {self.url}") from None
            except lib.SFTPError as exc:
                raise StorageError(f"SFTP error for {path!r} on {self.url}: {exc.reason}") from None
        raise AssertionError("unreachable")

    async def _reset(self) -> None:
        async with self._connect_lock:
            if self._conn is not None:
                self._conn.close()
            self._conn = self._sftp = None

    # Storage interface

    def _full(self, path: str) -> str:
        return posixpath.join(self.root, path) if self.root else (path or ".")

    def _entry(self, path: str, attrs: Any) -> Entry:
        is_dir = attrs.type == self._asyncssh.FILEXFER_TYPE_DIRECTORY
        return Entry(name_of(path), path, is_dir, None if is_dir else attrs.size, attrs.mtime)

    def stat(self, path: str = "") -> Entry:
        path = normalise_path(path)
        return self._entry(path, self._run(self._call(lambda sftp: sftp.stat(self._full(path)), path)))

    def list(self, path: str = "") -> list[Entry]:
        path = normalise_path(path)
        full = self._full(path)

        async def operation(sftp: "asyncssh.SFTPClient") -> list[Entry]:
            if not await sftp.isdir(full):
                if await sftp.exists(full):
                    raise NotADirectoryError(f"Not a folder: {path!r} on {self.url}")
                raise self._asyncssh.SFTPNoSuchFile("")
            names = [n for n in await sftp.readdir(full) if n.filename not in (".", "..")]
            links = [cast(str, n.filename) for n in names if n.attrs.type == self._asyncssh.FILEXFER_TYPE_SYMLINK]
            targets = await asyncio.gather(*(sftp.stat(posixpath.join(full, n)) for n in links), return_exceptions=True)
            attrs = {cast(str, n.filename): n.attrs for n in names}
            for name, target in zip(links, targets, strict=True):
                if isinstance(target, BaseException):
                    del attrs[name]  # broken link
                else:
                    attrs[name] = target
            return [self._entry(f"{path}/{name}".lstrip("/"), a) for name, a in attrs.items()]

        return self._run(self._call(operation, path))

    def read_range(self, path: str, offset: int, size: int) -> bytes:
        path = normalise_path(path)

        async def operation(sftp: "asyncssh.SFTPClient") -> bytes:
            async with sftp.open(self._full(path), "rb", max_requests=_MAX_REQUESTS) as file:  # reads run in parallel
                return cast(bytes, await file.read(size, offset))

        return self._run(self._call(operation, path))

    def close(self) -> None:
        if self._closed:
            return
        self._run(self._reset())
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
