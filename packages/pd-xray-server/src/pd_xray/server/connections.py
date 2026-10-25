"""Remote storage connections (SFTP, S3), kept per browser in memory only.

Each browser gets a random id cookie; its connections are invisible to other browsers. Credentials live only inside
the open `Storage` objects and are gone when the server stops. Only the server's own known_hosts file is read; the
server account's SSH keys, agent and AWS credentials are never used, so every connection needs its own credentials.
"""

import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pd_xray.core import PdXrayError
from pd_xray.io.sequence import natural_key
from pd_xray.server.datasets import NotFoundError
from pd_xray.storage import S3Storage, SFTPStorage, Storage

BROWSER_COOKIE = "pd_xray_browser"

_SETTINGS = {
    "sftp": {"host", "port", "username", "root", "host_key"},
    "s3": {"bucket", "prefix", "endpoint_url", "region"},
}
_SECRETS = {"sftp": {"password", "key_data", "passphrase"}, "s3": {"access_key", "secret_key", "session_token"}}


def new_browser_id() -> str:
    return secrets.token_urlsafe(16)


@dataclass
class Connection:
    """An open remote storage. `settings` holds no secrets."""

    id: str
    kind: str
    name: str
    settings: dict[str, Any]
    storage: Storage

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "name": self.name, "url": self.storage.url, "settings": self.settings}


def open_storage(kind: str, settings: dict[str, Any], credentials: dict[str, str]) -> Storage:
    """Build and test a storage from UI fields. Empty fields are ignored.

    Raises:
        PdXrayError: For an unknown kind, unknown fields or missing credentials.
        StorageError subclasses, FileNotFoundError: If connecting fails (see `Storage.check`).
    """
    if kind not in _SETTINGS:
        raise PdXrayError(f"Unknown connection kind '{kind}'. Use one of: {', '.join(_SETTINGS)}.")
    settings = {k: v for k, v in settings.items() if v not in (None, "")}
    credentials = {k: v for k, v in credentials.items() if v}
    unknown = (set(settings) - _SETTINGS[kind]) | (set(credentials) - _SECRETS[kind])
    if unknown:
        raise PdXrayError(f"Unknown {kind} fields: {', '.join(sorted(unknown))}.")
    fields: dict[str, Any] = {**settings, **credentials}
    storage: Storage
    if kind == "sftp":
        if not {"password", "key_data"} & set(credentials):
            raise PdXrayError("Give a password or a private key.")
        if not {"host", "username"} <= set(settings):
            raise PdXrayError("Give the host and the user name.")
        try:
            fields["port"] = int(fields.get("port", 22))
        except ValueError:
            raise PdXrayError(f"The port must be a number, got '{fields['port']}'.") from None
        storage = SFTPStorage(**fields)
    else:
        if not {"access_key", "secret_key"} <= set(credentials):
            raise PdXrayError("Give the access key and the secret key.")
        if "bucket" not in settings:
            raise PdXrayError("Give the bucket name.")
        storage = S3Storage(**fields)
    try:
        storage.check()
    except BaseException:
        storage.close()
        raise
    return storage


class ConnectionStore:
    """Open connections per browser id. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connections: dict[str, dict[str, Connection]] = {}

    def add(
        self, browser: str, kind: str, name: str, settings: dict[str, Any], credentials: dict[str, str]
    ) -> Connection:
        """Connect (testing the connection first) and remember it for `browser`."""
        storage = open_storage(kind, settings, credentials)
        saved = {k: v for k, v in settings.items() if v not in (None, "")}
        connection = Connection(secrets.token_hex(6), kind, name or storage.url, saved, storage)
        with self._lock:
            self._connections.setdefault(browser, {})[connection.id] = connection
        return connection

    def get(self, browser: str, connection_id: str) -> Connection:
        with self._lock:
            connection = self._connections.get(browser, {}).get(connection_id)
        if connection is None:
            raise NotFoundError(f"No connection '{connection_id}'. It may have closed when the server restarted.")
        return connection

    def list(self, browser: str) -> list[Connection]:
        with self._lock:
            return list(self._connections.get(browser, {}).values())

    def remove(self, browser: str, connection_id: str) -> None:
        with self._lock:
            connection = self._connections.get(browser, {}).pop(connection_id, None)
        if connection is not None:
            connection.storage.close()

    def close_all(self) -> None:
        with self._lock:
            connections = [c for per_browser in self._connections.values() for c in per_browser.values()]
            self._connections.clear()
        for connection in connections:
            connection.storage.close()


def list_remote(connection: Connection, path: str, show_hidden: bool = False) -> dict[str, Any]:
    """A remote folder (`path` normalised) shaped like `paths.list_directory` (folders first, natural order), plus
    "connection".

    Remote files can't be opened as datasets yet, so `openable` is False and no sequences are listed.
    """
    entries = [
        {
            "name": e.name,
            "path": e.path,
            "kind": "dir" if e.is_dir else "file",
            "size": e.size,
            "modified": e.modified,
            "openable": False,
        }
        for e in connection.storage.list(path)
        if show_hidden or not e.name.startswith(".")
    ]
    entries.sort(key=lambda e: (e["kind"] != "dir", natural_key(Path(str(e["name"])))))
    parent = None if not path else path.rpartition("/")[0]
    return {"connection": connection.id, "path": path, "parent": parent, "entries": entries, "sequences": []}
