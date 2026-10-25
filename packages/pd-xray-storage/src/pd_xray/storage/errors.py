"""Storage errors. Missing paths raise the built-in `FileNotFoundError` / `NotADirectoryError` / `PermissionError`."""

from pd_xray.core import PdXrayError


class StorageError(PdXrayError):
    """Base class for storage errors."""


class StorageConnectionError(StorageError):
    """The server can't be reached: wrong host/port/endpoint, network down or timeout."""


class AuthenticationError(StorageError):
    """The server refused the credentials, or none were found."""


class HostKeyError(StorageError):
    """The SSH server's host key is unknown or doesn't match. Check `fingerprint`, then pass it as `host_key=`.

    Attributes:
        fingerprint: The key the server presented, e.g. "SHA256:…".
    """

    def __init__(self, message: str, fingerprint: str) -> None:
        super().__init__(message)
        self.fingerprint = fingerprint


class InvalidPathError(StorageError, ValueError):
    """A path that leaves the storage root (".." components) or is otherwise malformed."""
