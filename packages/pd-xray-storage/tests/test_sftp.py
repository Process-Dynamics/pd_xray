"""SFTPStorage against an in-process asyncssh SFTP server (password "secret", chrooted to a temp folder)."""

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

asyncssh = pytest.importorskip("asyncssh")

from pd_xray.storage import AuthenticationError, HostKeyError, SFTPStorage, StorageConnectionError  # noqa: E402

PASSWORD = "secret"


class _Server(asyncssh.SSHServer):  # type: ignore[misc]
    def begin_auth(self, username: str) -> bool:
        return True

    def password_auth_supported(self) -> bool:
        return True

    def validate_password(self, username: str, password: str) -> bool:
        return password == PASSWORD


class _Sftp:
    """A running SFTP server: `port`, `fingerprint` of its host key, `root` folder it serves, and `stop()`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        key = asyncssh.generate_private_key("ssh-ed25519")
        self.fingerprint = key.get_fingerprint()

        def sftp_factory(channel: Any) -> Any:
            return asyncssh.SFTPServer(channel, chroot=str(root))

        async def start() -> Any:
            return await asyncssh.listen(
                "127.0.0.1", 0, server_host_keys=[key], server_factory=_Server, sftp_factory=sftp_factory
            )

        self.server = asyncio.run_coroutine_threadsafe(start(), self.loop).result()
        self.port = self.server.sockets[0].getsockname()[1]

    def stop(self) -> None:
        self.server.close()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


@pytest.fixture
def server(tmp_path: Path) -> Iterator[_Sftp]:
    (tmp_path / "scans" / "0042").mkdir(parents=True)
    (tmp_path / "scans" / "0042" / "a.bin").write_bytes(bytes(range(256)) * 4096)  # 1 MiB
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "to_scans").symlink_to("scans")
    running = _Sftp(tmp_path)
    yield running
    running.stop()


def _storage(server: _Sftp, **kwargs: Any) -> SFTPStorage:
    options: dict[str, Any] = {"port": server.port, "password": PASSWORD, "host_key": server.fingerprint}
    return SFTPStorage("127.0.0.1", "user", **(options | kwargs))


def test_list_stat_open(server: _Sftp) -> None:
    with _storage(server) as storage:
        storage.check()
        assert {(e.name, e.is_dir, e.size) for e in storage.list()} == {
            ("scans", True, None),
            ("notes.txt", False, 2),
            ("to_scans", True, None),  # symlinks are followed
        }
        [entry] = storage.list("scans/0042")
        assert (entry.path, entry.size) == ("scans/0042/a.bin", 1 << 20)
        assert storage.stat("/scans/0042/a.bin") == entry
        with storage.open("scans/0042/a.bin") as f:
            f.seek(1000)
            assert f.read(4) == bytes([232, 233, 234, 235])
            f.seek(-2, 2)
            assert f.read() == bytes([254, 255])
            f.seek(0)
            assert f.read() == bytes(range(256)) * 4096
        with pytest.raises(FileNotFoundError):
            storage.stat("missing")
        with pytest.raises(NotADirectoryError):
            storage.list("notes.txt")
        with pytest.raises(IsADirectoryError):
            storage.open("scans")


def test_root(server: _Sftp) -> None:
    with _storage(server, root="/scans") as storage:
        assert [e.path for e in storage.list()] == ["0042"]
        assert storage.stat("0042/a.bin").size == 1 << 20


def test_errors(server: _Sftp) -> None:
    with pytest.raises(AuthenticationError, match="refused the login"):
        _storage(server, password="wrong").check()
    with pytest.raises(HostKeyError) as caught:
        _storage(server, host_key=None).check()  # not in known_hosts
    assert caught.value.fingerprint == server.fingerprint
    assert PASSWORD not in str(caught.value)
    with pytest.raises(HostKeyError, match="doesn't match"):
        _storage(server, host_key="SHA256:wrong").check()
    with pytest.raises(StorageConnectionError):
        SFTPStorage("127.0.0.1", "user", port=1, password=PASSWORD).check()


def test_reconnects(server: _Sftp) -> None:
    with _storage(server) as storage:
        storage.check()
        storage._run(storage._reset())  # as if the server dropped the connection
        assert storage.stat("notes.txt").size == 2


def test_copy_folder(server: _Sftp, tmp_path_factory: pytest.TempPathFactory) -> None:
    from pd_xray.storage import LocalStorage, copy

    destination = tmp_path_factory.mktemp("dst")
    with _storage(server) as storage:
        result = copy(storage, "scans", LocalStorage(destination), "scans", verify=True)
        assert (result.copied, result.checksummed) == (1, 1)
        assert (destination / "scans" / "0042" / "a.bin").read_bytes() == bytes(range(256)) * 4096
