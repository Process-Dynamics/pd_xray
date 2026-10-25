"""Command line: `pd-xray-storage ls URL` and `pd-xray-storage copy SOURCE DESTINATION`.

Locations are local paths, `sftp://user@host[:port]/path` or `s3://bucket/key`. SFTP logs in with the SSH agent or
~/.ssh keys, `--key-file`, or `--password` (asked for, never on the command line). S3 uses `--profile` or the usual
AWS environment variables and ~/.aws files.
"""

import argparse
import getpass
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pd_xray.core import PdXrayError, Progress
from pd_xray.storage.base import Storage, name_of, normalise_path
from pd_xray.storage.errors import HostKeyError
from pd_xray.storage.local import LocalStorage
from pd_xray.storage.s3 import S3Storage
from pd_xray.storage.sftp import SFTPStorage
from pd_xray.storage.transfer import copy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pd-xray-storage", description="List and copy files on local, SFTP and S3.")
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group("connection")
    group.add_argument("--password", action="store_true", help="Ask for the SFTP password.")
    group.add_argument("--key-file", type=Path, help="SFTP private key file.")
    group.add_argument("--host-key", help="Trusted SFTP host key fingerprint (SHA256:…) for a host not in known_hosts.")
    group.add_argument("--endpoint-url", help="S3-compatible service URL, e.g. https://minio.lab:9000.")
    group.add_argument("--profile", help="AWS profile in ~/.aws.")
    group.add_argument("--region", help="S3 region.")
    commands = parser.add_subparsers(dest="command", required=True)
    ls = commands.add_parser("ls", parents=[options], help="List a folder.")
    ls.add_argument("location")
    cp = commands.add_parser("copy", parents=[options], help="Copy a file or folder (resumes interrupted copies).")
    cp.add_argument("source")
    cp.add_argument("destination", help="Local path. An existing folder receives the copy inside it, like cp.")
    cp.add_argument("--overwrite", action="store_true", help="Replace destination files that differ.")
    cp.add_argument("--verify", action="store_true", help="Compare SHA-256 of every copy (reads both twice).")
    cp.add_argument("--workers", type=int, default=4, help="Files copied at once (default: %(default)s).")
    return parser


def _open(location: str, args: argparse.Namespace) -> tuple[Storage, str]:
    """Storage and path inside it for a location."""
    url = urlsplit(location)
    if url.scheme == "sftp":
        if not url.hostname or not url.username:
            raise PdXrayError(f"Use sftp://user@host[:port]/path, got '{location}'.")
        password = getpass.getpass(f"Password for {url.username}@{url.hostname}: ") if args.password else None
        storage: Storage = SFTPStorage(
            url.hostname,
            unquote(url.username),
            port=url.port or 22,
            root="/",
            password=password,
            key_file=args.key_file,
            host_key=args.host_key,
        )
        return storage, unquote(url.path)
    if url.scheme == "s3":
        storage = S3Storage(url.netloc, endpoint_url=args.endpoint_url, region=args.region, profile=args.profile)
        return storage, unquote(url.path)
    if url.scheme:
        raise PdXrayError(f"Unknown location type '{url.scheme}://'. Use a local path, sftp:// or s3://.")
    return LocalStorage("/"), str(Path(location).expanduser().resolve())


def _show_progress(update: Progress, state: dict[str, float]) -> None:
    now = time.monotonic()
    if now - state.get("shown", 0) < 0.25 and update.current != update.total:
        return
    state["shown"] = now
    if update.total is None:
        print(f"\r{update.message}…", end="", file=sys.stderr, flush=True)
        return
    state.setdefault("start", now)
    rate = update.current / max(now - state["start"], 1e-3) / 1e6
    print(
        f"\r{100 * (update.fraction or 1):5.1f}%  {update.current / 1e9:.2f}/{update.total / 1e9:.2f} GB  "
        f"{rate:6.1f} MB/s  {update.message[:40]:<40}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the `pd-xray-storage` command."""
    args = _parser().parse_args(argv)
    storages: list[Storage] = []
    try:
        source, path = _open(args.location if args.command == "ls" else args.source, args)
        storages.append(source)
        if args.command == "ls":
            for entry in sorted(source.list(path), key=lambda e: (not e.is_dir, e.name)):
                print(f"{entry.name}/" if entry.is_dir else f"{entry.name}\t{entry.size}")
            return
        destination, target = _open(args.destination, args)
        storages.append(destination)
        if isinstance(destination, LocalStorage) and Path("/", target).is_dir():
            target = f"{target}/{name_of(normalise_path(path)) or 'copy'}"
        state: dict[str, float] = {}
        result = copy(
            source,
            path,
            destination,
            target,
            overwrite=args.overwrite,
            verify=args.verify,
            workers=args.workers,
            progress=lambda update: _show_progress(update, state),
        )
        print(file=sys.stderr)
        print(
            f"Copied {result.copied} files ({result.bytes_copied / 1e9:.2f} GB), skipped {result.skipped} identical, "
            f"checksum compared for {result.checksummed}."
        )
    except KeyboardInterrupt:
        print("\nCancelled. Run the same command again to resume.", file=sys.stderr)
        sys.exit(130)
    except HostKeyError as exc:
        sys.exit(f"error: {exc}\nIf you trust it, add: --host-key {exc.fingerprint}")
    except (PdXrayError, OSError) as exc:
        sys.exit(f"\nerror: {exc}")
    finally:
        for storage in storages:
            storage.close()


if __name__ == "__main__":
    main()
