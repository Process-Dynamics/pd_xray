"""Copying files and folders between storages: parallel, resumable, with progress, cancel and integrity checks.

Each file downloads into a hidden ".<name>.<size>-<mtime>.pd-xray-part" file next to the target, which is renamed
when complete and checked. A cancelled or failed copy keeps its part files, so running it again resumes them. The
size and modification time in the part name make a changed source start afresh.
"""

import glob
import hashlib
import logging
import os
import threading
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pd_xray.core import CancelToken, OperationCancelledError, Progress, ProgressCallback
from pd_xray.storage.base import Entry, Storage, join_path, normalise_path
from pd_xray.storage.errors import StorageError
from pd_xray.storage.local import LocalStorage

logger = logging.getLogger(__name__)

CHUNK_BYTES = 8 << 20
"""Bytes per read request."""
PARALLEL_CHUNKS = 4
"""Read requests in flight per file. With `workers` files at once, memory use is about workers * 4 * 8 MB."""
PART_SUFFIX = ".pd-xray-part"


class IntegrityError(StorageError):
    """A copied file doesn't match its source (size or checksum). The bad copy is deleted."""


@dataclass(frozen=True)
class CopyResult:
    """What `copy` did.

    Attributes:
        copied: Files copied (including resumed ones).
        skipped: Files skipped because an identical copy (same size and modification time) was already there.
        bytes_copied: Bytes transferred, excluding skipped files and resumed parts.
        checksummed: Copied files whose checksum was compared (S3 MD5, or SHA-256 with `verify`).
    """

    copied: int
    skipped: int
    bytes_copied: int
    checksummed: int


def copy(
    source: Storage,
    source_path: str,
    destination: Storage,
    destination_path: str,
    *,
    overwrite: bool = False,
    verify: bool = False,
    workers: int = 4,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> CopyResult:
    """Copy a file, or a folder with everything below it, to `destination_path` (the path of the copy itself).

    Files that are already at the destination with the same size and modification time are skipped, so running a copy
    again only fetches what is missing. Copies keep the source's modification time. Every copy's size is checked; its
    MD5 too when the source knows it without reading (S3 single-part uploads). `verify` reads both copies again and
    compares SHA-256, which roughly doubles the time.

    Args:
        source: Where to copy from.
        source_path: File or folder in `source`.
        destination: Where to copy to. Only `LocalStorage` for now.
        destination_path: Path of the copied file or folder in `destination`.
        overwrite: Replace destination files that differ. Otherwise they stop the copy before anything is copied.
        verify: Compare SHA-256 of source and copy after copying.
        workers: Files copied at once.
        progress: Called with bytes done/total; `message` is the file being copied.
        cancel: Stops the copy between chunks. Part files are kept for resuming.

    Raises:
        FileNotFoundError: If `source_path` doesn't exist.
        FileExistsError: If destination files differ and `overwrite` is False (lists how many).
        IntegrityError: If a copy doesn't match its source.
        OperationCancelledError: If cancelled.
    """
    if not isinstance(destination, LocalStorage):
        raise StorageError(f"Copying to {destination.url} isn't supported yet; copy to a local folder.")
    stop = CancelToken()  # set when one file fails, so the others stop too

    def check() -> None:
        if cancel is not None:
            cancel.raise_if_cancelled()
        if stop.cancelled:
            raise OperationCancelledError("Stopped because another file failed.")

    if progress is not None:
        progress(Progress(0, None, "Listing files"))
    todo, skipped, conflicts = [], 0, []
    for entry, target_path in _plan(source, normalise_path(source_path), normalise_path(destination_path), check):
        target = destination.full_path(target_path)
        if not target.exists():
            todo.append((entry, target))
        elif target.is_dir():
            raise FileExistsError(f"Can't copy file {entry.path!r} over the folder {target}.")
        elif _identical(entry, target):
            skipped += 1
        elif overwrite:
            todo.append((entry, target))
        else:
            conflicts.append(target)
    if conflicts:
        raise FileExistsError(
            f"{len(conflicts)} file(s) already exist with different size or date, e.g. {conflicts[0]}. Choose "
            "overwrite to replace them."
        )

    total = sum(entry.size or 0 for entry, _ in todo)
    done, transferred, checksummed = 0, 0, 0
    lock = threading.Lock()

    def report(entry: Entry, nbytes: int, resumed: bool = False) -> None:
        nonlocal done, transferred
        with lock:
            done += nbytes
            transferred += 0 if resumed else nbytes
            current = done
        if progress is not None:
            progress(Progress(current, total, entry.name))

    if progress is not None:
        progress(Progress(0, total, f"{len(todo)} files"))
    with ThreadPoolExecutor(max(1, workers), thread_name_prefix="copy") as pool:
        futures = [pool.submit(_copy_file, source, entry, target, verify, report, check) for entry, target in todo]
        try:
            for future in as_completed(futures):
                checksummed += future.result()
        except BaseException:
            stop.cancel()
            raise
    logger.info("Copied %d files (%d bytes) from %s to %s", len(todo), transferred, source.url, destination.url)
    return CopyResult(len(todo), skipped, transferred, checksummed)


def _plan(
    source: Storage, source_path: str, destination_path: str, check: Callable[[], None]
) -> list[tuple[Entry, str]]:
    """(source file, destination path) for every file to copy."""
    entry = source.stat(source_path)
    if not entry.is_dir:
        return [(entry, destination_path)]
    files: list[tuple[Entry, str]] = []
    folders = [(entry.path, destination_path)]
    while folders:
        check()
        folder, target = folders.pop()
        for item in source.list(folder):
            if item.is_dir:
                folders.append((item.path, join_path(target, item.name)))
            elif not item.name.endswith(PART_SUFFIX):
                files.append((item, join_path(target, item.name)))
    return files


def _identical(entry: Entry, target: Path) -> bool:
    result = target.stat()
    return entry.modified is not None and result.st_size == entry.size and int(result.st_mtime) == int(entry.modified)


def _copy_file(
    source: Storage,
    entry: Entry,
    target: Path,
    verify: bool,
    report: Callable[..., None],
    check: Callable[[], None],
) -> int:
    """Copy one file through its part file. Returns 1 if a checksum was compared, else 0."""
    size, mtime = entry.size or 0, entry.modified or 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f".{target.name}.{size}-{int(mtime)}{PART_SUFFIX}")
    for stale in target.parent.glob(f".{glob.escape(target.name)}.*{PART_SUFFIX}"):
        if stale != part:
            stale.unlink()  # from an older version of the source
    expected_md5 = source.md5(entry.path)
    md5 = hashlib.md5() if expected_md5 else None
    start = part.stat().st_size if part.exists() else 0
    if start > size:
        start = 0
    with part.open("r+b" if start else "wb") as out:
        if md5 is not None and start:
            for block in iter(lambda: out.read(CHUNK_BYTES), b""):  # the part so far (start == its size)
                md5.update(block)
        out.seek(start)
        out.truncate()
        if start:
            logger.info("Resuming %s at %d of %d bytes", entry.path, start, size)
            report(entry, start, resumed=True)
        for data in _chunks(source, entry.path, start, size, check):
            out.write(data)
            if md5 is not None:
                md5.update(data)
            report(entry, len(data))
        out.flush()
        os.fsync(out.fileno())
    try:
        if part.stat().st_size != size:
            raise IntegrityError(f"{entry.path}: copied {part.stat().st_size} bytes, expected {size}.")
        if md5 is not None and md5.hexdigest() != expected_md5:
            raise IntegrityError(f"{entry.path}: MD5 of the copy doesn't match the source. Copy it again.")
        if verify and _sha256_source(source, entry, check) != _sha256_file(part):
            raise IntegrityError(f"{entry.path}: SHA-256 of the copy doesn't match the source. Copy it again.")
    except IntegrityError:
        part.unlink()
        raise
    os.utime(part, (mtime, mtime))
    os.replace(part, target)
    return int(md5 is not None or verify)


def _chunks(source: Storage, path: str, start: int, size: int, check: Callable[[], None]) -> Iterator[bytes]:
    """Yield the bytes of `path` from `start` in order, keeping `PARALLEL_CHUNKS` reads in flight."""
    offsets = iter(range(start, size, CHUNK_BYTES))
    with ThreadPoolExecutor(PARALLEL_CHUNKS, thread_name_prefix="chunk") as reader:

        def submit() -> None:
            offset = next(offsets, None)
            if offset is not None:
                pending.append(reader.submit(source.read_range, path, offset, min(CHUNK_BYTES, size - offset)))

        pending: deque[Future[bytes]] = deque()
        for _ in range(PARALLEL_CHUNKS):
            submit()
        while pending:
            check()
            data = pending.popleft().result()
            submit()
            yield data


def _sha256_source(source: Storage, entry: Entry, check: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    for data in _chunks(source, entry.path, 0, entry.size or 0, check):
        digest.update(data)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()
