"""Copying files and folders between any two storages: parallel, resumable, with progress, cancel and integrity checks.

Each file is written through `Storage.upload`: into hidden partial data that becomes the file when complete. A
cancelled or failed copy keeps its partial data, so running it again resumes. Remote-to-remote copies stream through
this machine without temporary files.
"""

import hashlib
import logging
import threading
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from pd_xray.core import CancelToken, OperationCancelledError, Progress, ProgressCallback
from pd_xray.storage.base import (
    PART_SUFFIX,
    Entry,
    Storage,
    is_stale_part,
    join_path,
    normalise_path,
    part_name,
)
from pd_xray.storage.errors import StorageError

logger = logging.getLogger(__name__)

CHUNK_BYTES = 8 << 20
"""Bytes per read request."""
PARALLEL_CHUNKS = 4
"""Read requests in flight per file. With `workers` files at once, memory use is about workers * 4 * 8 MB (plus
`s3.PARALLEL_PARTS` parts per file when writing to S3)."""
PARALLEL_STATS = 16
"""Destination folders listed, or files checked, at once while planning."""


class IntegrityError(StorageError):
    """A copied file doesn't match its source (size or checksum). The bad copy is deleted."""


@dataclass(frozen=True)
class CopyResult:
    """What `copy` did.

    Attributes:
        copied: Files copied (including resumed ones).
        skipped: Files skipped because an identical copy (same size and modification time) was already there.
        bytes_copied: Bytes transferred, excluding skipped files and resumed parts.
        checksummed: Copied files whose checksum was compared (source MD5 from S3, or SHA-256 with `verify`).
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
    again only copies what is missing. Copies keep the source's modification time. Every copy's size is checked; its
    MD5 too when the source knows it without reading (S3 single-part uploads). `verify` reads both copies again and
    compares SHA-256, which roughly doubles the time.

    Args:
        source: Where to copy from.
        source_path: File or folder in `source`.
        destination: Where to copy to. May be the same storage as `source`.
        destination_path: Path of the copied file or folder in `destination`.
        overwrite: Replace destination files that differ. Otherwise they stop the copy before anything is copied.
        verify: Compare SHA-256 of source and copy after copying.
        workers: Files copied at once.
        progress: Called with bytes done/total; `message` is the file being copied.
        cancel: Stops the copy between chunks. Partial data is kept for resuming.

    Raises:
        FileNotFoundError: If `source_path` doesn't exist.
        FileExistsError: If destination files differ and `overwrite` is False (lists how many).
        IntegrityError: If a copy doesn't match its source.
        OperationCancelledError: If cancelled.
    """
    source_path, destination_path = normalise_path(source_path), normalise_path(destination_path)
    inside = destination_path.startswith(f"{source_path}/".lstrip("/")) and destination_path != source_path
    if source.url == destination.url and inside:
        raise StorageError(f"Can't copy {source_path or 'the root'!r} into itself ({destination_path!r}).")
    stop = CancelToken()  # set when one file fails, so the others stop too

    def check() -> None:
        if cancel is not None:
            cancel.raise_if_cancelled()
        if stop.cancelled:
            raise OperationCancelledError("Stopped because another file failed.")

    if progress is not None:
        progress(Progress(0, None, "Listing files"))
    plan = _plan(source, source_path, destination_path, check)
    todo, skipped = _compare(plan, destination, overwrite, check)

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
        futures = [
            pool.submit(_copy_file, source, entry, destination, target, verify, report, check) for entry, target in todo
        ]
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
    """(source file, destination path) for every file to copy, with exact dates."""
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
    if not source.list_dates_exact:
        with ThreadPoolExecutor(PARALLEL_STATS, thread_name_prefix="plan") as pool:
            entries = list(pool.map(source.stat, [entry.path for entry, _ in files]))
        files = [(entry, target) for entry, (_, target) in zip(entries, files, strict=True)]
    return files


def _compare(
    plan: list[tuple[Entry, str]], destination: Storage, overwrite: bool, check: Callable[[], None]
) -> tuple[list[tuple[Entry, str]], int]:
    """The planned files that need copying, and how many are skipped as identical. Lists each destination folder once
    and deletes partial data of older source versions.

    Raises:
        FileExistsError: For differing files without `overwrite`, or a folder in the way.
    """

    def listing(folder: str) -> list[Entry]:
        check()
        try:
            return destination.list(folder)
        except FileNotFoundError:
            return []

    folders = sorted({target.rpartition("/")[0] for _, target in plan})
    with ThreadPoolExecutor(PARALLEL_STATS, thread_name_prefix="plan") as pool:
        listings = dict(zip(folders, pool.map(listing, folders), strict=True))
    existing = {item.path: item for items in listings.values() for item in items}
    parts = {folder: [item for item in items if item.name.endswith(PART_SUFFIX)] for folder, items in listings.items()}

    todo, maybe_same, conflicts, stale = [], [], [], []
    for entry, target in plan:
        folder, _, name = target.rpartition("/")
        current = part_name(name, entry.size or 0, entry.modified or 0.0)
        stale += [item.path for item in parts[folder] if is_stale_part(item.name, name, current)]
        there = existing.get(target)
        if there is None:
            todo.append((entry, target))
        elif there.is_dir:
            raise FileExistsError(f"Can't copy file {entry.path!r} over the folder {target!r} in {destination.url}.")
        elif there.size == entry.size and entry.modified is not None:
            maybe_same.append((entry, target))
        elif overwrite:
            todo.append((entry, target))
        else:
            conflicts.append(target)

    with ThreadPoolExecutor(PARALLEL_STATS, thread_name_prefix="plan") as pool:  # S3 keeps dates in metadata: stat
        stats = list(pool.map(lambda pair: destination.stat(pair[1]), maybe_same))
    skipped = 0
    for (entry, target), there in zip(maybe_same, stats, strict=True):
        if there.modified is not None and int(there.modified) == int(entry.modified or 0):
            skipped += 1
        elif overwrite:
            todo.append((entry, target))
        else:
            conflicts.append(target)
    if conflicts:
        raise FileExistsError(
            f"{len(conflicts)} file(s) already exist with different size or date, e.g. {conflicts[0]!r} in "
            f"{destination.url}. Choose overwrite to replace them."
        )
    for path in stale:
        destination.remove(path)  # from an older version of the source
    return todo, skipped


def _copy_file(
    source: Storage,
    entry: Entry,
    destination: Storage,
    target: str,
    verify: bool,
    report: Callable[..., None],
    check: Callable[[], None],
) -> int:
    """Copy one file through `Storage.upload`. Returns 1 if a checksum was compared, else 0."""
    size, mtime = entry.size or 0, entry.modified or 0.0
    expected_md5 = source.md5(entry.path)
    md5 = hashlib.md5() if expected_md5 else None
    with destination.upload(target, size, mtime) as upload:
        start = upload.offset
        if start:
            logger.info("Resuming %s at %d of %d bytes", entry.path, start, size)
            if md5 is not None and upload.part_path is None:
                md5 = None  # S3 parts can't be read back; S3 checked each with Content-MD5
            elif md5 is not None and upload.part_path is not None:
                for data in _chunks(destination, upload.part_path, 0, start, check):
                    md5.update(data)
            report(entry, start, resumed=True)
        for data in _chunks(source, entry.path, start, size, check):
            upload.write(data)
            if md5 is not None:
                md5.update(data)
            report(entry, len(data))
        if md5 is not None and md5.hexdigest() != expected_md5:
            upload.discard()
            raise IntegrityError(f"{entry.path}: MD5 of the copy doesn't match the source. Copy it again.")
        upload.commit()
    try:
        copied = destination.stat(target).size
        if copied != size:
            raise IntegrityError(f"{entry.path}: copied {copied} bytes, expected {size}.")
        if verify and _sha256(source, entry.path, size, check) != _sha256(destination, target, size, check):
            raise IntegrityError(f"{entry.path}: SHA-256 of the copy doesn't match the source. Copy it again.")
    except IntegrityError:
        destination.remove(target)
        raise
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


def _sha256(storage: Storage, path: str, size: int, check: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    for data in _chunks(storage, path, 0, size, check):
        digest.update(data)
    return digest.hexdigest()
