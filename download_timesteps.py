#!/usr/bin/env python3
import argparse
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pd_xray.data.backends.sftp import SFTPBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(
    r"^[^_]+_\d{3}(\d{4})_[^_]+_[^_]+_(\d+)\.(tif|tiff)$",
    re.IGNORECASE,
)

_thread_local = threading.local()
_stop = threading.Event()


def parse_timestep(name: str) -> int | None:
    m = _FILENAME_RE.match(name)
    return int(m.group(1)) if m else None


def _get_backend(conn: dict) -> SFTPBackend:
    if _stop.is_set():
        raise RuntimeError("Shutdown in progress.")
    b = getattr(_thread_local, "backend", None)
    if b is None or not b.isconnected:
        b = SFTPBackend(host=conn["host"], port=conn["port"], root=conn["remote"])
        b.connect(username=conn["username"], password=conn["password"])
        _thread_local.backend = b
    return b


def download_file(conn: dict, remote_path: str, local_path: Path, retries: int = 3) -> bool:
    if _stop.is_set():
        return False
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            backend = _get_backend(conn)
            backend.read_file_to_local(remote_path, str(tmp))
            tmp.rename(local_path)
            return True
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if _stop.is_set():
                return False
            if attempt < retries:
                logger.warning(f"Attempt {attempt}/{retries} failed for {remote_path}: {e}")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Giving up on {remote_path} after {retries} attempts: {e}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SFTP tiff files grouped by time step.")
    parser.add_argument("--host", required=True, help="SFTP server hostname or IP.")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--remote", required=True, help="Remote directory containing the files.")
    parser.add_argument("--local", required=True, help="Local root directory for downloads.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download threads.")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N files.")
    args = parser.parse_args()

    local_root = Path(args.local)
    local_root.mkdir(parents=True, exist_ok=True)

    conn = {
        "host": args.host,
        "port": args.port,
        "remote": args.remote,
        "username": args.username,
        "password": args.password,
    }

    listing_backend = SFTPBackend(host=args.host, port=args.port, root=args.remote)
    listing_backend.connect(username=args.username, password=args.password)
    logger.info(f"Connected to {args.host}:{args.port}{args.remote}")

    logger.info("Listing remote files...")
    all_files = listing_backend.list_files(pattern="*.tif")
    listing_backend.disconnect()
    logger.info(f"Found {len(all_files)} files.")

    by_timestep: dict[int, list[str]] = {}
    for fi in all_files:
        ts = parse_timestep(Path(fi.path).name)
        if ts is not None:
            by_timestep.setdefault(ts, []).append(fi.path)

    logger.info(f"Time steps found: {sorted(by_timestep)}")

    to_download: dict[int, list[str]] = {}
    for ts, files in sorted(by_timestep.items()):
        folder = local_root / f"t{ts}"
        if folder.exists():
            logger.info(f"Skipping t{ts} — folder already exists ({len(files)} files).")
        else:
            to_download[ts] = files

    if not to_download:
        logger.info("Nothing to download.")
        return

    total = sum(len(v) for v in to_download.values())
    logger.info(
        f"Downloading {total} files across {len(to_download)} time steps "
        f"with {args.workers} workers."
    )

    completed = 0
    failed = 0
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)

    try:
        future_to_path: dict = {}
        for ts, files in sorted(to_download.items()):
            folder = local_root / f"t{ts}"
            folder.mkdir(parents=True, exist_ok=True)
            for remote_path in files:
                local_path = folder / Path(remote_path).name
                f = pool.submit(download_file, conn, remote_path, local_path)
                future_to_path[f] = remote_path

        for future in as_completed(future_to_path):
            try:
                ok = future.result()
            except Exception as e:
                logger.error(f"Unhandled error for {future_to_path[future]}: {e}")
                ok = False
            if ok:
                completed += 1
            else:
                failed += 1
            done = completed + failed
            if done % args.progress_every == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else float("inf")
                logger.info(
                    f"Progress {done}/{total} ({100 * done / total:.1f}%) | "
                    f"OK: {completed}  Failed: {failed} | "
                    f"{rate:.1f} files/s | ETA {eta / 60:.1f} min"
                )

        logger.info(f"Finished. {completed}/{total} downloaded. {failed} failed.")

    except KeyboardInterrupt:
        logger.info("Interrupted — stopping workers, finishing in-progress downloads...")
        _stop.set()
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        except KeyboardInterrupt:
            logger.warning("Second interrupt — forcing immediate exit.")
            pool.shutdown(wait=False, cancel_futures=True)
        for tmp in local_root.rglob("*.tmp"):
            tmp.unlink(missing_ok=True)
            logger.info(f"Removed partial file: {tmp}")
        logger.info(f"Stopped. {completed}/{total} downloaded before interrupt. {failed} failed.")

    finally:
        pool.shutdown(wait=False)


if __name__ == "__main__":
    main()
