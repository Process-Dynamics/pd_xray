"""Local file browsing, restricted to the configured root directories."""

import fnmatch
import glob
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pd_xray.core import PdXrayError
from pd_xray.io import supported_extensions
from pd_xray.io.sequence import natural_key

_SEQUENCE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


class AccessDeniedError(PdXrayError):
    """Raised for a path outside the server's root directories."""


def resolve_path(path: str | Path, roots: Sequence[Path]) -> Path:
    """Absolute, symlink-free form of `path`, checked to lie inside one of `roots`.

    Raises:
        AccessDeniedError: If the path is relative or resolves outside every root (including via symlinks or "..").
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise AccessDeniedError(f"Use an absolute path, got '{path}'.")
    resolved = candidate.resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise AccessDeniedError(f"'{path}' is outside the allowed folders: {', '.join(map(str, roots))}.")
    return resolved


def find_sequences(names: Sequence[str]) -> list[dict[str, Any]]:
    """Groups of image files that differ only in their last number, e.g. "scan_0001.tif", "scan_0002.tif".

    Returns:
        [{"pattern" (glob for `open_sequence`, e.g. "scan_*.tif"), "count" (files the pattern matches), "first"}] for
        groups of two or more files, in natural order of the pattern.
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        if not name.lower().endswith(_SEQUENCE_EXTENSIONS):
            continue
        parts = re.split(r"(\d+)", name)  # odd indices are digit runs, so the last one is at len - 2
        if len(parts) < 3:
            continue
        last = len(parts) - 2
        pattern = "".join(glob.escape(p) if i != last else "*" for i, p in enumerate(parts))
        groups.setdefault(pattern, []).append(name)
    sequences = []
    for pattern, members in groups.items():
        if len(members) >= 2:
            matched = sorted((n for n in names if fnmatch.fnmatchcase(n, pattern)), key=lambda n: natural_key(Path(n)))
            sequences.append({"pattern": pattern, "count": len(matched), "first": matched[0]})
    return sorted(sequences, key=lambda s: natural_key(Path(s["pattern"])))


def list_directory(directory: Path, roots: Sequence[Path], show_hidden: bool = False) -> dict[str, Any]:
    """Entries of a directory (already resolved with `resolve_path`): folders first, then files, in natural order.

    Returns:
        {"path", "parent" (None at a root), "entries": [{"name", "path", "kind" ("dir"/"file"), "size", "modified",
        "openable"}], "sequences": `find_sequences` of the files}. `size` is in bytes (None for folders) and `modified`
        is a POSIX timestamp.

    Raises:
        NotADirectoryError: If `directory` isn't a directory.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    extensions = set(supported_extensions())
    entries = []
    with os.scandir(directory) as scan:
        for entry in scan:
            if not show_hidden and entry.name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir()
                stat = entry.stat()
            except OSError:  # broken symlink or vanished file
                continue
            entries.append(
                {
                    "name": entry.name,
                    "path": str(directory / entry.name),
                    "kind": "dir" if is_dir else "file",
                    "size": None if is_dir else stat.st_size,
                    "modified": stat.st_mtime,
                    "openable": not is_dir and Path(entry.name).suffix.lower() in extensions,
                }
            )
    entries.sort(key=lambda e: (e["kind"] != "dir", natural_key(Path(str(e["name"])))))
    parent = None if directory in roots else str(directory.parent)
    files = [str(e["name"]) for e in entries if e["kind"] == "file"]
    return {"path": str(directory), "parent": parent, "entries": entries, "sequences": find_sequences(files)}
