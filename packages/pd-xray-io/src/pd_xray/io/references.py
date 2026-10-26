"""Reference frames (e.g. darks and flats) named by a folder or a list of files, as pipelines store them."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from pd_xray.core import PdXrayError
from pd_xray.io.convert import iter_frames
from pd_xray.io.formats import FileStore, open_dataset, supported_extensions
from pd_xray.io.sequence import natural_key


def reference_files(
    reference: str | Sequence[str], base: str | Path | None = None, storage: FileStore | None = None
) -> list[Path]:
    """The files a reference names. Each entry is a file, or a folder meaning every openable file directly in it.

    Args:
        reference: One path or several, as in a pipeline file.
        base: Folder that relative paths start from (usually the dataset's folder). The working directory (or the
            storage's root) if None.
        storage: Look in here instead of the local disk; absolute paths then start at its root.

    Raises:
        PdXrayError: If a path doesn't exist or a folder has no openable files.
    """
    extensions = set(supported_extensions())
    files: list[Path] = []
    for entry in (reference,) if isinstance(reference, str) else reference:
        path = Path(base or ".") / (Path(entry) if storage else Path(entry).expanduser())  # absolute replaces base
        if storage is not None:
            path = Path(path.as_posix().lstrip("/"))  # storage paths are relative to its root
            try:
                is_dir, is_file = storage.stat(str(path)).is_dir, True
            except FileNotFoundError:
                is_dir = is_file = False
            listed = [Path(e.path) for e in storage.list(str(path)) if not e.is_dir] if is_dir else []
        else:
            is_dir, is_file = path.is_dir(), path.is_file()
            listed = [p for p in path.iterdir() if p.is_file()] if is_dir else []
        if is_dir:
            found = sorted((p for p in listed if p.suffix.lower() in extensions), key=natural_key)
            if not found:
                raise PdXrayError(f"Folder '{path}' has no image files to use as a reference.")
            files.extend(found)
        elif is_file:
            files.append(path)
        else:
            raise PdXrayError(f"Reference '{entry}' doesn't exist (looked for {path}).")
    return files


def reference_frames(
    reference: str | Sequence[str], base: str | Path | None = None, storage: FileStore | None = None
) -> Iterator[NDArray[Any]]:
    """Every frame of every file a reference names (see `reference_files`), one at a time, in native dtype.

    Average them with `pd_xray.processing.average_frames` to resolve a pipeline:
    `pipeline.resolve(lambda ref: average_frames(reference_frames(ref, base)))`. With `storage`, files are read from
    there (e.g. darks on the NAS next to the scan).

    Raises:
        PdXrayError: If a path doesn't exist, or frames differ in shape.
    """
    shape = None
    for path in reference_files(reference, base, storage):
        with open_dataset(path, storage=storage) as dataset:
            for frame in iter_frames(dataset):
                if shape is not None and frame.shape != shape:
                    raise PdXrayError(f"{path.name} has {frame.shape} frames, but earlier files have {shape}.")
                shape = frame.shape
                yield frame
