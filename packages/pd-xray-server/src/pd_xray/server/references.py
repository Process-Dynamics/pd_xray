"""Reference frames for pipelines (e.g. darks and flats): a folder or a list of files, averaged into one frame."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from pd_xray.core import CancelToken, PdXrayError
from pd_xray.io import iter_frames, open_dataset, supported_extensions
from pd_xray.io.sequence import natural_key
from pd_xray.processing import average_frames
from pd_xray.processing.pipeline import Reference
from pd_xray.server.paths import resolve_path


def reference_files(reference: Reference, base: Path, roots: Sequence[Path]) -> list[Path]:
    """The files a reference names. Each entry is a file or a folder (all openable files directly inside it).

    Args:
        reference: One path or several. Relative paths are relative to `base` (the dataset's folder).
        base: Folder for relative paths.
        roots: Allowed folders.

    Raises:
        AccessDeniedError: For a path outside the roots.
        PdXrayError: If a path doesn't exist or a folder has no openable files.
    """
    extensions = set(supported_extensions())
    files: list[Path] = []
    for entry in (reference,) if isinstance(reference, str) else reference:
        path = resolve_path(base / entry, roots)  # an absolute `entry` replaces `base`
        if path.is_dir():
            found = sorted(
                (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in extensions), key=natural_key
            )
            if not found:
                raise PdXrayError(f"Folder '{path}' has no image files to use as a reference.")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise PdXrayError(f"Reference '{entry}' doesn't exist (looked for {path}).")
    return files


def average_files(files: Sequence[Path], cancel: CancelToken | None = None) -> NDArray[Any]:
    """Mean of every frame in `files` (single- or multi-frame files), as float32.

    Raises:
        PdXrayError: If a file can't be opened or frames differ in shape.
    """

    def frames() -> Iterator[NDArray[Any]]:
        shape = None
        for path in files:
            with open_dataset(path) as dataset:
                for frame in iter_frames(dataset):
                    if shape is not None and frame.shape != shape:
                        raise PdXrayError(f"{path.name} has {frame.shape} frames, but earlier files have {shape}.")
                    shape = frame.shape
                    yield frame

    return average_frames(frames(), cancel=cancel)
