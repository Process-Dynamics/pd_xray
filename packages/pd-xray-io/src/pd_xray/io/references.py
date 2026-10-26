"""Reference frames (e.g. darks and flats) named by a folder or a list of files, as pipelines store them."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from pd_xray.core import PdXrayError
from pd_xray.io.convert import iter_frames
from pd_xray.io.formats import open_dataset, supported_extensions
from pd_xray.io.sequence import natural_key


def reference_files(reference: str | Sequence[str], base: str | Path | None = None) -> list[Path]:
    """The files a reference names. Each entry is a file, or a folder meaning every openable file directly in it.

    Args:
        reference: One path or several, as in a pipeline file.
        base: Folder that relative paths start from (usually the dataset's folder). The working directory if None.

    Raises:
        PdXrayError: If a path doesn't exist or a folder has no openable files.
    """
    extensions = set(supported_extensions())
    files: list[Path] = []
    for entry in (reference,) if isinstance(reference, str) else reference:
        path = Path(base or ".") / Path(entry).expanduser()  # an absolute `entry` replaces `base`
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


def reference_frames(reference: str | Sequence[str], base: str | Path | None = None) -> Iterator[NDArray[Any]]:
    """Every frame of every file a reference names (see `reference_files`), one at a time, in native dtype.

    Average them with `pd_xray.processing.average_frames` to resolve a pipeline:
    `pipeline.resolve(lambda ref: average_frames(reference_frames(ref, base)))`.

    Raises:
        PdXrayError: If a path doesn't exist, or frames differ in shape.
    """
    shape = None
    for path in reference_files(reference, base):
        with open_dataset(path) as dataset:
            for frame in iter_frames(dataset):
                if shape is not None and frame.shape != shape:
                    raise PdXrayError(f"{path.name} has {frame.shape} frames, but earlier files have {shape}.")
                shape = frame.shape
                yield frame
