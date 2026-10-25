"""Picks a format reader for a file and opens it as a `Dataset`."""

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pd_xray.core import PdXrayError
from pd_xray.io.dataset import Dataset


class UnsupportedFormatError(PdXrayError):
    """Raised when no reader handles a file's extension."""


# Extension -> "module:function". Modules are imported only when used, so a format's dependency is only needed when
# opening that format. Each function has the signature `(path: Path, axes: Sequence[str] | None, **options) -> Dataset`.
_OPENERS: dict[str, str] = {
    ".h5": "pd_xray.io.hdf5:open_hdf5",
    ".hdf5": "pd_xray.io.hdf5:open_hdf5",
    ".hdf": "pd_xray.io.hdf5:open_hdf5",
    ".nxs": "pd_xray.io.hdf5:open_hdf5",
    ".tif": "pd_xray.io.tiff:open_tiff",
    ".tiff": "pd_xray.io.tiff:open_tiff",
    ".png": "pd_xray.io.image:open_image",
    ".jpg": "pd_xray.io.image:open_image",
    ".jpeg": "pd_xray.io.image:open_image",
}


def supported_extensions() -> list[str]:
    """File extensions `open_dataset` can open."""
    return sorted(_OPENERS)


def open_dataset(path: str | Path, axes: Sequence[str] | None = None, **options: Any) -> Dataset:
    """Open a file as a lazy `Dataset`, choosing the reader by file extension.

    Args:
        path: Local file path.
        axes: Axis names overriding what the reader detects, e.g. ("angle", "y", "x").
        **options: Reader-specific options, e.g. `dataset="projections"` for HDF5.

    Raises:
        FileNotFoundError: If `path` doesn't exist.
        UnsupportedFormatError: If no reader handles the extension.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    target = _OPENERS.get(path.suffix.lower())
    if target is None:
        raise UnsupportedFormatError(
            f"Can't open '{path.name}': unsupported extension. Supported: {', '.join(supported_extensions())}."
        )
    module_name, func_name = target.split(":")
    opener = getattr(importlib.import_module(module_name), func_name)
    dataset: Dataset = opener(path, axes, **options)
    return dataset
