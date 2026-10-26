"""PNG/JPG reader (extra `[png]`: Pillow). These are single, modest-sized images, so they are read whole."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core import CHANNEL, X, Y, import_optional
from pd_xray.io.dataset import ArrayDataset
from pd_xray.io.formats import FileStore, readable

METADATA_KEY = "pd_xray"
"""PNG text chunk / JPEG comment holding pd_xray metadata as JSON."""


def _pillow() -> Any:
    return import_optional("PIL.Image", "png", "pd-xray-io")


def read_image(path: Path, storage: FileStore | None = None) -> tuple[NDArray[Any], dict[str, Any]]:
    """Read a PNG/JPG as an array in its stored bit depth (8/16-bit grey, RGB, RGBA), plus its metadata."""
    with readable(path, storage) as source, _pillow().open(source) as img:
        if img.mode == "P":
            img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        array = np.asarray(img)
        info = dict(img.info)
    metadata = {
        k: v for k, v in info.items() if isinstance(v, (str, int, float)) and k not in (METADATA_KEY, "comment")
    }
    stored = info.get(METADATA_KEY) or info.get("comment")
    if stored:
        try:
            metadata.update(json.loads(stored))
        except (ValueError, TypeError):
            metadata["comment"] = stored.decode(errors="replace") if isinstance(stored, bytes) else str(stored)
    return array, metadata


def open_image(path: Path, axes: Sequence[str] | None = None, storage: FileStore | None = None) -> ArrayDataset:
    """Open a PNG/JPG. Axes default to ("y", "x"), or ("y", "x", "channel") for colour."""
    array, metadata = read_image(path, storage)
    return ArrayDataset(array, axes or ((Y, X) if array.ndim == 2 else (Y, X, CHANNEL)), metadata)
