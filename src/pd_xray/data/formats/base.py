from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Any
from pathlib import Path


class FormatReader(ABC):
    """Abstract base for raw data format readers.

    A FormatReader knows how to open ONE type of file and extract:
    1. Pixel data as a float32 NumPy array
    2. Header metadata as a plain dict

    It operates on LOCAL files only. If the file is on a NAS, the StorageBackend downloads it first, then the
    FormatReader reads the local copy.
    """

    @abstractmethod
    def read(self, path: str | Path) -> NDArray[np.float32]:
        """Read pixel data from a single file.

        Returns:
            2D, 3D, or 4D array of pixel values as float32 and shape (T, Z)YX where T is time steps.
        """
        ...

    @abstractmethod
    def read_header(self, path: str | Path) -> dict[str, Any]:
        """Read metadata without loading pixel data.

        This is much faster than read() for large files. Use it to:
        - Determine array shape before pre-allocating
        - Check file type before attempting a full read
        - Gather metadata for Zarr attrs

        Returns:
            Dict with at least: {"shape": tuple, "dtype": str}. Additional keys depend on the format.
        """
        ...

    @abstractmethod
    def can_read(self, path: str | Path) -> bool:
        """Check if this reader can handle the give file.

        Checks file extension and optionally magic bytes. Does NOT attempt a full read, so it's fast.
        """
        ...