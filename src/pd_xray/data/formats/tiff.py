import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Any

import tifffile  # type: ignore[import]

from pd_xray.data.formats.base import FormatReader


class TIFFReader(FormatReader):
    """Reader for TIFF (.tif, .tiff) files using tifffile.

    Supports single-frame and multi-frame (stack) TIFFs. Each file is read
    as a float32 array; multi-frame files return shape (T, Y, X).
    """

    _EXTENSIONS = {".tif", ".tiff"}

    def read_single_file(self, path: str | Path) -> NDArray[np.float32]:
        """Read a single-frame TIFF file."""
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found or unsupported: {path}")
        try:
            data = tifffile.imread(str(path))
            return data.astype(np.float32)
        except Exception as e:
            raise IOError(f"Error reading TIFF file {path}: {e}")

    def read(
        self,
        path: str | Path,
        start_frame: int = -1,
        end_frame: int = -1,
    ) -> NDArray[np.float32]:
        """Read a TIFF file, optionally slicing a frame range.

        Args:
            path        : Path to the TIFF file.
            start_frame : First frame index (inclusive). Negative means start at 0.
            end_frame   : Last frame index (exclusive). Negative means read until the end.

        Returns:
            2D array (Y, X) for single-frame files, or 3D array (T, Y, X) for stacks.
        """
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found or unsupported: {path}")
        try:
            with tifffile.TiffFile(str(path)) as tif:
                n_frames = len(tif.pages)
                start = 0 if start_frame < 0 else start_frame
                end = n_frames if end_frame < 0 else end_frame

                if start >= n_frames:
                    raise IndexError(
                        f"start_frame {start} is out of bounds for file with {n_frames} frames"
                    )
                if end > n_frames:
                    raise IndexError(
                        f"end_frame {end} is out of bounds for file with {n_frames} frames"
                    )

                frames = [tif.pages[i].asarray().astype(np.float32) for i in range(start, end)]

            if not frames:
                raise ValueError(
                    f"No frames read from {path} with start_frame={start_frame}, end_frame={end_frame}"
                )
            if len(frames) == 1:
                return frames[0]
            return np.stack(frames, axis=0)
        except (IndexError, ValueError):
            raise
        except Exception as e:
            raise IOError(f"Error reading TIFF file {path}: {e}")

    def read_header(self, path: str | Path) -> dict[str, Any]:
        """Read metadata from a TIFF file without loading all pixel data."""
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found or unsupported: {path}")
        try:
            with tifffile.TiffFile(str(path)) as tif:
                page = tif.pages[0]
                n_frames = len(tif.pages)
                shape = (n_frames, page.shape[0], page.shape[1]) if n_frames > 1 else page.shape
                metadata: dict[str, Any] = {
                    "shape": shape,
                    "dtype": str(np.dtype(page.dtype)),
                    "n_frames": n_frames,
                }
                if tif.imagej_metadata:
                    metadata["imagej"] = tif.imagej_metadata
                if tif.ome_metadata:
                    metadata["ome"] = tif.ome_metadata
            return metadata
        except Exception as e:
            raise IOError(f"Error reading TIFF header from {path}: {e}")

    def can_read(self, path: str | Path) -> bool:
        """Check if the file exists and has a recognised TIFF extension."""
        path = Path(path)
        if not path.exists() or not path.is_file():
            return False
        return path.suffix.lower() in self._EXTENSIONS
