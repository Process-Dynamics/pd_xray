import dask
import dask.array as da
import fabio  # type: ignore[import]
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Any

from pd_xray.data.formats.base import FormatReader


class EDFReader(FormatReader):
    """Reader for ESRF Data Format (.edf) files using fabio.

    EDF files can contain single or multiple projections. The header should contain beamline metadata.

    fabio.open() reads both header and data. For read_header(), we still call fabio.open() but discard the data.
    """

    def read_single_file(self, path: str | Path) -> NDArray[np.float32]:
        """Read a single EDF file (e.g., dark or flat field)."""
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found: {path}")
        try:
            with fabio.open(str(path)) as img:
                data = img.data
            return data.astype(np.float32)
        except Exception as e:
            raise IOError(f"Error reading EDF file {path}: {e}")
        
    def read(
        self,
        path: str | Path,
        start_frame: int = -1,
        end_frame: int = -1,
    ) -> NDArray[np.float32]:
        """Read a stack of projections from large EDF file.

        Args:
            path        : Path to the EDF file containing projections.
            start_frame : First frame index (inclusive). Negative means read from the start.
            end_frame   : Last frame index (exclusive). Negative means read until the end.
        
        Returns:
            3D array of shape (T, Y, X) where T is time steps.
        """
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        start_frame = 0 if start_frame < 0 else start_frame

        frame_list: list[NDArray[np.float32]] = []
        try:
            with fabio.open(str(path)) as img:
                end_frame = img.nframes if end_frame < 0 else end_frame
                if start_frame >= img.nframes:
                    raise IndexError(f"start_frame {start_frame} is out of bounds for file with {img.nframes} frames")
                if end_frame > img.nframes:
                    raise IndexError(f"end_frame {end_frame} is out of bounds for file with {img.nframes} frames")
                for i in range(start_frame, end_frame):
                    try:
                        frame = img.getframe(i).data.astype(np.float32)
                        frame_list.append(frame)
                    except Exception as e:
                        raise IOError(f"Error reading frame {i} from EDF file {path}: {e}")
            if not frame_list:
                raise ValueError(f"No frames read from EDF file {path} with start_frame={start_frame} and end_frame={end_frame}")
            return np.stack(frame_list, axis=0)
        except Exception as e:
            raise IOError(f"Error reading EDF file {path}: {e}")
    
    def read_with_index(
        self,
        paths: list[str],
        start_frame: int,
        end_frame: int,
    ) -> da.Array:
        """Read a list of EDF files and stack them into a lazy 3D Dask array."""
        frame_shape = self.read_header(path=paths[0])["shape"]
        height = frame_shape[0]
        width = frame_shape[1]
        delayed_read = dask.delayed(self.read)(paths[0], start_frame, end_frame),
        return da.from_delayed(delayed_read, shape=(height, width), dtype=np.float32)


    def read_header(self, path: str | Path) -> dict[str, Any]:
        """Read metadata from EDF header without loading pixel data."""
        path = Path(path)
        if not self.can_read(path):
            raise FileNotFoundError(f"File not found: {path}")
        try:
            with fabio.open(str(path)) as img:
                header = dict(img.header)
                header["shape"] = img.data.shape
                header["dtype"] = str(img.data.dtype)
            return header
        except Exception as e:
            raise IOError(f"Error reading EDF header from file {path}: {e}")
    
    def can_read(self, path: str | Path) -> bool:
        """Check if the file is an EDF file by extension and magic bytes."""
        path = Path(path)
        if not path.exists() or not path.is_file():
            return False
        if path.suffix.lower() != ".edf":
            return False
        return True