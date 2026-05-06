import numpy as np
from numpy.typing import NDArray
import pyvista as pv

from pd_xray.data.formats.base import FormatReader
from pd_xray.core import get_logger
from pd_xray.core import T

logger = get_logger(__name__)


class VTIReader(FormatReader):
    """Reader for VTI files using PyVista for conversions."""

    def can_read(self, path):
        return NotImplementedError

    def read(self, path):
        return NotImplementedError
    
    def read_header(self, path):
        return NotImplementedError

    def convert_from_np(
        self,
        array: NDArray[T],
        file_name: str = "tiff_converted.vti",
        spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> None:
        """Save a 3D numpy array as a .vti file for ParaView
        
        Args:
            volume   : 3D numpy array (Z, Y, X)
            filename : output file name
            spacing  : voxel spacing
        """
        # Ensure that it's numpy array
        volume = np.asarray(array)

        # VTK expects data in (X, Y, Z) order
        volume = np.transpose(volume, (2, 1, 0))

        grid = pv.ImageData()
        grid.dimensions = volume.shape
        grid.spacing = spacing
        grid.point_data["values"] = volume.flatten(order="F")
        grid.save(file_name)
        logger.info(f"Saved numpy array of shape {volume.shape} into file {file_name}")