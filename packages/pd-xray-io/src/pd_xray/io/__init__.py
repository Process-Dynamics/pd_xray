"""File format readers/writers and lazy datasets for pd_xray.

Format libraries (h5py, tifffile, Pillow) are imported on first use, so each extra is only needed for its format.
"""

from pd_xray.io.convert import compute_value_range, convert, iter_frames
from pd_xray.io.dataset import FRAME, ArrayDataset, Dataset
from pd_xray.io.formats import FileStore, UnsupportedFormatError, open_dataset, supported_extensions
from pd_xray.io.mapped import MappedDataset, map_frames
from pd_xray.io.references import reference_files, reference_frames
from pd_xray.io.sequence import FileSequence, open_sequence
from pd_xray.io.writers import supports, write_frames

__all__ = [
    "FRAME",
    "ArrayDataset",
    "Dataset",
    "FileSequence",
    "FileStore",
    "MappedDataset",
    "UnsupportedFormatError",
    "compute_value_range",
    "convert",
    "iter_frames",
    "map_frames",
    "open_dataset",
    "open_sequence",
    "reference_files",
    "reference_frames",
    "supported_extensions",
    "supports",
    "write_frames",
]
