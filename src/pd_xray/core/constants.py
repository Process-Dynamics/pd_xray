"""Package-wide constants"""


ZARR_DEFAULT_COMPRESSION: str = "zstd"
ZARR_DEFAULT_COMPRESSION_LEVEL: int = 3
ZARR_SCHEMA_VERSION: str = "1.0"

ZARR_GROUP_RAW: str = "raw"
ZARR_GROUP_PROCESSED: str = "processed"
ZARR_GROUP_SEGMENTED: str = "segmented"
ZARR_GROUP_ANALYSIS: str = "analysis"

SUPPORTED_RAW_FORMATS: frozenset[str] = frozenset(
    {"edf", "hdf5", "h5", "nxs", "tiff", "tif"})

DEFAULT_MAX_WORKERS: int = 4
DEFAULT_CHUNK_SIZE_MB: int = 64        # Target chunk size in MB
MIN_FREE_MEMORY_FRACTION: float = 0.2  # Keep 20% RAM free
