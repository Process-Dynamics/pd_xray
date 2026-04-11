"""Contains the core types that will be used for different modules"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetMetadata(BaseModel):
    """Metadata describing a dataset.

    This is general enough for tomography and radiography.
    Domain-specific packages can subclass this to add fields.

    Only 'name' is required. Every other field is optional so that partial metadata is valid.

    Example:
        >>> meta = DatasetMetadata(
        ...     name="Al10Cu_scan_001",
        ...     facility="ESRF",
        ...     beamline="ID19",
        ...     energy_kev=25.0,
        ...     pixel_size_um=6.5,
        ...     sample="Al-10wt%Cu",
        ... )
        >>> meta.model_dump_json()  # Serialise to JSON for Zarr attrs
    """
    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description="Human-readable dataset name. Must be non-empty.",
    )
    experiment_id: str | None = Field(
        default=None,
        description=(
            "Unique experiment identifier, e.g. 'EXP_2025_06_15_001'. "
            "Matches the experiment.id field in the YAML config."
        ),
    )

    facility: str | None = Field(
        default=None,
        description="Synchrotron facility name, e.g. 'ESRF', 'Diamond', 'TOMCAT'.",
    )
    beamline: str | None = Field(
        default=None,
        description="Beamline identifier within the facility, e.g. 'ID19', 'I13'.",
    )

    energy_kev: float | None = Field(
        default=None,
        ge=0.0,
        description="X-ray energy in keV. Must be non-negative.",
    )
    pixel_size_um: float | None = Field(
        default=None,
        gt=0.0,
        description="Effective detector pixel size in micrometers. Must be positive.",
    )
    distance_mm: float | None = Field(
        default=None,
        gt=0.0,
        description="Sample-to-detector distance in millimetres (mm). Must be positive.",
    )
    num_time_steps: int | None = Field(
        default=None,
        ge=1,
        description="Number of time steps in dataset.",
    )
    num_projections: int | None = Field(
        default=None,
        ge=1,
        description="Number of angular projections per time step.",
    )

    sample: str | None = Field(
        default=None,
        description="Sample description, e.g. 'Al-10wt%Cu', 'Mg-25wt%Al'.",
    )
    sample_environment: str | None = Field(
        default=None,
        description=(
            "Sample environment description, e.g. 'furnace', 'cryo stage'. "
            "Relevant for in-situ experiments."
        ),
    )

    author: str | None = Field(
        default=None,
        description="Name of the scientist responsible for this dataset.",
    )
    created_at: str | None = Field(
        default=None,
        description="Timestamp when this metadata was created.",
    )

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Flexible overflow for domain-specific metadata not covered by the field above. Keys and values are "
            "arbitrary. Domain packages should document what they store here."
        ),
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        """Reject blank or whitespace-only names."""
        if not value.strip():
            raise ValueError("name must not be empty or whitespace-only")
        return value


class VolumeInfo(BaseModel):
    """Describes a data array in a Zarr store.

    Used both for introspection (what does this store contain?) and for creating new arrays (what shape/chunks should
    they have?).

    Example:
        >>> vol = VolumeInfo(
        ...     group="raw/projections",
        ...     shape=(45, 500, 2048, 2048),
        ...     dtype="float32",
        ...     chunks=(1, 1, 2048, 2048),
        ...     compression="zstd",
        ...     compression_level=3,
        ... )
        >>> vol.size_bytes  # Computed if not provided
        None
        >>> vol.ndim
        4
    """

    model_config = ConfigDict(frozen=True)

    group: str = Field(
        description=(
            "Zarr group path relative to the store root, e.g. 'raw/projections', 'processed/reconstructed'."
        ),
    )

    shape: tuple[int, ...] = Field(
        description=(
            "Array shape as a tuple of positive integers. For 4D tomography: (n_time, n_z, n_y, n_x)."
        )
    )
    dtype: str = Field(
        description=(
            "NumPy-compatible dtype string, e.g. 'float32', 'uint16', 'int8'. Used to reconstruct the dtype with "
            "np.dtype(dtype)."
        ),
    )
    chunks: tuple[int, ...] = Field(
        description=(
            "Chunk shape. Must have the same number of dimensions as shape. Each chunk dimension must be >= 1 and "
            "<= corresponding shape dimention."
        ),
    )

    compression: str | None = Field(
        default=None,
        description=(
            "Compression codec name, e.g. 'zstd', 'blosc', 'gzip'. None means no compression. In Zarr, this maps to "
            "a codec in the array's codec pipeline."
        ),
    )
    compression_level: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Compression level. Interpretation depends on codec. For zstd: 1-22 (default 3). For gzip: 1-9."
        ),
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Total uncompressed size in bytes. Computed as product(shape) * itemsize(dtype). Optional: can "
            "be None if shape or dtype are not yet known."
        ),
    )

    description: str | None = Field(
        default=None,
        description=(
            "Human-readable description of what this array contains, "
            "e.g. 'Flat-field corrected projections before reconstruction'."
        ),
    )
    attrs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional Zarr array attributes stored alongside the data.",
    )

    @model_validator(mode="after")
    def validate_shape_and_chunks_compatible(self) -> "VolumeInfo":
        """Check that chunks and shape have the same number of dimensions."""
        if len(self.chunks) != len(self.shape):
            raise ValueError(
                f"chunks has {len(self.chunks)} dimensions but shape has {
                    len(self.shape)} dimensions - they must "
                f"match. Got shape={self.shape}, chunks={self.chunks}."
            )
        for dim_idx, (chunk_dim, shape_dim) in enumerate(zip(self.chunks, self.shape, strict=False)):
            if chunk_dim < 1:
                raise ValueError(
                    f"chunks[{dim_idx}] must be >= 1, got {chunk_dim}."
                )
            if chunk_dim > shape_dim:
                raise ValueError(
                    f"chunks[{dim_idx}]={chunk_dim} exceeds shape[{dim_idx}]={
                        shape_dim}. A chunk cannot be larger "
                    "than the array along any dimension."
                )
        return self

    @field_validator("shape")
    @classmethod
    def shape_must_have_positive_dims(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject zero or negative shape dimensions."""
        for i, dim in enumerate(value):
            if dim < 1:
                raise ValueError(
                    f"shape[{i}] must be >= 1, got {dim}. "
                    "All shape dimensions must be positive."
                )
        return value

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def __repr__(self) -> str:
        compression_str = (
            f"{self.compression}:{self.compression_level}"
            if self.compression and self.compression_level is not None
            else (self.compression or "none")
        )
        size_str = (
            f"{self.size_bytes / 1e9:2f} GB"
            if self.size_bytes is not None
            else "unknown"
        )
        return (
            f"VolumeInfo("
            f"group='{self.group}', "
            f"shape={self.shape}, "
            f"dtype='{self.dtype}', "
            f"chunks={self.chunks}, "
            f"compression='{compression_str}', "
            f"size={size_str}"
            f")"
        )


class ProcessingResult(BaseModel):
    """Result of a processing operation.

    Every operation in the pipeline, reconstruction, segmentation, format conversion, ring removal, returns a
    ProcessingResult. This makes it trivial to log results, check success, and chain operations.

    Example:
        >>> result = ProcessingResult(
        ...     status='success',
        ...     message='Reconstructed 45 time steps',
        ...     duration_seconds=312.4,
        ...     metrics={'psnr': 28.5, 'peak_memory_gb': 12.3},
        ...     output_path='/data/working/Al10Cu.zarr',
        ... )
        >>> if result.status == "success":
        ...     print(f"Done in {result.duration_seconds:.1f}s")
        """
    model_config = ConfigDict(frozen=True)

    status: Literal["success", "failed", "skipped"] = Field(
        description=(
            "Outcome of the operation. "
            "'success': completed without error. "
            "'failed': could not complete. "
            "'skipped': deliberately not run (e.f. already done, or disabled in config)."
        ),
    )

    message: str | None = Field(
        default=None,
        description=(
            "Human-readable summary of what happened. "
            "For failures, this should describe what went wrong. "
            "For success, a bried description of what was done."
        ),
    )

    duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Wall-clock time for the operation in seconds.",
    )

    metrics: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Quantitative metrics from this operation. Keys and values are operation-specific. Examples: 'psnr', "
            "'dice-socre', 'peak_memory_gb', 'num_projections_processed'."
        ),
    )

    errors: list[str] = Field(
        default_factory=list,
        description=(
            "Error messages if status is 'failed'. "
            "May contain multiple errors if a batch operation had partial failures."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal warnings encountered during the operation. The operation completed but the user should be "
            "aware of these."
        ),
    )

    output_path: str | None = Field(
        default=None,
        description=(
            "Path to the primary output of this operation, if any. For reconstruction: path to the output Zarr store. "
            "For segmentation: path to the segmented array."
        ),
    )

    @model_validator(mode="after")
    def failed_status_should_have_errors(self) -> "ProcessingResult":
        """Warn if a failed result has no error messages.

        This is not enforced as a hard error because we want to allow ProcessingResult(status='failed', message='...')
        without errors=[...]. But it is a soft expectation: failed results should explain why.
        """
        return self

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def __repr__(self) -> str:
        duration_str = (
            f"{self.duration_seconds:.1f}s"
            if self.duration_seconds is not None
            else "?"
        )
        metrics_str = (
            ", ".join(f"{k}={v:.3g}" for k, v in self.metrics.items())
            if self.metrics
            else "none"
        )
        return (
            f"ProcessingResult("
            f"status='{self.status}', "
            f"duration={duration_str}, "
            f"metrics=[{metrics_str}]"
            f")"
        )


class StoreInfo(BaseModel):
    """Summary of a Zarr store, as returned by zarr_schema.get_store_info().

    This is what a scientist sees when they inspect a store, e.g.:
        >>> info = get_store_info("/data/working/Al10Cu.zarr")
        >>> print(info)

    The 'groups' list describes every array in the store, not just top-level groups. This allows pd-core to report on
    deeply nested arrrays such as 'raw/projections', 'processed/reconstructed', 'segmented/solid'.

    Example:
        >>> info = StoreInfo(
        ...     path='/data/working/Al10Cu.zarr',
        ...     schema_version="1.0",
        ...     metadata=DatasetMetadata(name="Al10Cu_scan_001"),
        ...     groups=[
        ...         VolumeInfo(
        ...             group="raw/projections",
        ...             shape=(45, 500, 2048, 2048),
        ...             dtype="float32",
        ...             chunks=(1, 1, 2048, 2048),
        ...         ),
        ...     ],
        ...     total_size_bytes=int(45 * 500 * 2048 * 2048 * 4),
        ... )
        >>> print(info)
    """

    model_config = ConfigDict(frozen=True)

    path: str = Field(
        description="Filesystem or object-storage path to the Zarr store root.",
    )
    schema_version: str = Field(
        description=(
            "pd-core schema version recorded in this store's root attributes. "
            "Compare against pd_core.constants.ZARR_SCHEMA_VERSION to check "
            "if the store needs migration."
        ),
    )

    metadata: DatasetMetadata = Field(
        description="Dataset metadata attached to the store root.",
    )
    groups: list[VolumeInfo] = Field(
        default_factory=list,
        description=(
            "Descriptions of all arrays found in this store. "
            "Ordered by group path for consistent output."
        ),
    )

    total_size_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Total uncompressed size of all arrays in the store, in bytes. "
            "None if any array's size is unknown."
        ),
    )
    compressed_size_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Total on-disk (compressed) size of the store, in bytes. "
            "None if compression ratio is not available."
        ),
    )

    created_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when the store was created.",
    )
    created_by: str | None = Field(
        default=None,
        description=(
            "Package and version that created this store, "
            "e.g. 'pd-core 0.1.0'. Stored as a Zarr root attribute."
        ),
    )

    @property
    def num_arrays(self) -> int:
        """Number of arrays in the store."""
        return len(self.groups)

    @property
    def total_size_gb(self) -> float | None:
        """Total uncompressed size in GB, or None if unknown."""
        if self.total_size_bytes is None:
            return None
        return self.total_size_bytes / 1e9

    @property
    def compression_ratio(self) -> float | None:
        """Compression ratio (uncompressed / compressed), or None if unavailable."""
        if self.total_size_bytes is None or self.compressed_size_bytes is None:
            return None
        if self.compressed_size_bytes == 0:
            return None
        return self.total_size_bytes / self.compressed_size_bytes

    def __repr__(self) -> str:
        """Notebook-friendly summary, one line per array."""
        lines: list[str] = [
            f"StoreInfo @ '{self.path}'",
            f"  schema_version : {self.schema_version}",
            f"  dataset        : {self.metadata.name}",
            f"  arrays         : {self.num_arrays}",
        ]

        if self.total_size_gb is not None:
            lines.append(f"  total size     : {
                         self.total_size_gb:.2f} GB (uncompressed)")

        if self.compression_ratio is not None:
            lines.append(f"  compression    : {
                         self.compression_ratio:.1f}× ratio")

        if self.created_at:
            lines.append(f"  created        : {self.created_at}")

        if self.groups:
            lines.append("  arrays:")
            for vol in self.groups:
                lines.append(f"    {vol}")

        return "\n".join(lines)


class TimeStep(BaseModel):
    """Describes one time step in a 4D dataset.

    Used when iterating over a time series to know what angular range and
    projection count to expect at each step. This may vary between steps
    in practice (e.g. if a projection was dropped due to beam instability).

    Example:
        >>> step = TimeStep(index=5, num_projections=499)
        >>> step.index
        5
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(
        ge=0,
        description="Zero-based time step index.",
    )
    num_projections: int | None = Field(
        default=None,
        ge=1,
    )
    angle_start_deg: float | None = Field(
        default=None,
        description="Starting rotation angle in degrees.",
    )
    angle_end_deg: float | None = Field(
        default=None,
        description="Ending rotation angle in degrees.",
    )
    source_path: str | None = Field(
        default=None,
        description=(
            "Path (local, NAS, or S3) to the raw data file for this time step. "
            "Stored for provenance — tells us exactly which file was used."
        ),
    )

    @property
    def angular_range_deg(self) -> float | None:
        """Total angular range in degrees, or None if start/end not set."""
        if self.angle_start_deg is None or self.angle_end_deg is None:
            return None
        return abs(self.angle_end_deg - self.angle_start_deg)

    def __repr__(self) -> str:
        parts = [f"TimeStep(index={self.index}"]
        if self.num_projections is not None:
            parts.append(f"projections={self.num_projections}")
        if self.angular_range_deg is not None:
            parts.append(f"angle_range={self.angular_range_deg:.1f}°")
        return ", ".join(parts) + ")"


class ChunkStrategy(BaseModel):
    """Recommended chunk shape for a given access pattern.

    The Data Reader uses this when creating a Zarr store to pick chunk shapes
    that optimise for the expected read pattern downstream.

    The shape uses -1 as a sentinel for "full extent along this dimension",
    which is resolved by zarr_schema.resolve_chunks() before use.

    Example:
        >>> strategy = ChunkStrategy(
        ...     pattern="single_timestep",
        ...     chunks=(-1, 64, -1, -1),
        ...     description="Optimised for loading one full volume at a time",
        ... )
    """

    model_config = ConfigDict(frozen=True)

    pattern: str = Field(
        description=(
            "Access pattern name, e.g. 'single_timestep', 'z_slice_timeseries', "
            "'ml_patch'. Used for logging and documentation."
        ),
    )
    chunks: tuple[int, ...] = Field(
        description=(
            "Chunk shape. Use -1 for 'full extent along this dimension'. "
            "Resolved against the actual array shape by resolve_chunks()."
        ),
    )
    description: str | None = Field(
        default=None,
        description="Human-readable explanation of when to use this strategy.",
    )

    @field_validator("chunks")
    @classmethod
    def chunks_must_be_nonzero_or_minus_one(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Each chunk dimension must be a positive integer or the sentinel -1."""
        for i, dim in enumerate(value):
            if dim == 0 or (dim < -1):
                raise ValueError(
                    f"chunks[{i}]={dim} is invalid. "
                    "Chunk dimensions must be positive integers or -1 (full extent)."
                )
        return value
