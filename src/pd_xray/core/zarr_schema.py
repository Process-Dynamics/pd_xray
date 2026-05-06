"""Zarr store schema definition, creation, validation, and inspection.

This module defines what a "valid" pd-core Zarr store looks like. It provides functions to create new conformant
stores, validate exisitng ones, and inspect their contents. It knows nothing about sinograms, projections, or
dendrites, those concepts belong in downstream packages.

Schema Contract
---------------
Every pd-core Zarr store must have the following attributes on its root group:

    root.attrs["schema_version"] -> str     e.g. "1.0"
    root.attrs["name"]           -> str     human-readable dataset name
    root.attrs["created_at"]     -> str     ISO 8601 timestamp
    root.attrs["created_by"]     -> str     e.g. "pd-core 0.1.0"

All other attributes are optional. Domain packages add their own groups (raw/, processed/, segmented/, analysis/) and
arrays. The schema validates only the root-level contract, domain groups are the responsiblity of each downstream.

Usage
-----
    from pd_core.zarr_schema import create_store, create_array, validate_store, get_store_info
    from pd_core.types import DatasetMetadata

    # Create a new conformant store
    root = create_store("/data/Al10Cu.zarr",
                        DatasetMetadata(name="Al10Cu_001"))

    # Add a domain-specific array
    arr = create_array(root, "raw/projections", shape=(45, 500, 2048, 2048), chunks=(1, 1, 2048, 2048),
                        dtype="float32")

    # Validate later
    is_valid, issues = validate_store("/data/Al10Cu.zarr")

    # Inspect
    info = get_store_info("/data/Al10Cu.zarr")
    print(info)
"""
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zarr
import zarr.codecs
import zarr.storage

from pd_xray import __version__
from pd_xray.core.constants import (
    ZARR_DEFAULT_COMPRESSION,
    ZARR_DEFAULT_COMPRESSION_LEVEL,
    ZARR_SCHEMA_VERSION,
)
from pd_xray.core.exceptions import SchemaError
from pd_xray.core.logging import get_logger
from pd_xray.core.types import DatasetMetadata, StoreInfo, VolumeInfo

logger = get_logger(__name__)


_REQUIRED_ROOT_ATTRS: tuple[str, ...] = (
    "schema_version",
    "name",
    "created_at",
    "created_by",
)

_KNOWN_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})

_CODEC_MAP: dict[str, zarr.codecs.BloscCname] = {
    "zstd": zarr.codecs.BloscCname.zstd,
    "zlib": zarr.codecs.BloscCname.zlib,
    "blosclz": zarr.codecs.BloscCname.blosclz,
    "lz4": zarr.codecs.BloscCname.lz4,
    "lz4hc": zarr.codecs.BloscCname.lz4hc,
    "snappy": zarr.codecs.BloscCname.snappy,
}


def _build_compressor(
    compression: str | None,
    compression_level: int | None,
) -> list[zarr.codecs.BloscCodec]:
    """Build a Zarr v3 compressor list from a codec name and level.

    In Zarr v3, compression is specified as a list of codec objects passed to the 'compressors' parameter of
    create_array. This helper translates the human-readable (name, level) pair used in configs and VolumeInfo into
    the codec object from that Zarr v3 expects.

    Args:
        compression       : Codec name, e.g. "zstd", "gzip", "blosc". If None, falls back to the package default
        compression_level : Compression level for codecs that support it. If None, falls back to default level

    Returns:
        A single-element list containing the codec object ready for Zarr v3.

    Raises:
        SchemaError if the codec name is not supported
    """
    codec_name = compression if compression is not None else ZARR_DEFAULT_COMPRESSION
    level = compression_level if compression_level is not None else ZARR_DEFAULT_COMPRESSION_LEVEL

    codec_class = _CODEC_MAP.get(codec_name)

    if codec_class is None:
        supported = ", ".join(sorted(_CODEC_MAP))
        raise SchemaError(
            f"Compression codec '{codec_name}' is not supported. Supported codecs: {
                supported}. To add support for a "
            "new codec, extend _CODEC_MAP in zarr_schema.py"
        )

    codec = zarr.codecs.BloscCodec(cname=codec_class, clevel=level)
    return [codec]


def create_store(
    path: str | Path,
    metadata: DatasetMetadata,
    overwrite: bool = False,
) -> zarr.Group:
    """Create a new Zarr store conforming to the pd-core schema.

    Writes the root group with the mandatory schema attributes. Does NOT create any data arrays, those are added by
    downstream packages via create_array().

    Downstream packages are expected to create their own top-level groups:
        pd-data   -> raw/
        tomo-proc -> processed/
        tomo-ml   -> segmented/, analysis/

    Args:
        path      : Filesystem path for the new Zarr store directory.
        metadata  : Dataset metadata to attach to the root group.
        overwrite : If True, silently delete and recreate an existing store. If False (default),
                    raise SchemaError if the store exists.

    Returns:
        The root zarr.Group, open in append mode (ready for writing arrays).

    Raises:
        SchemaError: If the store already exists and overwrite=False.

    Example:
        >>> root = create_store(
        ...     "/data/working/Al10Cu.zarr",
        ...     DatasetMetadata(name="Al10Cu_scan_001", facility="ESRF"),
        ... )
        >>> proc_group = root.require_group("processed")
    """
    store_path = Path(path)

    if store_path.exists():
        if not overwrite:
            raise SchemaError(
                f"Zarr store already exists at '{
                    store_path}'. Set overwrite=True to replace it, or choose a "
                "different path. Existing stores are protected by default to prevent accidental data loss."
            )
        logger.warning(f"Overwriting existing store at {store_path}")

    root: zarr.Group = zarr.open_group(str(store_path), mode="w")

    root_attrs: dict[str, Any] = {
        "schema_version": ZARR_SCHEMA_VERSION,
        "name": metadata.name,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "created_by": f"pd-core {__version__}",
    }

    # Flatten the metadata model into the root attributes. We store each non-None field directly so that tools other
    # than pd-core can read them without knowing pydantic.
    for field_name, value in metadata.model_dump().items():
        if value is None:
            continue
        if field_name == "extra":
            # Expand the extra dict directly into root attrs.
            for extra_key, extra_val in value.items():
                root_attrs[extra_key] = extra_val
        else:
            root_attrs[field_name] = value

    root.attrs.update(root_attrs)

    logger.info(f"Created Zarr store at {store_path} (schema_version={
                ZARR_SCHEMA_VERSION}, name={metadata.name}).")
    return root


def create_array(
    group: zarr.Group,
    name: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str = "float32",
    compression: str | None = None,
    compression_level: int | None = None,
    attrs: dict[str, Any] | None = None,
) -> zarr.Array:
    """Create a new array within a Zarr group.

    Handles -1 chunk sentinels, applies default compression if not specified, and attaches any provided attributes.
    Intermediate ghroups in 'name' are created automatically (e.g. "raw/projections" creates "raw/" if absent).

    Args:
        group             : the parent zarr group to create the array in.
        name              : name (or path) of the new array, e.g. "projections" or "raw/projections". Intermediate
                            groups are created automatically.
        shape             : full array shape. all dimensions must be >= 1.
        chunks            : Chunk shape. Use -1 for "full extent along this dimesion". Resolved via resolve_chunks()
                            before creating the array.
        dtype             : NumPy-compatible dtype string. Defaults to "float32".
        compression       : Codec name, e.g. "zstd", "gzip". None -> package default.
        compression_level : compression level. None -> package default
        attrs             : Optional dict of attributes to attach to the new array.

    Returns:
        The newly created zarr.Array.

    Raises:
        ValueError: From resolve_chunks() if chunk dimensions are invalid.
        SchemaError: If the compression codec is not supported.

    Example:
        >>> arr = create_array(
        ...     root,
        ...     "raw/projections",
        ...     shape=(45, 600, 2048, 2048),
        ...     chunks=(1, 1, 2048, 2048),
        ...     dtype="float32",
        ...     compression="zstd",
        ...     compression_level=3,
        ...     attrs={"description": "Flat-field corrected projections"},
    """
    resolved_chunks = resolve_chunks(shape, chunks)
    _build_compressor(compression, compression_level)

    array: zarr.Array = group.create_array(
        name,
        shape=shape,
        chunks=resolved_chunks,
        dtype=dtype,
        compressors=zarr.codecs.BloscCodec(clevel=9),
        overwrite=False,
    )

    if attrs:
        array.attrs.update(attrs)

    logger.debug(
        f"Created array {name} shape={shape} chunks={
            resolved_chunks}, dtype={dtype} "
        f"compression={compression or ZARR_DEFAULT_COMPRESSION}"
    )
    return array


def validate_store(path: str | Path) -> tuple[bool, list[str]]:
    """Validate a Zarr store against the pd-core schema.

    Checks the root-level contract only. domain-specific group structure (raw/, processed/, etc.) is the responsibility
    of each downstream package.

    Checks performed:
        1. The store path exists and is readable.
        2. The root group has all required attributes.
        3. schema_version is a recognised version string.
        4. All arrays in the store have valid chunk shapes (no chunk dimension exceeds its array dimension).

    Args:
        path : Path to the Zarr store to validate

    Returns:
        A tuple (is_valid, issues) where:
        - is_valid is True if all checks passed, False otherwise.
        - issues is a list of human-readable problem descriptions. Empty if is_valid is True.

    Example:
        >>> is_valid, issues = validate_store("/data/Al10Cu.zarr")
        >>> if not is_valid:
        ...     for issue in issues:
        ...         print(f"    x {issue}")
    """
    store_path = Path(path)
    issues: list[str] = []

    if not store_path.exists():
        issues.append(
            f"Store path does not exist: '{store_path}'. "
            "Check the path and ensure the store has been created."
        )
        return False, issues

    try:
        root: zarr.Group = zarr.open_group(str(store_path), mode="r")
    except Exception as e:
        issues.append(
            f"Could not open store at '{store_path}': {e}. "
            "The store may be corrupt or not a valid Zarr store."
        )
        return False, issues

    root_attrs = dict(root.attrs)

    for required_key in _REQUIRED_ROOT_ATTRS:
        if required_key not in root_attrs:
            issues.append(
                f"Required root attribute '{required_key}' is missing. "
                "This attribute must be set when creating the store via create_store()."
            )

    schema_version = root_attrs.get("schema_version")
    if schema_version is not None and schema_version not in _KNOWN_SCHEMA_VERSIONS:
        issues.append(
                f"schema_version='{
                    schema_version}' is not a recognised version."
                f"Known versions: {sorted(_KNOWN_SCHEMA_VERSIONS)}. "
                "The store may have been created by a newer version of pd-core."
            )

    try:
        for member_path, member in root.members(max_depth=None):
            if not isinstance(member, zarr.Array):
                continue
            for dim_idx, (shape_dim, chunk_dim) in enumerate(zip(member.shape, member.chunks, strict=False)):
                if chunk_dim > shape_dim:
                    issues.append(
                        f"Array '{member_path}': chunk dimension {dim_idx} "
                        f"({chunk_dim}) exceeds array dimension ({shape_dim}). "
                        "This indicates the store was created with inconsistent parameters."
                    )
    except Exception as e:
        issues.append(f"Error while walking store hierarchy: {
                      e}. The store structure may be partially corrupt.")

    is_valid = len(issues) == 0

    if is_valid:
        logger.debug(
            f"Store '{store_path}' is valid (schema_version={schema_version}).")
    else:
        logger.warning(f"Store at '{store_path}' failed validation with {
                       len(issues)} issue(s).")
    return is_valid, issues


def get_store_info(path: str | Path) -> StoreInfo:
    """Read and return summary information about a Zarr store.

    Walks the full store hierarchy and returns a StoreInfo object describing the schema version, metadata, all arrays,
    and size information. This is the function that powers the 'inspect' CLI command.

    Args:
        path : Path to the Zarr store to inspect.

    Returns:
        A StoreInfo object. Always returns an object, if the store cannot be read, raises SchemaError rather than
        returning partial result.

    Raises:
        SchemaError: If the store cannot be opened or is not a valid Zarr store.

    Example:
        >>> info = get_store_info("/data/Al10Cu.zarr")
        >>> print(info)  # Notebook-friendly summary
        >>> info.metadata.facility
        'ESRF'
    """
    store_path = Path(path)

    if not store_path.exists():
        raise SchemaError(
            f"Cannot inspect store: path does not exist: '{
                store_path}'. Ensure the store has been created."
        )

    try:
        root: zarr.Group = zarr.open_group(str(store_path), mode="r")
    except Exception as e:
        raise SchemaError(
            f"Cannot open store at '{store_path}': {e}. "
            "The path exists but may not be a valid Zarr store."
        ) from e

    root_attrs = dict(root.attrs)

    known_metadata_fields = set(DatasetMetadata.model_fields)
    metadata_kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    _schema_managed_keys = frozenset(
        {"schema_version", "created_at", "created_by"})

    for key, value in root_attrs.items():
        if key in _schema_managed_keys:
            continue
        if key in known_metadata_fields:
            metadata_kwargs[key] = value
        else:
            extra[key] = value

    if extra:
        metadata_kwargs["extra"] = extra

    if "name" not in metadata_kwargs:
        metadata_kwargs["name"] = "<unknown>"

    try:
        metadata = DatasetMetadata.model_validate(metadata_kwargs)
    except Exception:
        metadata = DatasetMetadata(
            name=metadata_kwargs.get("name", "<unknown>"))

    volume_infos: list[VolumeInfo] = []
    total_uncompresed: int = 0
    total_compressed: int = 0
    size_unknown = False

    for member_path, member in root.members(max_depth=None):
        if not isinstance(member, zarr.Array):
            continue

        compression_name: str | None = None
        compression_level_val: int | None = None
        for codec in member.metadata.codecs:
            codec_name = type(codec).__name__
            if codec_name == "BytesCodec":
                continue  # First codec in zarr.Array is always the "identity" codec that does no compression.
            for human_name, codec_cls in _CODEC_MAP.items():
                if codec.cname == codec_cls:
                    compression_name = human_name
                    if hasattr(codec, "level"):
                        compression_level_val = codec.level
                    elif hasattr(codec, "clevel"):
                        compression_level_val = codec.clevel
                    break

        uncompressed = member.nbytes
        try:
            compressed = member.nbytes_stored()
        except Exception:
            compressed = 0
            size_unknown = True

        total_uncompresed += uncompressed
        total_compressed += compressed

        vol = VolumeInfo(
            group=member_path,
            shape=tuple(member.shape),
            dtype=str(member.dtype),
            chunks=tuple(member.chunks),
            compression=compression_name,
            compression_level=compression_level_val,
            size_bytes=uncompressed,
            attrs=dict(member.attrs),
        )
        volume_infos.append(vol)

    volume_infos.sort(key=lambda v: v.group)

    return StoreInfo(
        path=str(store_path),
        schema_version=root_attrs.get("schema_version", "<missing>"),
        metadata=metadata,
        groups=volume_infos,
        total_size_bytes=total_uncompresed if not size_unknown else None,
        compressed_size_bytes=total_compressed if not size_unknown else None,
        created_at=root_attrs.get("created_at"),
        created_by=root_attrs.get("created_by"),
    )


def resolve_chunks(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolve chunk dimensions, replacing -1 with full extent.

    The -1 sentinel means "use full array extent along this dimension", matching the convention used in the YAML config
    schema. All -1 values must be resolved to concrete integers before passing to Zarr.

    Args:
        shape: Array shape. All dimensions muyst be >= 1.
        chunks: Desired chunk shape. Use -1 for "full extent along this dim.". All other values must be >= 1.

    Returns:
        Resolved chunk shape with no -1 values remaining.

    Raises:
        ValueError: If shape and chunks have different numbers of dimensions.
        ValueError: If any chunk dimension is 0, < -1, or exceeds its corresponding shape dimension.

    Examples:
        >>> resolve_chunks((45, 2048, 2048), (1, -1, -1))
        (1, 2048, 2048)

        >>> resolve_chunks((100, 200), (10, 20))
        (10, 20)

        >>> resolve_chunks((45, 2048, 2048), (1, 64, 64))
        (1, 64, 64)
    """
    if len(shape) != len(chunks):
        raise ValueError(
            f"shape and chunks must have the same number of dimensions. "
            f"Got shape with {len(shape)} dims {shape} and chunks with {
                len(chunks)} dims {chunks}."
        )

    resolved: list[int] = []
    for dim_idx, (shape_dim, chunk_dim) in enumerate(zip(shape, chunks, strict=False)):
        if chunk_dim == -1:
            resolved.append(shape_dim)
        elif chunk_dim < 1:
            raise ValueError(
                f"chunks[{dim_idx}]={
                    chunk_dim} is invalid. Chunk dimensions must be positive integers or "
                "-1 (full extent). Zero and values less than -1 are not allowed."
            )
        elif chunk_dim > shape_dim:
            raise ValueError(
                f"chunks[{dim_idx}]={chunk_dim} exceeds shape[{dim_idx}]={
                    shape_dim}. A chunk dimension cannot be "
                "larger than the corresponding array dimension. Consider using -1 (full extent) instead of {chunk_dim}"
            )
        else:
            resolved.append(chunk_dim)
    return tuple(resolved)
