"""Interactive 3D visualisation of a segmented tomography volume using napari.

Loads segmentation mask TIFFs produced by ``segment_volume``, assembles them
into a (Z, Y, X) label volume, applies optional cropping / slicing /
downsampling, and launches an interactive napari viewer.

Python usage::

    from pd_xray.visualisation import visualise_tomogram

    visualise_tomogram(
        mask_dir="/data/scan_001_masks/",
        classes=[1, 2],                        # show eutectic + dendrite only
        crop=(50, 50, 50, 50),                 # trim 50 px from every edge
        downsample=2,                          # half resolution for speed
        dimensions=((0, 500), (0, 800), (0, 800)),  # (z, y, x) sub-volume
    )

CLI usage::

    python -m pd_xray.visualisation.visualise_tomogram \\
        --mask_dir /data/scan_001_masks/ \\
        --classes 1 2 \\
        --crop 50 50 50 50 \\
        --downsample 2 \\
        --z 0 500 --y 0 800 --x 0 800
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pd_xray.core.logging import get_logger
from pd_xray.data.formats.tiff import TIFFReader

logger = get_logger(__name__)

# Re-use the project's TIFFReader for consistency with the rest of the pipeline.
_reader = TIFFReader()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_volume(
    mask_dir: Path,
    glob_pattern: str,
    z_range: Optional[tuple[int, int]],
) -> NDArray[np.int32]:
    """Read sorted mask TIFFs and stack into a (Z, Y, X) int32 volume.

    Only the slices in *z_range* are read from disk — avoids loading the full
    stack when a z sub-volume is requested.
    """
    files = sorted(mask_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob_pattern}' found in {mask_dir}"
        )

    if z_range is not None:
        z0, z1 = z_range
        files = files[z0:z1]
        if not files:
            raise ValueError(
                f"z_range={z_range} produced no slices "
                f"(directory has {len(sorted(mask_dir.glob(glob_pattern)))} files)."
            )

    logger.info("Loading %d slices from %s …", len(files), mask_dir)
    slices = [_reader.read(f).astype(np.int32) for f in files]
    volume = np.stack(slices, axis=0)  # (Z, Y, X)
    logger.info("Loaded volume  shape=%s  dtype=%s", volume.shape, volume.dtype)
    return volume


def _apply_crop(
    volume: NDArray[np.int32],
    crop: tuple[int, int, int, int],
) -> NDArray[np.int32]:
    """Remove pixels from the edges of every YX slice.

    Args:
        volume: (Z, Y, X) array.
        crop:   (left, right, top, bottom) pixel counts to remove.
    """
    left, right, top, bottom = crop
    h, w = volume.shape[1], volume.shape[2]
    y_end = h - bottom if bottom > 0 else h
    x_end = w - right if right > 0 else w
    cropped = volume[:, top:y_end, left:x_end]
    logger.info("After crop %s → shape=%s", crop, cropped.shape)
    return cropped


def _apply_yx_slice(
    volume: NDArray[np.int32],
    y_range: Optional[tuple[int, int]],
    x_range: Optional[tuple[int, int]],
) -> NDArray[np.int32]:
    if y_range is not None:
        volume = volume[:, y_range[0]:y_range[1], :]
    if x_range is not None:
        volume = volume[:, :, x_range[0]:x_range[1]]
    if y_range is not None or x_range is not None:
        logger.info("After YX slice  shape=%s", volume.shape)
    return volume


def _filter_classes(
    volume: NDArray[np.int32],
    classes: list[int],
) -> NDArray[np.int32]:
    """Zero-out every voxel not in *classes* (background becomes 0)."""
    keep = np.zeros(volume.shape, dtype=bool)
    for cls in classes:
        keep |= volume == cls
    return np.where(keep, volume, 0)


def _downsample(
    volume: NDArray[np.int32],
    factor: int,
) -> NDArray[np.int32]:
    """Nearest-neighbour downsample along all three axes by *factor*."""
    ds = volume[::factor, ::factor, ::factor]
    logger.info("After downsample ×%d  shape=%s", factor, ds.shape)
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualise_tomogram(
    mask_dir: str | Path,
    *,
    image_dir: Optional[str | Path] = None,
    classes: Optional[list[int]] = None,
    crop: Optional[tuple[int, int, int, int]] = None,
    downsample: Optional[int] = None,
    dimensions: Optional[
        tuple[
            Optional[tuple[int, int]],  # z: (start, end)
            Optional[tuple[int, int]],  # y: (start, end)
            Optional[tuple[int, int]],  # x: (start, end)
        ]
    ] = None,
    glob_pattern: str = "*.tif",
    label_opacity: float = 0.7,
    class_names: Optional[dict[int, str]] = None,
    ndisplay: int = 3,
) -> None:
    """Launch an interactive napari viewer for a segmented tomography volume.

    Args:
        mask_dir:      Directory of segmentation mask TIFFs — one file per Z
                       slice, sorted lexicographically by filename.
        image_dir:     Optional directory of raw greyscale TIFFs to overlay
                       beneath the labels (same filenames as *mask_dir*).
        classes:       Class IDs to display. ``None`` shows all classes found
                       in the volume. Non-selected classes are set to 0
                       (transparent in napari).
        crop:          ``(left, right, top, bottom)`` pixel counts to remove
                       from every slice before visualisation.
        downsample:    Integer stride applied to every axis (Z, Y, X).
                       E.g. ``2`` halves the resolution in all directions.
                       Applied last, after all slicing and cropping.
        dimensions:    Sub-volume selection as
                       ``((z0, z1), (y0, y1), (x0, x1))``.  Any element may
                       be ``None`` to keep the full axis.
                       Z slicing is applied at load time (only needed slices
                       are read); Y/X slicing is applied after cropping.
        glob_pattern:  Glob used to collect mask files. Default ``"*.tif"``.
        label_opacity: Opacity of the labels layer (0–1). Default ``0.7``.
        class_names:   ``{class_id: name}`` mapping shown in the napari layer
                       control panel tooltip. Optional cosmetic only.
        ndisplay:      Number of display dimensions. ``3`` opens in 3D volume
                       mode (default); ``2`` opens in 2D slice mode.
    """
    import napari

    mask_dir = Path(mask_dir)

    # ---- Unpack dimension ranges ----------------------------------------
    z_range = y_range = x_range = None
    if dimensions is not None:
        z_range, y_range, x_range = dimensions

    # ---- Load (only the z-slice window) ----------------------------------
    volume = _load_volume(mask_dir, glob_pattern, z_range)

    # ---- Spatial pre-processing (order matters) --------------------------
    if crop is not None:
        volume = _apply_crop(volume, crop)

    if y_range is not None or x_range is not None:
        volume = _apply_yx_slice(volume, y_range, x_range)

    # ---- Class filtering -------------------------------------------------
    if classes is not None:
        logger.info("Showing classes: %s", classes)
        volume = _filter_classes(volume, classes)

    # ---- Downsample ------------------------------------------------------
    if downsample is not None and downsample > 1:
        volume = _downsample(volume, downsample)

    logger.info("Final volume  shape=%s  unique_labels=%s", volume.shape, np.unique(volume).tolist())

    # ---- Build napari colour map -----------------------------------------
    # Build a label colour map from class_names keys if provided, otherwise
    # let napari assign colours automatically.
    color: Optional[dict] = None
    if class_names is not None:
        # napari Labels accepts {label_id: colour_name_or_rgba}
        # We leave the colour values as None so napari auto-picks them, but
        # we use class_names purely for the metadata dict below.
        pass

    # ---- Launch viewer ---------------------------------------------------
    viewer = napari.Viewer(title="pd_xray — Segmented Volume", ndisplay=ndisplay)

    # Optional raw image underlay
    if image_dir is not None:
        image_dir = Path(image_dir)
        img_files = sorted(image_dir.glob(glob_pattern))
        if img_files:
            if z_range is not None:
                img_files = img_files[z_range[0]:z_range[1]]
            logger.info("Loading %d raw image slices …", len(img_files))
            img_slices = [_reader.read(f) for f in img_files]
            raw = np.stack(img_slices, axis=0)
            if crop is not None:
                raw = _apply_crop(raw, crop)  # type: ignore[arg-type]
            if y_range is not None or x_range is not None:
                raw = _apply_yx_slice(raw, y_range, x_range)  # type: ignore[arg-type]
            if downsample is not None and downsample > 1:
                raw = raw[::downsample, ::downsample, ::downsample]
            viewer.add_image(
                raw,
                name="raw",
                colormap="gray",
                blending="additive",
            )
        else:
            logger.warning("image_dir provided but no files matched '%s'", glob_pattern)

    # Metadata for the labels layer (shown in napari's layer tooltip).
    metadata: dict = {}
    if class_names is not None:
        metadata["class_names"] = class_names

    viewer.add_labels(
        volume,
        name="segmentation",
        opacity=label_opacity,
        metadata=metadata,
    )

    # Print a quick legend to the console so the user knows which colour = which class.
    if class_names is not None:
        present = np.unique(volume)
        logger.info("Classes in view:")
        for cid in present:
            if cid == 0:
                continue
            name = class_names.get(int(cid), str(cid))
            logger.info("  %d → %s", cid, name)

    napari.run()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualise a segmented tomography volume interactively with napari.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mask_dir", required=True, help="Directory of mask TIFF slices.")
    p.add_argument(
        "--image_dir",
        default=None,
        help="Optional directory of raw greyscale TIFFs to overlay.",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        metavar="CLS",
        help="Class IDs to display (space-separated). Omit to show all.",
    )
    p.add_argument(
        "--crop",
        nargs=4,
        type=int,
        default=None,
        metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"),
        help="Pixel counts to remove from each edge of every slice.",
    )
    p.add_argument(
        "--downsample",
        type=int,
        default=None,
        metavar="N",
        help="Downsample every axis by this integer stride.",
    )
    p.add_argument("--z", nargs=2, type=int, default=None, metavar=("Z0", "Z1"))
    p.add_argument("--y", nargs=2, type=int, default=None, metavar=("Y0", "Y1"))
    p.add_argument("--x", nargs=2, type=int, default=None, metavar=("X0", "X1"))
    p.add_argument("--glob", default="*.tif", help="Glob pattern for mask files.")
    p.add_argument(
        "--opacity",
        type=float,
        default=0.7,
        help="Label layer opacity (0–1).",
    )
    return p


if __name__ == "__main__":
    from pd_xray.core.logging import setup_logging

    setup_logging(level="INFO")
    args = _build_parser().parse_args()

    dims = None
    if any(v is not None for v in (args.z, args.y, args.x)):
        dims = (
            tuple(args.z) if args.z else None,
            tuple(args.y) if args.y else None,
            tuple(args.x) if args.x else None,
        )

    visualise_tomogram(
        mask_dir=args.mask_dir,
        image_dir=args.image_dir,
        classes=args.classes,
        crop=tuple(args.crop) if args.crop else None,
        downsample=args.downsample,
        dimensions=dims,
        glob_pattern=args.glob,
        label_opacity=args.opacity,
    )
