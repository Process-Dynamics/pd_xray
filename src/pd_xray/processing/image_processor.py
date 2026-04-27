import time
from typing import ClassVar

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    median_filter,
    rotate,
    zoom,
)
from skimage import exposure, morphology
from skimage.restoration import denoise_bilateral

from pd_xray.core import ProcessingResult, T, get_logger

_logger = get_logger(__name__)


class ImageProcessor:
    """Applies a configurable sequence of filters and transforms to a 2D or 3D image array.

    Configure the pipeline once, then apply it to images or volumes. The processor is
    stateless with respect to image data — safe to share across processes and compatible
    with ProcessPoolExecutor.map() and dask.array.map_blocks().

    All operations accept any numeric NumPy dtype. Methods that require float arithmetic
    cast internally and return float32 by default. Pass ``dtype`` to any fluent method to
    override the output dtype of that specific step.

    3D arrays are assumed to have shape (Z, Y, X). Operations that are inherently 2D
    (bilateral filter, rolling ball, CLAHE, rotate) are applied slice-by-slice along Z.

    Two equivalent ways to build a pipeline:

    Fluent builder (recommended — tab-completion and type-checker friendly):
        proc = (
            ImageProcessor()
            .gaussian_blur(sigma=1.5)
            .normalise()
            .clip(vmin=0.0, vmax=1.0, dtype=np.float32)
        )

    Dict-based (for YAML / JSON config files):
        proc = ImageProcessor(steps=[
            {"name": "gaussian_blur", "sigma": 1.5},
            {"name": "normalise"},
            {"name": "clip", "vmin": 0.0, "vmax": 1.0, "output_dtype": "float32"},
        ])

    Running the pipeline:
        result_array        = proc(image)               # __call__, returns ndarray
        result, info        = proc.process(image)       # also returns ProcessingResult
        executor.map(proc, slices)                      # parallel over many slices
        dask.array.map_blocks(proc, zarr_vol)           # dask
    """

    _DISPATCH: ClassVar[dict[str, str]] = {
        "gaussian_blur"           : "_gaussian_blur",
        "median_filter"           : "_median_filter",
        "bilateral_filter"        : "_bilateral_filter",
        "rolling_ball"            : "_rolling_ball",
        "erode"                   : "_erode",
        "dilate"                  : "_dilate",
        "open"                    : "_open",
        "close"                   : "_close",
        "fill_holes"              : "_fill_holes",
        "remove_small_objects"    : "_remove_small_objects",
        "remove_small_holes"      : "_remove_small_holes",
        "normalise"               : "_normalise",
        "clip"                    : "_clip",
        "clahe"                   : "_clahe",
        "rotate"                  : "_rotate",
        "crop"                    : "_crop",
        "resize"                  : "_resize",
        "circular_mask"           : "_circular_mask",
        "cylindrical_mask"        : "_extract_cylinder",
        "extract_segmented_class" : "_extract_class_from_segmentation",
    }

    def __init__(self, steps: list[dict] | None = None) -> None:
        """
        Args:
            steps: Ordered list of operation descriptors. Each entry must have a ``name``
                   key matching a step name, plus any kwargs for that operation.
                   An optional ``output_dtype`` key in any step casts the result to that
                   dtype after the operation runs.
                   Omit (or pass None) to start with an empty pipeline and build it with
                   the fluent methods.
        """
        self._steps: list[dict] = list(steps) if steps is not None else []

    @property
    def steps(self) -> list[dict]:
        """Configured pipeline steps (read-only snapshot)."""
        return list(self._steps)

    @classmethod
    def available_steps(cls) -> list[str]:
        """Return the names of all available processing steps.

        Example:
            >>> ImageProcessor.available_steps()
            ['gaussian_blur', 'median_filter', ...]
        """
        return list(cls._DISPATCH.keys())

    def __repr__(self) -> str:
        if not self._steps:
            return "ImageProcessor(steps=[])"
        step_lines = "\n".join(
            f"  {i + 1}. {s.get('name', '?')}"
            f"({', '.join(f'{k}={v!r}' for k, v in s.items() if k != 'name')})"
            for i, s in enumerate(self._steps)
        )
        return f"ImageProcessor(\n{step_lines}\n)"

    # ------------------------------------------------------------------
    # Callable interface — primary entry point for map / map_blocks
    # ------------------------------------------------------------------

    def __call__(self, image: NDArray) -> NDArray:
        """Apply the configured pipeline to an image or volume.

        Args:
            image: 2D or 3D array of any numeric dtype. 3D arrays must be (Z, Y, X).

        Returns:
            Processed array.
        """
        result, _ = self.process(image)
        return result

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------

    def process(self, image: NDArray) -> tuple[NDArray, ProcessingResult]:
        """Apply the configured pipeline and return the result with metrics.

        Use this instead of __call__ when you need the ProcessingResult
        (timing, warnings, errors).

        Args:
            image: 2D or 3D array of any numeric dtype. 3D arrays must be (Z, Y, X).

        Returns:
            Tuple of (processed array, ProcessingResult).
        """
        current: NDArray = np.asarray(image)
        warnings: list[str] = []
        t0 = time.perf_counter()

        for step in self._steps:
            step = step.copy()
            name = step.pop("name", None)
            output_dtype: DTypeLike | None = step.pop("output_dtype", None)

            if name is None:
                warnings.append("Step missing 'name' key, skipping.")
                continue
            if name not in self._DISPATCH:
                warnings.append(f"Unknown step '{name}', skipping.")
                continue

            try:
                fn = getattr(self, self._DISPATCH[name])
                current = fn(current, **step)
                if output_dtype is not None:
                    current = current.astype(output_dtype, copy=False)
            except Exception as exc:
                return current, ProcessingResult(
                    status="failed",
                    duration_seconds=time.perf_counter() - t0,
                    errors=[f"Step '{name}' failed: {exc}"],
                    warnings=warnings,
                )

        return current, ProcessingResult(
            status="success",
            duration_seconds=time.perf_counter() - t0,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Fluent builder — internal helper
    # ------------------------------------------------------------------

    def _add_step(self, name: str, dtype: DTypeLike | None, **kwargs) -> "ImageProcessor":
        step: dict = {"name": name, **kwargs}
        if dtype is not None:
            step["output_dtype"] = np.dtype(dtype)
        self._steps.append(step)
        return self

    # ------------------------------------------------------------------
    # Fluent builder — public methods
    # ------------------------------------------------------------------

    def gaussian_blur(self, sigma: float, dtype: DTypeLike | None = None) -> "ImageProcessor":
        """Gaussian smoothing. ``sigma``: kernel standard deviation in pixels."""
        return self._add_step("gaussian_blur", dtype, sigma=sigma)

    def median_filter(self, size: int, dtype: DTypeLike | None = None) -> "ImageProcessor":
        """Median filter for impulse-noise removal. ``size``: square kernel side length."""
        return self._add_step("median_filter", dtype, size=size)

    def bilateral_filter(
        self, sigma_spatial: float, sigma_color: float, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Edge-preserving bilateral filter. Applied per Z-slice on 3D arrays."""
        return self._add_step(
            "bilateral_filter", dtype, sigma_spatial=sigma_spatial, sigma_color=sigma_color
        )

    def rolling_ball(
        self, radius: int = 50, smoothing_sigma: float = 2.0, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Rolling ball background subtraction. Applied per Z-slice on 3D arrays."""
        return self._add_step(
            "rolling_ball", dtype, radius=radius, smoothing_sigma=smoothing_sigma
        )

    def erode(
        self, kernel_size: int, iterations: int = 1, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Morphological erosion."""
        return self._add_step("erode", dtype, kernel_size=kernel_size, iterations=iterations)

    def dilate(
        self, kernel_size: int, iterations: int = 1, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Morphological dilation."""
        return self._add_step("dilate", dtype, kernel_size=kernel_size, iterations=iterations)

    def open(
        self, kernel_size: int, iterations: int = 1, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Morphological opening (erode then dilate)."""
        return self._add_step("open", dtype, kernel_size=kernel_size, iterations=iterations)

    def close(
        self, kernel_size: int, iterations: int = 1, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Morphological closing (dilate then erode)."""
        return self._add_step("close", dtype, kernel_size=kernel_size, iterations=iterations)

    def fill_holes(self, dtype: DTypeLike | None = None) -> "ImageProcessor":
        """Fill enclosed background holes in a binary array."""
        return self._add_step("fill_holes", dtype)

    def remove_small_objects(
        self, min_size: int = 64, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Remove foreground connected components smaller than ``min_size`` pixels."""
        return self._add_step("remove_small_objects", dtype, min_size=min_size)

    def remove_small_holes(
        self, area_threshold: int = 64, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Fill enclosed holes smaller than ``area_threshold`` pixels."""
        return self._add_step("remove_small_holes", dtype, area_threshold=area_threshold)

    def normalise(
        self,
        low: float | None = None,
        high: float | None = None,
        dtype: DTypeLike | None = None,
    ) -> "ImageProcessor":
        """Rescale intensities to [low, high] (defaults to [array-min, array-max])."""
        return self._add_step("normalise", dtype, low=low, high=high)

    def clip(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        dtype: DTypeLike | None = None,
    ) -> "ImageProcessor":
        """Clip pixel values to [vmin, vmax]."""
        return self._add_step("clip", dtype, vmin=vmin, vmax=vmax)

    def clahe(
        self,
        clip_limit: float = 3.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        dtype: DTypeLike | None = None,
    ) -> "ImageProcessor":
        """Contrast Limited Adaptive Histogram Equalization. Applied per Z-slice on 3D arrays."""
        return self._add_step(
            "clahe", dtype, clip_limit=clip_limit, tile_grid_size=tile_grid_size
        )

    def rotate(self, angle_deg: float, dtype: DTypeLike | None = None) -> "ImageProcessor":
        """Rotate about the image centre (CCW). Applied per Z-slice on 3D arrays."""
        return self._add_step("rotate", dtype, angle_deg=angle_deg)

    def crop(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        slice_start: int | None = None,
        slice_end: int | None = None,
        dtype: DTypeLike | None = None,
    ) -> "ImageProcessor":
        """Crop a rectangular region. ``slice_start``/``slice_end`` apply to Z on 3D arrays."""
        return self._add_step(
            "crop", dtype,
            row_start=row_start, row_end=row_end,
            col_start=col_start, col_end=col_end,
            slice_start=slice_start, slice_end=slice_end,
        )

    def resize(
        self,
        height: int,
        width: int,
        depth: int | None = None,
        dtype: DTypeLike | None = None,
    ) -> "ImageProcessor":
        """Resize to (height, width). ``depth`` optionally resizes the Z axis on 3D arrays."""
        return self._add_step("resize", dtype, height=height, width=width, depth=depth)

    def circular_mask(
        self,
        mask_ratio: float = 0.5,
        crop_to_circle: bool = False,
        dtype: DTypeLike | None = None,
    ) -> "ImageProcessor":
        """Mask pixels outside a centred circular FOV. Same mask applied to every Z-slice."""
        return self._add_step(
            "circular_mask", dtype, mask_ratio=mask_ratio, crop_to_circle=crop_to_circle
        )

    def cylindrical_mask(
        self, mask_ratio: float = 0.5, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Apply a cylindrical/circular mask and crop to the bounding box."""
        return self._add_step("cylindrical_mask", dtype, mask_ratio=mask_ratio)

    def extract_segmented_class(
        self, class_value: int = 0, dtype: DTypeLike | None = None
    ) -> "ImageProcessor":
        """Extract one class from an integer-labelled segmentation as a boolean mask."""
        return self._add_step("extract_segmented_class", dtype, class_value=class_value)

    # ------------------------------------------------------------------
    # Spatial filters
    # ------------------------------------------------------------------

    def _gaussian_blur(self, image: NDArray, sigma: float) -> NDArray:
        """Apply Gaussian smoothing.

        Args:
            image : Input 2D or 3D array.
            sigma : Standard deviation of the Gaussian kernel in pixels.

        Returns:
            Blurred float32 array (same number of dimensions as input).
        """
        return gaussian_filter(image.astype(np.float32, copy=False), sigma=sigma)

    def _median_filter(self, image: NDArray, size: int) -> NDArray:
        """Apply median filtering for impulse-noise removal.

        Args:
            image : Input 2D or 3D array.
            size  : Side length of the square kernel (must be odd).

        Returns:
            Filtered float32 array.
        """
        return median_filter(image.astype(np.float32, copy=False), size=size)

    def _bilateral_filter(
        self, image: NDArray, sigma_spatial: float, sigma_color: float
    ) -> NDArray:
        """Apply edge-preserving bilateral filter.

        Applied per Z-slice for 3D arrays.

        Args:
            image         : Input 2D or 3D array.
            sigma_spatial : Standard deviation for spatial distance.
            sigma_color   : Standard deviation for intensity distance.

        Returns:
            Filtered float32 array.
        """
        img = image.astype(np.float32, copy=False)
        if img.ndim == 2:
            return denoise_bilateral(
                img, sigma_color=sigma_color, sigma_spatial=sigma_spatial
            ).astype(np.float32)
        return np.stack([
            denoise_bilateral(s, sigma_color=sigma_color, sigma_spatial=sigma_spatial).astype(np.float32)
            for s in img
        ])

    def _rolling_ball(
        self, image: NDArray, radius: int = 50, smoothing_sigma: float = 2.0
    ) -> NDArray:
        """Estimate and subtract spatially-varying background via rolling ball.

        Uses morphological opening as a proxy for the rolling ball algorithm.
        NaN pixels are filled with nanmedian before processing and restored after.
        Applied per Z-slice for 3D arrays.

        Args:
            image           : Input 2D or 3D array. May contain NaN.
            radius          : Rolling ball radius in pixels.
            smoothing_sigma : Gaussian sigma applied to the background estimate (0 to disable).

        Returns:
            Background-subtracted float32 array.
        """
        if image.ndim == 3:
            return np.stack([
                self._rolling_ball(s, radius=radius, smoothing_sigma=smoothing_sigma)
                for s in image
            ])

        img = image.astype(np.float32, copy=False)
        nan_mask = np.isnan(img)
        fill_value = float(np.nanmedian(img))

        img_filled = img.copy()
        img_filled[nan_mask] = fill_value

        background = morphology.opening(img_filled, footprint=morphology.disk(radius))
        if smoothing_sigma > 0.0:
            background = gaussian_filter(background, sigma=smoothing_sigma)

        corrected = (img_filled - background).astype(np.float32)
        corrected[nan_mask] = np.nan
        return corrected

    # ------------------------------------------------------------------
    # Morphological operations
    # ------------------------------------------------------------------

    def _erode(self, image: NDArray, kernel_size: int, iterations: int = 1) -> NDArray:
        """Morphological erosion.

        Args:
            image       : Input 2D or 3D binary array.
            kernel_size : Side length of the structuring element.
            iterations  : Number of erosion iterations.

        Returns:
            Eroded float32 array.
        """
        struct = np.ones((kernel_size,) * image.ndim, dtype=bool)
        return binary_erosion(image, structure=struct, iterations=iterations).astype(np.float32)

    def _dilate(self, image: NDArray, kernel_size: int, iterations: int = 1) -> NDArray:
        """Morphological dilation.

        Args:
            image       : Input 2D or 3D binary array.
            kernel_size : Side length of the structuring element.
            iterations  : Number of dilation iterations.

        Returns:
            Dilated float32 array.
        """
        struct = np.ones((kernel_size,) * image.ndim, dtype=bool)
        return binary_dilation(image, structure=struct, iterations=iterations).astype(np.float32)

    def _open(self, image: NDArray, kernel_size: int, iterations: int = 1) -> NDArray:
        """Morphological opening (erode then dilate).

        Removes small bright blobs and thin protrusions without shrinking larger structures.

        Args:
            image       : Input 2D or 3D binary array.
            kernel_size : Side length of the structuring element.
            iterations  : Number of open operations.

        Returns:
            Opened float32 array.
        """
        struct = np.ones((kernel_size,) * image.ndim, dtype=bool)
        return binary_opening(image, structure=struct, iterations=iterations).astype(np.float32)

    def _close(self, image: NDArray, kernel_size: int, iterations: int = 1) -> NDArray:
        """Morphological closing (dilate then erode).

        Fills small dark holes and bridges narrow gaps without growing larger structures.

        Args:
            image       : Input 2D or 3D binary array.
            kernel_size : Side length of the structuring element.
            iterations  : Number of close operations.

        Returns:
            Closed float32 array.
        """
        struct = np.ones((kernel_size,) * image.ndim, dtype=bool)
        return binary_closing(image, structure=struct, iterations=iterations).astype(np.float32)

    def _fill_holes(self, image: NDArray) -> NDArray:
        """Fill enclosed background holes in a binary array.

        Args:
            image : Input 2D or 3D binary array.

        Returns:
            Hole-filled float32 array.
        """
        return binary_fill_holes(image).astype(np.float32)

    def _remove_small_objects(self, image: NDArray, min_size: int = 64) -> NDArray:
        """Remove foreground connected components smaller than min_size pixels.

        Args:
            image    : Input 2D or 3D binary array.
            min_size : Minimum component size to keep.

        Returns:
            Cleaned float32 array.
        """
        return morphology.remove_small_objects(
            image.astype(bool), min_size=min_size
        ).astype(np.float32)

    def _remove_small_holes(self, image: NDArray, area_threshold: int = 64) -> NDArray:
        """Fill enclosed background holes smaller than area_threshold pixels.

        Args:
            image          : Input 2D or 3D binary array.
            area_threshold : Maximum hole area to fill.

        Returns:
            Float32 array with small holes filled.
        """
        return morphology.remove_small_holes(
            image.astype(bool), area_threshold=area_threshold
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Intensity adjustments
    # ------------------------------------------------------------------

    def _normalise(
        self, image: NDArray, low: float | None = None, high: float | None = None
    ) -> NDArray:
        """Rescale intensities to [low, high].

        Args:
            image : Input 2D or 3D array.
            low   : Target minimum. Defaults to the array minimum.
            high  : Target maximum. Defaults to the array maximum.

        Returns:
            Normalised float32 array.
        """
        img = image.astype(np.float32, copy=False)
        src_min = low if low is not None else float(np.nanmin(img))
        src_max = high if high is not None else float(np.nanmax(img))

        span = src_max - src_min
        if span == 0.0:
            return np.zeros_like(img)
        return ((img - src_min) / span).astype(np.float32)

    def _clip(
        self, image: NDArray, vmin: float | None = None, vmax: float | None = None
    ) -> NDArray:
        """Clip pixel values to [vmin, vmax].

        Args:
            image : Input 2D or 3D array.
            vmin  : Lower bound. Defaults to the array minimum.
            vmax  : Upper bound. Defaults to the array maximum.

        Returns:
            Clipped array (same dtype as input).
        """
        a_min = vmin if vmin is not None else float(image.min())
        a_max = vmax if vmax is not None else float(image.max())
        return np.clip(image, a_min=a_min, a_max=a_max)

    def _clahe(
        self,
        image: NDArray,
        clip_limit: float = 3.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> NDArray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Scales to uint16 internally for skimage compatibility. NaN pixels are zeroed
        before processing and restored afterwards.
        Applied per Z-slice for 3D arrays.

        Args:
            image          : Input 2D or 3D array.
            clip_limit     : Clipping limit as a percentage (0–100).
            tile_grid_size : (rows, cols) tile grid for local histogram computation.

        Returns:
            CLAHE-equalised float32 array.
        """
        if image.ndim == 3:
            return np.stack([
                self._clahe(s, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
                for s in image
            ])

        img = image.astype(np.float32, copy=False)
        nan_mask = np.isnan(img)

        valid_min = float(np.nanmin(img))
        valid_max = float(np.nanmax(img))

        img_u16 = ((img - valid_min) / (valid_max - valid_min + 1e-8) * 65535).astype(np.uint16)
        img_u16[nan_mask] = 0

        result = exposure.equalize_adapthist(
            img_u16,
            clip_limit=clip_limit / 100.0,
            nbins=256,
            kernel_size=tile_grid_size,
        ).astype(np.float32)

        result[nan_mask] = np.nan
        return result

    # ------------------------------------------------------------------
    # Geometric transforms
    # ------------------------------------------------------------------

    def _rotate(self, image: NDArray, angle_deg: float) -> NDArray:
        """Rotate about the image centre (counter-clockwise).

        Applied per Z-slice for 3D arrays.

        Args:
            image     : Input 2D or 3D array.
            angle_deg : Rotation angle in degrees.

        Returns:
            Rotated float32 array.
        """
        img = image.astype(np.float32, copy=False)
        if img.ndim == 3:
            return np.stack([rotate(s, angle=angle_deg, reshape=False) for s in img])
        return rotate(img, angle=angle_deg, reshape=False)

    def _crop(
        self,
        image: NDArray,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        slice_start: int | None = None,
        slice_end: int | None = None,
    ) -> NDArray:
        """Crop a rectangular region of interest.

        Args:
            image       : Input 2D or 3D array.
            row_start   : First row index (inclusive).
            row_end     : Last row index (exclusive).
            col_start   : First column index (inclusive).
            col_end     : Last column index (exclusive).
            slice_start : First Z index (inclusive, 3D only).
            slice_end   : Last Z index (exclusive, 3D only).

        Returns:
            Cropped array (same dtype as input).
        """
        if image.ndim == 2:
            return image[row_start:row_end, col_start:col_end]
        return image[slice_start:slice_end, row_start:row_end, col_start:col_end]

    def _resize(
        self, image: NDArray, height: int, width: int, depth: int | None = None
    ) -> NDArray:
        """Resize to the target dimensions using bilinear interpolation.

        Args:
            image  : Input 2D or 3D array.
            height : Target number of rows.
            width  : Target number of columns.
            depth  : Target number of Z-slices (3D only; None keeps the current depth).

        Returns:
            Resized float32 array.
        """
        img = image.astype(np.float32, copy=False)
        if img.ndim == 2:
            return zoom(img, (height / img.shape[0], width / img.shape[1]), order=1)
        zz = depth / img.shape[0] if depth is not None else 1.0
        return zoom(img, (zz, height / img.shape[1], width / img.shape[2]), order=1)

    def _circular_mask(
        self,
        image: NDArray,
        mask_ratio: float = 0.5,
        crop_to_circle: bool = False,
    ) -> NDArray:
        """Mask pixels outside a centred circular field of view.

        Outside pixels are set to NaN for floating-point arrays and 0 for integer arrays.
        For 3D arrays the same 2D mask is broadcast across all Z-slices.

        Args:
            image          : Input 2D or 3D array.
            mask_ratio     : Fraction of min(height, width) used as circle radius. In (0, 1).
            crop_to_circle : If True, crop to the bounding square of the circle.

        Returns:
            Masked array (same dtype as input).
        """
        h, w = image.shape[-2], image.shape[-1]
        cy, cx = h // 2, w // 2
        radius = min(h, w) / 2.0 * mask_ratio

        y_idx, x_idx = np.ogrid[:h, :w]
        inside = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= radius ** 2

        result = image.copy()
        fill = np.nan if np.issubdtype(image.dtype, np.floating) else 0

        if image.ndim == 2:
            result[~inside] = fill
        else:
            result[:, ~inside] = fill

        if crop_to_circle:
            r = int(radius)
            rs, re = max(cy - r, 0), min(cy + r, h)
            cs, ce = max(cx - r, 0), min(cx + r, w)
            result = result[..., rs:re, cs:ce]

        return result

    def _extract_cylinder(self, array: NDArray[T], mask_ratio: float = 0.5) -> NDArray[T]:
        """Apply a cylindrical mask and crop to the bounding box.

        Args:
            array      : 2D or 3D array.
            mask_ratio : Fraction of min(height, width) used as circle radius.

        Returns:
            Cropped array (same dtype as input).
        """
        if array.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got {array.ndim}D.")

        h, w = array.shape[-2:]
        cy, cx = h // 2, w // 2
        radius = mask_ratio * (min(h, w) / 2)

        y, x = np.ogrid[:h, :w]
        mask_2d = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2

        if array.ndim == 2:
            masked = np.where(mask_2d, array, 0)
        else:
            masked = np.where(mask_2d[np.newaxis], array, 0)

        ys, xs = np.where(mask_2d)
        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1

        if array.ndim == 2:
            return masked[y_min:y_max, x_min:x_max]
        return masked[:, y_min:y_max, x_min:x_max]

    def _extract_class_from_segmentation(
        self, array: NDArray[T], class_value: int = 0
    ) -> NDArray[T]:
        """Extract one class from an integer-labelled segmentation as a boolean mask.

        Args:
            array       : 2D or 3D integer-labelled array.
            class_value : Class label to extract.

        Returns:
            Boolean mask of the same spatial shape.
        """
        if array.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got {array.ndim}D.")
        return array == class_value


# ------------------------------------------------------------------
# Segmentation post-processing utility
# ------------------------------------------------------------------

def postprocess_segmentation_mask(
    mask: NDArray[np.int32],
    steps: list[dict],
    class_ids: list[int] | None = None,
    background_class: int = 0,
) -> NDArray[np.int32]:
    """Apply morphological post-processing per class to a segmentation mask.

    Each class is extracted as a binary mask, processed independently through
    the given steps, then recomposed into a single label map. Only binary
    operations make sense here: erode, dilate, open, close, fill_holes,
    remove_small_objects, remove_small_holes.

    Conflict resolution: background_class is written first so all foreground classes
    overwrite it. If two foreground classes overlap after processing, the later one
    in class_ids wins.

    Example::
        cleaned = postprocess_segmentation_mask(
            pred_mask,
            steps=[
                {"name": "fill_holes"},
                {"name": "remove_small_objects", "min_size": 200},
                {"name": "close", "kernel_size": 3},
            ],
            background_class=0,
        )

    Args:
        mask:             2D or 3D int32 label map.
        steps:            ImageProcessor step dicts (binary operations only).
        class_ids:        Classes to process. Defaults to all unique classes in mask.
        background_class: Written first (lowest priority). Default 0.

    Returns:
        Post-processed int32 label map of the same shape as mask.
    """
    if class_ids is None:
        class_ids = [int(c) for c in np.unique(mask)]

    ordered = [background_class] + [c for c in class_ids if c != background_class]

    processor = ImageProcessor(steps=steps)
    output = np.full_like(mask, fill_value=background_class)

    for class_id in ordered:
        if class_id not in class_ids:
            continue
        binary = (mask == class_id).astype(np.float32)
        processed = processor(binary)
        output[processed > 0.5] = class_id

    return output
