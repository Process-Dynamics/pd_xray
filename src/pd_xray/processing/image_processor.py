import time
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    gaussian_filter,
    median_filter,
    binary_erosion,
    binary_dilation,
    rotate,
    zoom,
)
from skimage import exposure, morphology
from skimage.restoration import denoise_bilateral

from pd_xray.core.logging import get_logger
from pd_xray.core.types import ProcessingResult


_logger = get_logger(__name__)


class Image2DProcessor:
    """Applies a configurable sequence of 2D filters and transforms to a single image.

    Configure the pipeline once, then apply it to many 2D slices. The processor is
    stateless with respect to image data, making it safe to share across processes
    and compatible with ProcessPoolExecutor.map() and dask.array.map_blocks().

    Example:
        proc = Image2DProcessor(steps=[
            {"name": "gaussian_blur", "sigma": 1.5},
            {"name": "normalise", "low": 0.0, "high": 1.0},
        ])

        # Single slice
        result_array = proc(slice_2d)

        # Parallel over many slices
        executor.map(proc, list_of_slices)
        dask.array.map_blocks(proc, zarr_volume, dtype="float32")
    """

    def __init__(self, steps: list[dict]) -> None:
        """
        Args:
            steps : Ordered list of operation descriptors. Each entry must have a
                    'name' key matching a filter method, plus any kwargs for that method.
        """
        self._steps = steps

    @property
    def steps(self) -> list[dict]:
        """Configured pipeline steps."""
        return self._steps

    # ------------------------------------------------------------------
    # Callable interface — primary entry point for map / map_blocks
    # ------------------------------------------------------------------

    def __call__(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the configured pipeline to a single 2D image.

        Args:
            image: 2D float32 array of shape (Y, X).

        Returns:
            Processed float32 array of the same shape.
        """
        result, _ = self.process(image)
        return result

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------

    def process(
        self, image: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], ProcessingResult]:
        """Apply the configured pipeline and return the result with metrics.

        Use this instead of __call__ when you need the ProcessingResult
        (timing, warnings, errors).

        Args:
            image: 2D float32 array of shape (Y, X).

        Returns:
            Tuple of (processed array, ProcessingResult).
        """
        if image.ndim != 2:
            return image, ProcessingResult(
                status="failed",
                errors=[f"Expected a 2D array, got shape {image.shape}"],
            )

        dispatch: dict[str, object] = {
            "gaussian_blur":    self._gaussian_blur,
            "median_filter":    self._median_filter,
            "bilateral_filter": self._bilateral_filter,
            "rolling_ball":     self._rolling_ball,
            "erode":            self._erode,
            "dilate":           self._dilate,
            "normalise":        self._normalise,
            "clip":             self._clip,
            "clahe":            self._clahe,
            "rotate":           self._rotate,
            "crop":             self._crop,
            "resize":           self._resize,
            "circular_mask":    self._circular_mask,
        }

        current = image.astype(np.float32, copy=False)
        warnings: list[str] = []
        t0 = time.perf_counter()

        for step in self._steps:
            step = step.copy()
            name = step.pop("name", None)

            if name is None:
                warnings.append("Step missing 'name' key, skipping.")
                continue
            if name not in dispatch:
                warnings.append(f"Unknown step '{name}', skipping.")
                continue

            try:
                current = dispatch[name](current, **step)  # type: ignore[operator]
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
    # Spatial filters
    # ------------------------------------------------------------------

    def _gaussian_blur(
        self,
        image: NDArray[np.float32],
        sigma: float,
    ) -> NDArray[np.float32]:
        """Apply Gaussian smoothing.

        Args:
            image : Input 2D array.
            sigma : Standard deviation of the Gaussian kernel in pixels.

        Returns:
            Blurred float32 array.
        """
        return gaussian_filter(image, sigma=sigma)

    def _median_filter(
        self,
        image: NDArray[np.float32],
        size: int,
    ) -> NDArray[np.float32]:
        """Apply median filtering for impulse-noise removal.

        Args:
            image : Input 2D array.
            size  : Side length of the square kernel (must be odd).

        Returns:
            Filtered float32 array.
        """
        return median_filter(image, size=size)

    def _bilateral_filter(
        self,
        image: NDArray[np.float32],
        sigma_spatial: float,
        sigma_color: float,
    ) -> NDArray[np.float32]:
        """Apply edge-preserving bilateral filter.

        Args:
            image         : Input 2D array.
            sigma_spatial : Standard deviation for range distance
            sigma_color   : Standard deviation for grayvalue/color distance 

        Returns:
            Filtered float32 array.
        """
        return denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    
    def _rolling_ball(
        self,
        image: NDArray[np.float32],
        radius: int = 50,
        smoothing_sigma: float = 2.0,
    ) -> NDArray[np.float32]:
        """Estimate and subtract spatially-varying background via rolling ball.

        Uses morphological opening as a proxy for the rolling ball algorithm.
        NaN pixels are filled with nanmedian before processing and restored after.

        Args:
            image          : Input 2D slice, may contain NaN.
            radius         : Rolling ball radius in pixels. Larger radius captures broader background variations.
            smoothing_sigma: Gaussian smoothing applied to background estimate to reduce noise. Set to 0 to disable.

        Returns:
            Filtered float32 array.
        """
        nan_mask = np.isnan(image)
        fill_value = float(np.nanmedian(image))

        image_filled = image.copy()
        image_filled[nan_mask] = fill_value

        footprint = morphology.disk(radius)
        background = morphology.opening(image_filled, footprint=footprint)

        if smoothing_sigma > 0.0:
            background = self._gaussian_blur(background, sigma=smoothing_sigma)

        corrected = (image_filled - background).astype(np.float32)
        corrected[nan_mask] = np.nan
        return corrected

    # ------------------------------------------------------------------
    # Morphological operations
    # ------------------------------------------------------------------

    def _erode(
        self,
        image: NDArray[np.float32],
        kernel_size: int,
        iterations: int = 1,
    ) -> NDArray[np.float32]:
        """Morphological erosion.

        Args:
            image       : Input 2D array.
            kernel_size : Side length of the structuring element.
            iterations  : Number of iterations to run the erosion again.

        Returns:
            Eroded float32 array.
        """
        return binary_erosion(
            image,
            structure=np.ones((kernel_size, kernel_size), dtype=bool),
            iterations=iterations,
        ).astype(np.float32)

    def _dilate(
        self,
        image: NDArray[np.float32],
        kernel_size: int,
        iterations: int = 1,
    ) -> NDArray[np.float32]:
        """Morphological dilation.

        Args:
            image       : Input 2D array.
            kernel_size : Side length of the structuring element.
            iterations  : Number of iterations to run the erosion again.

        Returns:
            Dilated float32 array.
        """
        return binary_dilation(
            image,
            structure=np.ones((kernel_size, kernel_size), dtype=bool),
            iterations=iterations,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Intensity adjustments
    # ------------------------------------------------------------------

    def _normalise(
        self,
        image: NDArray[np.float32],
        low: float | None = None,
        high: float | None = None,
    ) -> NDArray[np.float32]:
        """Rescale intensities to [low, high].

        Args:
            image : Input 2D array.
            low   : Target minimum value. If None, min from the image array is taken.
            high  : Target maximum value. If None, max from the image array is taken.

        Returns:
            Normalised float32 array.
        """
        src_min = low if low is not None else float(image.min())
        src_max = high if high is not None else float(image.max())

        span = src_max - src_min
        if span == 0.0:
            return np.zeros_like(image)

        return ((image - src_min) / span).astype(np.float32)

    def _clip(
        self,
        image: NDArray[np.float32],
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> NDArray[np.float32]:
        """Clip pixel values to [vmin, vmax].

        Args:
            image : Input 2D array.
            vmin  :  Lower bound.
            vmax  :  Upper bound.

        Returns:
            Clipped float32 array.
        """
        vmin = vmin if vmin is not None else image.min()
        vmax = vmax if vmax is not None else image.max()
        return np.clip(
            image,
            a_min=vmin,
            a_max=vmax
        )
    
    def _clahe(
        self,
        image: NDArray[np.float32],
        clip_limit: float = 3.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> NDArray[np.float32]:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Scales the image to uint16 internally for skimage compatibility.
        NaN pixels are zeroed before processing and restored after.

        Args:
            image          : Input 2D slice. NaN pixels will be masked out.
            clip_limit     : Clipping limit as a percentage (0-100). Higher values give stronger contrast enhancement
                             at the cost of noise amplification.
            tile_grid_size : Number of tiles (rows, cols) for local histogram computation.

        Returns:
            Histogram equalized 2D array.
        """
        nan_mask = np.isnan(image)

        valid_min = float(np.nanmin(image))
        valid_max = float(np.nanmax(image))

        # Scale to uint16 range for skimage
        img_scaled = ((image - valid_min) / (valid_max - valid_min + 1e-8) * 65535).astype(
            np.uint16
        )
        img_scaled[nan_mask] = 0

        result = exposure.equalize_adapthist(
            img_scaled,
            clip_limit=clip_limit / 100.0,
            nbins=256,
            kernel_size=tile_grid_size,
        ).astype(np.float32)

        result[nan_mask] = np.nan

        return result

    # ------------------------------------------------------------------
    # Geometric transforms
    # ------------------------------------------------------------------

    def _rotate(
        self,
        image: NDArray[np.float32],
        angle_deg: float,
    ) -> NDArray[np.float32]:
        """Rotate the image about its centre.

        Args:
            image     :     Input 2D array.
            angle_deg : Counter-clockwise rotation angle in degrees.

        Returns:
            Rotated float32 array.
        """
        return rotate(
            image,
            angle=angle_deg,
            reshape=False,
        ).astype(np.float32)

    def _crop(
        self,
        image: NDArray[np.float32],
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> NDArray[np.float32]:
        """Crop a rectangular region of interest.

        Args:
            image     : Input 2D array.
            row_start : First row index (inclusive).
            row_end   : Last row index (exclusive).
            col_start : First column index (inclusive).
            col_end   : Last column index (exclusive).

        Returns:
            Cropped float32 array.
        """
        return image[row_start:row_end, col_start:col_end]

    def _resize(
        self,
        image: NDArray[np.float32],
        height: int,
        width: int,
    ) -> NDArray[np.float32]:
        """Resize the image to the given dimensions.

        Args:
            image  : Input 2D array.
            height : Target number of rows.
            width  : Target number of columns.

        Returns:
            Resized float32 array.
        """
        zoom_y = height / image.shape[0]
        zoom_x = width / image.shape[1]
        return zoom(image, (zoom_y, zoom_x), order=1).astype(np.float32)

    def _circular_mask(
        self,
        image: NDArray[np.float32],
        mask_ratio: float = 0.5,
    ) -> NDArray[np.float32]:
        """Mask pixels outside a centered circular field of view to NaN.
        
        Used to remove artefacts at the sinogram edges that fall outside the detector FOV.
        
        Args:
            image      : Input 2D array.
            mask_ratio : Fraction of min(height, width) to use as circle radius. Values in (0, 1)
        
        Returns:
            Masked float32 array.
        """
        height, width = image.shape
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) / 2.0 * mask_ratio

        y_coords, x_coords = np.ogrid[:height, :width]
        inside_fov = (x_coords - center_x) ** 2 + \
            (y_coords - center_y) ** 2 <= radius**2

        result = image.copy()
        result[~inside_fov] = np.nan

        return result
