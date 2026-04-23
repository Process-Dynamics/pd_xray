import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter

from pd_xray.processing.image_processor import Image2DProcessor

FEATURE_NAMES = [
    "intensity",
    "gauss_s1", "gauss_s2", "gauss_s4", "gauss_s8",
    "median_3", "median_7",
    "grad_mag",
    "local_std_5", "local_std_15",
    "clahe",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_features(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Extract multi-scale texture and gradient features from a 2D image.

    Produces N_FEATURES channels per pixel using Gaussian blur at four scales,
    median filters, gradient magnitude, local standard deviation, and CLAHE.

    Args:
        image: 2D float32 array of shape (H, W).

    Returns:
        Float32 array of shape (H*W, N_FEATURES), one row per pixel.
    """
    img_min, img_max = float(image.min()), float(image.max())
    span = img_max - img_min
    norm = ((image - img_min) / (span + 1e-8)).astype(np.float32)

    features: list[NDArray[np.float32]] = [norm.ravel()]

    for sigma in (1.0, 2.0, 4.0, 8.0):
        features.append(gaussian_filter(norm, sigma=sigma).astype(np.float32).ravel())

    features.append(median_filter(norm, size=3).astype(np.float32).ravel())
    features.append(median_filter(norm, size=7).astype(np.float32).ravel())

    gy = np.gradient(norm.astype(np.float64), axis=0)
    gx = np.gradient(norm.astype(np.float64), axis=1)
    grad_mag = np.sqrt(gy ** 2 + gx ** 2).astype(np.float32)
    features.append(grad_mag.ravel())

    for size in (5, 15):
        mean = uniform_filter(norm.astype(np.float64), size=size)
        mean_sq = uniform_filter((norm.astype(np.float64)) ** 2, size=size)
        local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0)).astype(np.float32)
        features.append(local_std.ravel())

    proc = Image2DProcessor(steps=[{"name": "clahe", "clip_limit": 3.0}])
    clahe_img = proc(image)
    c_min, c_max = float(clahe_img.min()), float(clahe_img.max())
    clahe_norm = ((clahe_img - c_min) / (c_max - c_min + 1e-8)).astype(np.float32)
    features.append(clahe_norm.ravel())

    return np.column_stack(features)
