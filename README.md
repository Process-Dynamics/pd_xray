# pd_xray
Process Dynamics Group Experiment Processing Library

---

## Configuration

`Config` loads a YAML file and lets you read values using dot-notation keys. It is the standard way to pass processing parameters through the pipeline without hardcoding them in scripts.

### Loading a config file

```python
from pd_xray.core.config import Config

config = Config("config.yaml")
```

A `FileNotFoundError` is raised if the file is missing, and a `ConfigError` if the YAML is invalid.

### Reading values

Use `.get()` for optional keys and `.require()` for keys that must be present:

```python
energy   = config.get("reconstruction.paganin.energy", default=25.0)
raw_path = config.require("data.raw_path")
gpu_ids  = config.get("hardware.gpu_ids", default=[0])
```

`.require()` raises `ConfigKeyError` immediately if the key is absent, so errors surface early rather than later in the pipeline.

If you want to enforce a type at the same time:

```python
energy = config.get_typed("reconstruction.paganin.energy", float)
```

### Checking for optional sections

```python
if config.has("segmentation.model"):
    model_path = config.require("segmentation.model")
```

### Building an ImageProcessor pipeline from config

Store your processing steps in YAML and load them directly into `ImageProcessor`:

```yaml
# config.yaml
processing:
  steps:
    - name: gaussian_blur
      sigma: 1.5
    - name: normalise
    - name: clip
      vmin: 0.0
      vmax: 1.0
```

```python
from pd_xray.processing import ImageProcessor

steps = config.require("processing.steps")
proc  = ImageProcessor(steps=steps)
result = proc(image)
```

### Overriding values for an experiment

`.set()` updates a value in memory without touching the file. Useful for running the same config with one parameter changed:

```python
config.set("reconstruction.paganin.energy", 30.0)
```

To apply a whole set of overrides at once, pass a dict to `.merge()`:

```python
config.merge({"reconstruction": {"paganin": {"energy": 30.0, "distance": 0.8}}})
```

### Passing a subsection to a function

If a function only needs one part of the config, hand it a scoped view rather than the whole thing:

```python
recon_cfg = config.get_section("reconstruction")
energy    = recon_cfg.get("paganin.energy")
```

---

## Image Processing

`ImageProcessor` applies a configurable sequence of filters and transforms to 2D `(H, W)` or 3D `(Z, H, W)` NumPy arrays of any dtype. Build the pipeline once and reuse it across many images or volumes.

### Building a pipeline

**Fluent builder** — recommended for interactive use and scripts, works with tab-completion:

```python
from pd_xray.processing import ImageProcessor
import numpy as np

proc = (
    ImageProcessor()
    .gaussian_blur(sigma=1.5)
    .normalise()
    .clip(vmin=0.0, vmax=1.0)
)
```

**Dict-based** — useful when loading a pipeline from a YAML or JSON config file:

```python
proc = ImageProcessor(steps=[
    {"name": "gaussian_blur", "sigma": 1.5},
    {"name": "normalise"},
    {"name": "clip", "vmin": 0.0, "vmax": 1.0},
])
```

Both styles can be mixed — start from a config and add extra steps with fluent calls.

### Running the pipeline

```python
# Returns the processed array directly
result = proc(image)

# Returns (processed array, ProcessingResult) with timing and any warnings/errors
result, info = proc.process(image)
print(info.status, info.duration_seconds)
```

The same pipeline works on 2D slices and full 3D volumes without any changes. Operations that are inherently 2D (bilateral filter, rolling ball, CLAHE, rotate) are automatically applied slice-by-slice along the Z axis.

```python
slice_2d  = np.random.rand(512, 512).astype(np.float32)
volume_3d = np.random.rand(200, 512, 512).astype(np.float32)

proc(slice_2d)   # shape (512, 512)
proc(volume_3d)  # shape (200, 512, 512)
```

### Controlling output dtype

Every fluent method accepts an optional `dtype` argument that casts the result of that specific step. If omitted, the operation's natural output type is preserved.

```python
# Read a uint16 scan, blur it, and keep the result as float32
proc = (
    ImageProcessor()
    .gaussian_blur(sigma=2.0, dtype=np.float32)
    .normalise()
)

# clip preserves the input dtype when dtype is not specified
proc = ImageProcessor().clip(vmin=0, vmax=200)   # uint8 in → uint8 out
```

In dict-based pipelines use the `output_dtype` key:

```python
{"name": "gaussian_blur", "sigma": 2.0, "output_dtype": "float32"}
```

### Discovering available steps

```python
ImageProcessor.available_steps()
# ['gaussian_blur', 'median_filter', 'bilateral_filter', 'rolling_ball',
#  'erode', 'dilate', 'open', 'close', 'fill_holes', 'remove_small_objects',
#  'remove_small_holes', 'normalise', 'clip', 'clahe', 'rotate', 'crop',
#  'resize', 'circular_mask', 'cylindrical_mask', 'extract_segmented_class']
```

### Parallel and distributed use

`ImageProcessor` is stateless with respect to image data — the same instance can be safely shared across threads and processes.

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as ex:
    results = list(ex.map(proc, list_of_slices))
```

```python
import dask.array as da

dask_volume = da.from_zarr("scan.zarr")
processed   = dask_volume.map_blocks(proc, dtype=np.float32)
```

### Flat-field and dark-field correction

`flat_dark_field_correction` applies the standard radiography correction formula
`(image - dark) / (flat - dark)` to a 2D frame or a 3D stack:

```python
from pd_xray.processing import flat_dark_field_correction

# Standalone function
corrected = flat_dark_field_correction(projections, flat=flat_stack, dark=dark_stack)
```

```
image : (H, W) or (N, H, W) — single frame or N-frame stack
flat  : (H, W) or (K, H, W) — if 3D, averaged along axis 0 before use
dark  : (H, W) or (K, H, W) — if 3D, averaged along axis 0 before use
```

Pixels where `flat - dark == 0` (dead pixels) are set to NaN. A `ValueError` is
raised if the spatial dimensions `(H, W)` do not match across all three inputs.

It is also available as a pipeline step:

```python
proc = (
    ImageProcessor()
    .flat_dark_field_correction(flat=flat_mean, dark=dark_mean)
    .gaussian_blur(sigma=1.0)
    .normalise()
)
```

### Segmentation post-processing

`postprocess_segmentation_mask` applies morphological cleanup per-class to an integer label map:

```python
from pd_xray.processing import postprocess_segmentation_mask

cleaned = postprocess_segmentation_mask(
    pred_mask,          # 2D or 3D int32 label map
    steps=[
        {"name": "fill_holes"},
        {"name": "remove_small_objects", "min_size": 200},
        {"name": "close", "kernel_size": 3},
    ],
    background_class=0,
)
```
