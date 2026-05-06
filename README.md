# pd_xray
Process Dynamics Group Experiment Processing Library

---

## Configuration

`Config` loads a YAML file and lets you read values using dot-notation keys. It is the standard way to pass processing parameters through the pipeline without hardcoding them in scripts.

### Loading a config file

```python
from pd_xray.core import Config

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

---

## Accessing Experiment Data from S3

`S3Backend` handles all storage operations against any S3-compatible store (AWS, Ceph, MinIO). It is format-agnostic: it lists keys, moves bytes, and exposes `open_fileobj()` so format readers can stream a file directly without downloading it first. `HDF5Reader` is responsible for all HDF5-specific operations.

### Connecting to the store

```python
from pd_xray.data import S3Backend, HDF5Reader

backend = S3Backend(
    bucket="xray-data",
    endpoint_url="https://s3.endpoint.ac.uk",
    prefix="campaign/experiment/data",
)

backend.connect(
    aws_access_key_id="YOUR_KEY",
    aws_secret_access_key="YOUR_SECRET",
)

reader = HDF5Reader()
```

`connect()` must be called before any data operation. All backend methods raise `RuntimeError` if called before `connect()`. `connect()` itself makes no network call — the first real request happens when you call `list_files()` or any read method.

`S3Backend` can also be used as a context manager, which calls `disconnect()` automatically on exit:

```python
with S3Backend(bucket="xray-data", endpoint_url="https://s3.endpoint.ac.uk") as b:
    b.connect(aws_access_key_id="YOUR_KEY", aws_secret_access_key="YOUR_SECRET")
    files = b.list_files()
```

### Listing available files

```python
# List everything under the configured prefix
files = backend.list_files()
for f in files:
    print(f.path, f.size_bytes)

# Filter to HDF5 files only
hdf5_files = backend.list_files(pattern="*.h5")
```

### Checking shape and dtype

`lazy_open_remote` reads only the HDF5 superblock and chunk index — a handful of range requests even for a 90 GB file. Use it to check shape and dtype before deciding what to load.

```python
with reader.lazy_open_remote(backend, "scan_0001.h5") as arr:
    print(arr.shape)   # e.g. (2000, 2048, 2048)
    print(arr.dtype)
```

For the full file structure (all groups, datasets, and attributes), `inspect()` only accepts a local path — download the file first:

```python
backend.read_file_to_local("scan_0001.h5", "/scratch/scan_0001.h5")

structure = reader.inspect("/scratch/scan_0001.h5")
print(structure["groups"])    # list of group paths
print(structure["datasets"])  # {path: {shape, dtype, attrs}}

header = reader.read_header("/scratch/scan_0001.h5")
print(header["shape"], header["n_frames"], header["attrs"])
```

### Loading slices without downloading

`lazy_open_remote` returns a `LazyHDF5Array` that fetches only the HDF5 chunks you index, using S3 range requests under the hood. No pixel data is downloaded until you slice the array.

```python
with reader.lazy_open_remote(backend, "scan_0001.h5") as arr:
    frame = arr[0]            # fetches only the chunks for frame 0
    stack = arr[100:120]      # fetches only those 20 frames
    tile  = arr[0, :512, :512]
```

**Performance and block size.** HDF5 access requires many scattered reads — the library must traverse a B-tree to locate each chunk before it can fetch the data. Each unique region of the file that h5py touches is one S3 `GetObject` request. The `block_size` parameter (default 32 MB) controls the granularity of those requests. A larger value means fewer round trips, which matters on high-latency servers.

As a rough guide for a 2048x2048 uint16 dataset (one frame ~8 MB uncompressed):

| block_size | Approximate requests for 10 frames |
|---|---|
| 2 MB | ~150 |
| 32 MB (default) | ~12 |
| 128 MB | ~5 |

```python
backend = S3Backend(
    bucket="xray-data",
    endpoint_url="https://s3.endpoint.ac.uk",
    prefix="campaign/experiment/data",
    block_size=128 * 1024 * 1024,   # 128 MB for a high-latency server
)
```

The right value depends on your server's latency and the size of the HDF5 chunks in your files. Start with the default; if slices are taking several minutes, increase `block_size`.

### Downloading a file to local disk

If you need to make many different slices from the same file, or run operations that require the full dataset, download it once and read locally. A single sequential download is faster than many range requests.

```python
backend.read_file_to_local("scan_0001.h5", "/scratch/scan_0001.h5")

arr = reader.lazy_read("/scratch/scan_0001.h5")
frame = arr[0]
stack = arr[100:120]
```

`lazy_read` does not need a context manager — the file is opened and closed on each slice access. The parent directory is created automatically if it does not exist.

---

## Visualisation

`view_frames`, `save_selection`, and `apply_and_view` are designed for inspecting and saving array sections after loading them from remote or local storage.

```python
from pd_xray.visualisation import view_frames, save_selection, apply_and_view
```

### Viewing frames

`view_frames` accepts a 2D `(H, W)` image or a 3D `(Z, H, W)` stack. In Jupyter notebooks it renders an HTML slider that works with the default inline backend — no extra packages needed. Outside notebooks it uses a matplotlib Slider widget.

```python
with reader.lazy_open_remote(backend, "scan_0001.h5") as arr:
    section = arr[100:120]   # loads 20 frames into memory

view_frames(section)
view_frames(section, cmap="viridis", title="Scan 0001 frames 100-120")
```

### Saving a selection

`save_selection` wraps `np.save`. It appends `.npy` if the path does not already have that extension and creates any missing parent directories.

```python
save_selection(section, "/scratch/scan_0001_frames_100_120.npy")

# Load it back
import numpy as np
arr = np.load("/scratch/scan_0001_frames_100_120.npy")
```

### Applying image processing and viewing

`apply_and_view` runs an `ImageProcessor` pipeline on the array, displays the result, and returns the processed array so you can save it or inspect it further.

```python
from pd_xray.processing import ImageProcessor

proc = (
    ImageProcessor()
    .gaussian_blur(sigma=1.5)
    .normalise()
)

processed = apply_and_view(section, proc)
save_selection(processed, "/scratch/scan_0001_processed.npy")
```

### End-to-end example

```python
from pd_xray.data import S3Backend, HDF5Reader
from pd_xray.processing import ImageProcessor
from pd_xray.visualisation import view_frames, apply_and_view, save_selection

backend = S3Backend(
    bucket="xray-data",
    endpoint_url="https://s3.endpoint.ac.uk",
    prefix="campaign/experiment/data",
    block_size=128 * 1024 * 1024,
)
backend.connect(aws_access_key_id="YOUR_KEY", aws_secret_access_key="YOUR_SECRET")
reader = HDF5Reader()

# Check what is in the file
with reader.lazy_open_remote(backend, "scan_0001.h5") as arr:
    print(arr.shape, arr.dtype)

# Load a section and inspect it
with reader.lazy_open_remote(backend, "scan_0001.h5") as arr:
    section = arr[-20:]        # last 20 frames

view_frames(section)           # browse raw frames

# Apply processing and save
proc = ImageProcessor().gaussian_blur(sigma=1.5).normalise()
processed = apply_and_view(section, proc, title="Processed")
save_selection(processed, "/scratch/scan_0001_last20_processed.npy")
```

---

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
