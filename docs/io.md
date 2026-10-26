# `pd_xray.io`: reading and writing image data

```python
from pd_xray.io import open_dataset, open_sequence, convert, write_frames, iter_frames, map_frames, reference_frames
```

Formats: HDF5 (`.h5 .hdf5 .hdf .nxs`), TIFF (single and multi-page, BigTIFF), PNG, JPG. Install the extras you need:
`pd-xray-io[hdf5]`, `[tiff]`, `[png]` or `[all]`.

## Datasets

Every reader returns a `Dataset`: a lazy array. Nothing is read until you index it, and then only that region.

```python
ds = open_dataset("data/pr_110_proj_0040.h5", dataset="projections")
ds.shape, ds.dtype, ds.axes        # (500, 512, 512), float32, ('frame', 'y', 'x')
ds.nbytes                          # uncompressed size in bytes
ds.metadata                        # file and dataset attributes as plain Python values
ds.warnings                        # problems found while opening (e.g. missing frames); empty if none

ds[250]                            # one frame, a NumPy array of shape (512, 512)
ds[10:20, :64, :64]                # a block: (10, 64, 64)
ds[[400, 3, 250]]                  # chosen frames, in that order
ds[-1]                             # the last frame
ds.close()                         # or: with open_dataset(...) as ds:
```

Data comes back in its stored dtype (usually `uint16`). Convert explicitly when you need to, e.g.
`ds[0].astype("float32")`.

### Axis names

Axes name what each dimension means. They always end with `("y", "x")` (plus `"channel"` for RGB). If the file
doesn't say, the frame axis is called `"frame"`. Give your own with `axes=`:

```python
ds = open_dataset("scan.h5", dataset="projections", axes=("angle", "y", "x"))
```

## HDF5

```python
from pd_xray.io.hdf5 import inspect_hdf5

for info in inspect_hdf5("scan.h5"):          # lists datasets without reading pixels
    print(info.name, info.shape, info.dtype, info.chunks, info.compression, info.is_image, info.warnings)

ds = open_dataset("scan.h5", dataset="entry/data/data")
```

- `dataset=` can be left out when the file has exactly one image dataset; otherwise `DatasetSelectionError` lists
  the candidates.
- Axis names come from HDF5 dimension labels when present.
- Compressed files (gzip, blosc, lz4, zstd, … via hdf5plugin) are read transparently. Reading one frame costs one
  chunk read, so files chunked per frame are fastest.
- **Broken extents**: some acquisition software writes a wrong frame count (e.g. 8.9e18 frames). The dataset is
  trimmed to the frames actually written, frames that were never written are listed in
  `ds.metadata["missing_frames"]` (they read as zeros), and `ds.warnings` explains it.

## TIFF, PNG, JPG

```python
ds = open_dataset("data/0042_img_multipage.tif")   # multi-page: one frame per page, memory-mapped when possible
img = open_dataset("photo.png")                    # single image: shape (y, x) or (y, x, channel)
img.metadata.get("tiff_orientation")               # TIFF orientation tag, recorded but NOT applied
```

## Folders of single images (sequences)

```python
images = open_sequence("data/24_I20231030_1/0042", "*_img.tif")   # natural order: 2, 10, 100
darks = open_sequence("data/24_I20231030_1/0042/dar")             # every image file in the folder
files = open_sequence(["a.tif", "c.tif", "b.tif"])                # explicit list, kept in this order
images.files[0], images[0].shape
```

Each frame is read from its own file on demand. All files must have the same shape and dtype (a mismatch raises
when that frame is read).

## Iterating frame by frame

`iter_frames` yields 2D frames one at a time, so memory stays at one frame however big the dataset:

```python
from pd_xray.io import iter_frames, compute_value_range

for frame in iter_frames(ds):            # all frames
    ...
for frame in iter_frames(ds, first=range(0, 500, 10)):   # every 10th
    ...
low, high = compute_value_range(ds)      # min/max over all frames (reads everything once)
```

## Writing

`write_frames` writes frames from any iterable (e.g. a generator), so the output never has to fit in memory. The
extension picks the format.

```python
import numpy as np
from pd_xray.io import write_frames

frames = (ds[i] * 2 for i in range(ds.shape[0]))
write_frames("out.tif", frames, shape=ds.shape, dtype=np.dtype("float32"), axes=ds.axes, metadata={"note": "x2"})
```

| Format | Notes | Options |
|---|---|---|
| `.tif` / `.tiff` | Multi-page, any dtype, BigTIFF above ~4 GB. Uncompressed by default (can be memory-mapped). | `compression="zlib"` |
| `.h5` / `.hdf5` | One dataset, chunked per frame, axes as dimension labels, metadata as attributes. | `dataset="data"`, `compression="gzip"`, `compression_opts=4` |
| `.png` | One file per frame (`out_00000.png`, …). uint8/uint16 only. | |
| `.jpg` / `.jpeg` | One file per frame, uint8, lossy. | `quality=95` |

The frames' dtype must match `dtype`, and the format must support it (`supports(path, dtype, frame_shape)` tells
you); otherwise `write_frames` raises before writing. Output goes to a hidden `.out.tif.pd-xray-writing/` folder
next to it and is moved into place when complete, so a failed or cancelled write leaves nothing behind and an
existing file is only replaced on success.

## Converting

`convert` copies a dataset (or some frames) to another format:

```python
from pd_xray.io import convert

convert(ds, "out/scan.tif")                          # exact copy, any dtype
convert(ds, "out/scan.h5", compression="gzip")
convert(ds, "out/png/frame.png", frames=slice(0, 100))   # first 100 frames as PNG files
convert(ds, "out/preview.jpg", frames=7, quality=90)
```

- Values are copied **exactly** when the format can store the dtype (TIFF and HDF5 always; PNG for uint8/uint16).
- Otherwise (float → PNG, uint16 → JPG) values are scaled linearly with **one** range for all frames, so frames stay
  comparable. The mapping is saved in the output's metadata as `value_scaling` (original ≈ offset + pixel × scale).
  By default the range is the data's min/max (an extra pass); give `value_range=(low, high)` to skip that and clip.
- `dtype="uint16"` (or `"uint8"`, `"float32"`, …) chooses the output type: other integer types are scaled as above,
  float types are exact casts.
- `progress=` and `cancel=` work as described in [core](core.md).

## Processing a whole dataset

`map_frames(ds, function, workers=…)` is a lazy dataset of `function(frame)` for every frame, e.g. a processing
pipeline. Pass it to `convert`: frames are read one at a time and `workers` of them are processed at once (filters
scale almost linearly with threads; integer scaling also runs in those threads).

```python
from pd_xray.processing import Pipeline, Step

pipeline = Pipeline([Step("median", {"size": 3}), Step("crop", {"top": 100, "height": 2000})])
processed = map_frames(ds, pipeline, workers=16)     # shape from processing frame 0 once
processed[5]                                         # one processed frame
convert(processed, "out/median.tif", frames=slice(0, 1000), dtype="uint16", value_range=(0, 65535))
```

The source stays open (`processed.close()` does nothing); close it yourself.

## Reference frames

`reference_frames(reference, base)` yields every frame a pipeline reference names: a folder (every image file
directly in it, natural order) or a list of files, relative to `base` (usually the dataset's folder). Average them
to resolve a pipeline:

```python
from pd_xray.processing import average_frames
resolved = pipeline.resolve(lambda ref: average_frames(reference_frames(ref, base="data/scan")))
```

`reference_files(reference, base)` lists the files only.

## In-memory arrays

`ArrayDataset` wraps a NumPy array (or a `np.memmap`) so it can be used wherever a `Dataset` is expected:

```python
from pd_xray.io import ArrayDataset
ArrayDataset(np.zeros((10, 256, 256), np.uint16), axes=("time", "y", "x"))
```
