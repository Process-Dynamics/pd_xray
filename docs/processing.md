# `pd_xray.processing`: operations and pipelines

```python
from pd_xray.processing import OPERATIONS, Pipeline, Step, apply_pipeline, average_frames, tone_map
```

Processing works on NumPy frames, independent of file formats: read with [`pd_xray.io`](io.md), process, write with
`pd_xray.io.write_frames`. Inputs can be any numeric dtype; every step works in `float32` and returns `float32`.

## A pipeline

A `Pipeline` is a list of `Step`s. Call it on a frame:

```python
pipeline = Pipeline([
    Step("median", {"size": 3}),
    Step("gaussian", {"sigma": 1.5}),
    Step("crop", {"top": 100, "left": 0, "height": 1800, "width": 0}),
    Step("rotate", {"angle": "180"}, enabled=False),     # kept, but skipped
])
out = pipeline(ds[100])            # float32 frame
```

Parameters left out take their default. Unknown operations or invalid values raise `PdXrayError` when the pipeline is
built, not halfway through a run.

## Operations

| Name | Parameters (default) | What it does |
|---|---|---|
| `flat_field` | `dark`, `flat` (reference, optional) | (frame − dark) / (flat − dark); pixels where flat ≤ dark become 0 |
| `invert` | `low` (0), `high` (65535) | high + low − value |
| `gaussian` | `sigma` (1.0, 0.1–100) | Gaussian filter, reflected edges |
| `median` | `size` (3, 1–51) | Median over a square window: removes hot pixels, zingers |
| `normalise` | `low_percentile` (0), `high_percentile` (100) | This frame's percentiles → 0..1 (per frame) |
| `clip` | `low`, `high` (optional) | Limit values to [low, high] |
| `threshold` | `value` (0), `keep` ("above"/"below") | 1 where kept, 0 elsewhere |
| `tone` | `black` (0), `white` (1), `exposure`, `contrast`, `highlights`, `shadows`, `whites`, `blacks` (0), `curve` | Photo-style adjustments between black and white; output stays in black..white |
| `crop` | `top`, `left`, `height`, `width` (0 = to the edge) | Keep a rectangle |
| `rotate` | `angle` ("90"/"180"/"270") | Clockwise rotation |
| `flip` | `direction` ("horizontal"/"vertical") | Mirror |
| `rolling_ball` | `radius` (50), `light_background` (False) | ImageJ's rolling-ball background subtraction |

The same list, with descriptions and limits, is available in code:

```python
for op in OPERATIONS.values():
    print(op.name, op.title, [(p.name, p.kind, p.default) for p in op.params])
```

Rough speed on a 2304 × 4096 frame (CPU): gaussian 0.7 s, median 3 about 3 s, rolling ball r=50 about 2.6 s
(r=10 about 15 s).

## Flat/dark correction with reference frames

Reference parameters (`dark`, `flat`) are file locations in a pipeline, so the pipeline can be saved. `resolve`
loads them: give it a function that turns a location into a frame. A common choice averages every frame found:

```python
from pathlib import Path
from pd_xray.io import open_dataset, open_sequence, iter_frames

def load(reference):
    """A folder (all images in it), a file, or a list of files -> one averaged float32 frame."""
    paths = [reference] if isinstance(reference, str) else list(reference)
    if len(paths) == 1 and Path(paths[0]).is_dir():
        dataset = open_sequence(paths[0])
    elif len(paths) == 1:
        dataset = open_dataset(paths[0])
    else:
        dataset = open_sequence(paths)
    return average_frames(iter_frames(dataset), method="mean")

scan = "data/24_I20231030_1/0042"
pipeline = Pipeline([Step("flat_field", {"dark": f"{scan}/dar", "flat": [f"{scan}/flat/0042__000010_img.tif"]})])
pipeline.references            # what still needs loading
corrected = pipeline.resolve(load)
out = corrected(open_sequence(scan, "*_img.tif")[0])
```

`average_frames(frames, method="median")` is robust to outliers but holds all frames in memory; `"mean"` doesn't.

## Pipeline files

Pipelines save to YAML, so a processing recipe can be reused and shared:

```python
pipeline.save("recipe.yaml")
pipeline = Pipeline.load("recipe.yaml")
text = pipeline.to_yaml()
```

```yaml
pd_xray_pipeline: 1
steps:
- op: median
  params:
    size: 3
  enabled: true
- op: flat_field
  params:
    dark: data/24_I20231030_1/0042/dar
    flat:
    - data/24_I20231030_1/0042/flat/0042__000010_img.tif
  enabled: true
- op: rotate
  params:
    angle: '180'
  enabled: false
```

A pipeline with loaded arrays can't be saved; save the unresolved one.

## Processing a whole dataset

`apply_pipeline` processes frames lazily, one at a time; pass it to `write_frames` so memory stays at one frame:

```python
import numpy as np
from pd_xray.core import CancelToken
from pd_xray.io import open_sequence, iter_frames, write_frames
from pd_xray.processing import apply_pipeline

ds = open_sequence(scan, "*_img.tif")
first = corrected(ds[0])                                   # output frame shape (crop/rotate may change it)
frames = apply_pipeline(corrected, iter_frames(ds), total=ds.shape[0], progress=print)
write_frames("out/corrected.tif", frames, shape=(ds.shape[0], *first.shape), dtype=np.dtype("float32"),
             axes=ds.axes, metadata={"pipeline": pipeline.to_dict()})
```

`progress` receives one update per frame; `cancel=CancelToken()` lets another thread stop the run between frames.

## Tone mapping

`tone_map(values, exposure=…, contrast=…, highlights=…, shadows=…, whites=…, blacks=…, curve=…)` adjusts values
already normalised to 0..1 (sliders −100..100, exposure in stops). It's the same function the web app uses for
display, so a `tone` step reproduces what you saw on screen.

```python
from pd_xray.processing import tone_map
x = np.clip((frame - 1000) / (3000 - 1000), 0, 1)
y = tone_map(x, exposure=0.5, contrast=20, curve=[[0, 0], [0.5, 0.6], [1, 1]])
```
