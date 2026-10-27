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
loads them: give it a function that turns a location into a frame. `pd_xray.io.reference_frames` reads a folder
(every image in it) or a list of files, relative to a base folder, as the web app does:

```python
from pd_xray.io import open_sequence, reference_frames
from pd_xray.processing import average_frames

scan = "data/24_I20231030_1/0042"
pipeline = Pipeline([Step("flat_field", {"dark": "dar", "flat": ["flat/0042__000010_img.tif"]})])
pipeline.references            # what still needs loading
corrected = pipeline.resolve(lambda ref: average_frames(reference_frames(ref, base=scan)))
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

Wrap the dataset with `pd_xray.io.map_frames` and write it with `convert`. Frames are read one at a time and
`workers` are processed at once:

```python
from pd_xray.core import CancelToken
from pd_xray.io import convert, map_frames, open_sequence

ds = open_sequence(scan, "*_img.tif")
convert(map_frames(ds, corrected, workers=16), "out/corrected.tif", progress=print)   # float32, exact
convert(map_frames(ds, corrected, workers=16), "out/corrected_u16.tif", dtype="uint16", value_range=(0, 2))
```

`progress` receives one update per frame; `cancel=CancelToken()` lets another thread stop the run between frames, and
a cancelled run leaves no output. `apply_pipeline(pipeline, frames)` is the lower-level form: a lazy iterator of
processed frames for `write_frames`.

### Reproducing a web app run

Every run from the web app writes `<output>.pipeline.yaml` next to its output: the pipeline (references relative to
the dataset's folder) and a `run` section with the source, frames, output type, value range, date and versions. The
same record is in the output's metadata (`pd_xray_run`). To repeat the run, or apply it to another scan with the same
layout:

```python
import yaml
from pathlib import Path
from pd_xray.io import convert, map_frames, open_dataset, reference_frames
from pd_xray.processing import Pipeline, average_frames

record = Path("out/scan_processed.tif.pipeline.yaml")
run = yaml.safe_load(record.read_text())["run"]
pipeline = Pipeline.load(record)                       # ignores the run section
# A dataset on a NAS/S3 has run["storage"] (its URL): open that storage and pass storage= to open_* and reference_frames.
with open_dataset(run["source"], dataset=run["dataset"]) as ds:  # TIFF: no dataset=; folder: open_sequence(source, pattern)
    base = Path(run["source"]).parent                  # the dataset's folder (the folder itself for a sequence)
    resolved = pipeline.resolve(lambda ref: average_frames(reference_frames(ref, base)))
    frames = slice(*run["frames"]) if run["frames"] else None
    convert(map_frames(ds, resolved, workers=16), "out/again.tif", frames, run["value_range"], dtype=run["dtype"])
```

The result is identical to the web app's.

## Tone mapping

`tone_map(values, exposure=…, contrast=…, highlights=…, shadows=…, whites=…, blacks=…, curve=…)` adjusts values
already normalised to 0..1 (sliders −100..100, exposure in stops). It's the same function the web app uses for
display, so a `tone` step reproduces what you saw on screen.

```python
from pd_xray.processing import tone_map
x = np.clip((frame - 1000) / (3000 - 1000), 0, 1)
y = tone_map(x, exposure=0.5, contrast=20, curve=[[0, 0], [0.5, 0.6], [1, 1]])
```
