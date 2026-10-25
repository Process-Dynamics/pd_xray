# Manual testing guide: `pd_xray.core`, `io`, `processing`, `server`, web UI (steps 1.1–1.7)

Each section is a block to paste into Python, with the expected result next to it. Run them from the repo root.

## 0. Setup

```bash
source ~/miniconda3/bin/activate pd-xray-dev     # conda isn't on PATH in plain shells
cd ~/pd_xray
ruff check packages && ruff format --check packages && mypy && pytest   # expect: all pass, 66 tests
cd frontend && npm ci && npm run format:check && npm test && npm run build && cd ..   # 6 tests; build goes into pd-xray-server
python                                           # then paste the blocks below
```

Shared setup for all blocks:

```python
import logging, time, numpy as np
from pathlib import Path
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
from pd_xray.core import CancelToken, OperationCancelledError, track, validate_axes, frame_axes
from pd_xray.io import open_dataset, open_sequence, convert, ArrayDataset
from pd_xray.io.hdf5 import inspect_hdf5, DatasetSelectionError

SIM  = "data/pr_110_proj_0040.h5"
BIG  = "data/MG36212_1_0091_narrowfov_000000.h5"
SEQ  = "data/24_I20231030_1/0042"
MULTI = "data/0042_img_multipage.tif"
OUT = Path("/tmp/pd_xray_manual"); OUT.mkdir(exist_ok=True)   # all outputs go here; delete afterwards
```

## 1. core: axes, progress, cancel

```python
validate_axes(("angle", "y", "x"), 3)      # -> ('angle', 'y', 'x')
validate_axes(("x", "y"), 2)               # ValueError: must end with ('y', 'x')
validate_axes(("channel", "y", "x"), 3)    # ValueError: channel only allowed last
frame_axes(("time", "angle", "y", "x"))    # -> ('time', 'angle')

list(track(range(3), progress=print))      # prints Progress 0/3 … 3/3, returns [0, 1, 2]

stop = CancelToken()
for i in track(range(10), cancel=stop):
    if i == 2: stop.cancel()               # OperationCancelledError raised at i == 3
```

## 2. HDF5: simulated sample

```python
[(i.name, i.shape, i.dtype, i.compression) for i in inspect_hdf5(SIM)]
# -> projections and projections_E, both (500, 512, 512) float32 gzip

open_dataset(SIM)          # DatasetSelectionError: 2 image datasets; choose one with `dataset=` …

ds = open_dataset(SIM, dataset="projections", axes=("angle", "y", "x"))
ds, ds.metadata            # HDF5Dataset(shape=(500, 512, 512), dtype=float32, axes=('angle','y','x')), file attrs
ds[250].mean()             # ≈ 0.545264
ds[-1].shape, ds[10:20, :64, :64].shape, ds[[400, 3, 250]].shape, ds[::-100].shape
# -> (512, 512) (10, 64, 64) (3, 512, 512) (5, 512, 512)
ds[500]                    # IndexError
ds.close()
```

## 3. HDF5: broken 99 GB file

```python
t = time.perf_counter(); big = open_dataset(BIG); print(f"{(time.perf_counter()-t)*1e3:.0f} ms")
# 2 warnings logged (claims ~8.9e18 frames, using 5616; 18 frames never written). Open in ≲ 10 ms.
big.shape, big.dtype                       # (5616, 2304, 4096) uint16
big.warnings                               # the same two messages
big.metadata["missing_frames"][:5]         # [353, 743, 882, 1263, 1653]
big[353].max(), big[354].max()             # 0 (missing frame) vs a real value
t = time.perf_counter(); big[2500]; print(f"{(time.perf_counter()-t)*1e3:.0f} ms")   # ~10–25 ms
big[5616]                                  # IndexError
```

## 4. TIFF: sequences and multi-page

```python
img = open_sequence(SEQ, "*_img.tif"); dar = open_sequence(SEQ, "*_dar.tif")
img.shape, dar.shape                       # (16, 2101, 5120) (9, 2101, 5120)
[f.name for f in img.files][:4]            # natural order: …000010, …000011, …000012, …002015
img.metadata                               # {'tiff_orientation': 3}  (recorded, not applied)
img[0].dtype, img[0].max()                 # uint16, ≤ 4095 (12-bit)

m = open_dataset(MULTI)
m.shape, m._memmap is not None             # (16, 2101, 5120) True
all(np.array_equal(m[i], img[i]) for i in range(16))   # True
open_dataset(f"{SEQ}/0042__000010_img.tif").shape      # single file: (2101, 5120), axes ('y', 'x')
```

Optional, for the compressed page-wise path:

```python
convert(img, OUT / "seq_zlib.tif", compression="zlib")
z = open_dataset(OUT / "seq_zlib.tif"); z._memmap is None, np.array_equal(z[3], img[3])   # True True
```

## 5. Conversion

Exact copies (no `value_scaling` in the output metadata):

```python
convert(img, OUT / "seq.h5")                                     # TIFF sequence -> HDF5
h = open_dataset(OUT / "seq.h5"); h.axes, np.array_equal(h[5], img[5]), h.metadata   # ('frame','y','x') True {'tiff_orientation': 3}

convert(big, OUT / "big.png", frames=[0, 353, 5615])            # uint16 -> PNG, 3 files big_00000..2.png
p = open_dataset(OUT / "big_00002.png")
p.dtype, np.array_equal(p[...], big[5615]), p.metadata["source_frames"]   # uint16 True [0, 353, 5615]

convert(big, OUT / "big_subset.tif", frames=slice(0, 20))       # subset -> multi-page TIFF (~380 MB)
open_dataset(OUT / "big_subset.tif").metadata["source_missing_frames"][:3]   # source's missing frames, renamed
```

Scaled conversions (a warning is logged, and the mapping is stored):

```python
convert(ds := open_dataset(SIM, dataset="projections"), OUT / "sim.png", frames=slice(0, 500, 100))   # float32 -> uint16 PNG
s = open_dataset(OUT / "sim_00002.png"); vs = s.metadata["value_scaling"]; vs
recovered = vs["offset"] + s[...] * vs["scale"]
np.abs(recovered - ds[200]).max() <= vs["scale"] / 2 + 1e-9     # True: within half a step

convert(big, OUT / "big.jpg", frames=2500)                       # uint16 -> 8-bit JPG, lossy warning
j = open_dataset(OUT / "big.jpg"); j.dtype, j.metadata["value_scaling"]["value_range"]   # uint8 [min, max] of frame 2500

convert(ds, OUT / "sim_fixed.png", frames=0, value_range=(0.2, 0.5))   # values outside range clipped -> warning
```

Progress and cancel:

```python
stop = CancelToken()
def cb(p):
    print(p.current, p.total, p.message)
    if p.current == 5: stop.cancel()
convert(big, OUT / "cancel.tif", frames=slice(0, 100), progress=cb, cancel=stop)
# prints 0..5 of 100, then OperationCancelledError. The partial file is left behind for now (decision D8, step 1.8).
```

## 6. Error messages

```python
open_dataset("data/nope.h5")               # FileNotFoundError
open_dataset("README.md")                  # UnsupportedFormatError, lists supported extensions
convert(img, OUT / "x.bmp")                # UnsupportedFormatError
open_sequence(SEQ, "*_flat.tif")           # PdXrayError: No files match …
```

## 7. Processing

```python
from pd_xray.io import iter_frames, write_frames
from pd_xray.processing import OPERATIONS, Pipeline, Step, apply_pipeline, average_frames
sorted(OPERATIONS); OPERATIONS["gaussian"].params     # registry the UI builds controls from
OPERATIONS["median"](img[0], size=2.5)                # PdXrayError: must be an integer

pipe = Pipeline([Step("flat_field", {"dark": f"{SEQ}/*_dar.tif"}), Step("median", {"size": 3}),
                 Step("crop", {"top": 100, "height": 1000}), Step("invert", enabled=False)])
print(pipe.save(OUT / "p.yaml").read_text())          # defaults filled in, disabled step kept
def load(ref):                                        # turns a reference string into a frame
    d, pattern = ref.rsplit("/", 1)
    return average_frames(iter_frames(open_sequence(d, pattern)))
run = Pipeline.load(OUT / "p.yaml").resolve(load)
apply_pipeline(Pipeline.load(OUT / "p.yaml"), img)    # PdXrayError: Unresolved references …
first = run(img[0]); first.shape, first.dtype         # (1000, 5120) float32, ~3 s
frames = apply_pipeline(run, iter_frames(img, range(4)), total=4, progress=print)
write_frames(OUT / "proc.tif", frames, (4,) + first.shape, first.dtype, img.axes)
np.array_equal(open_dataset(OUT / "proc.tif")[0], first)   # True

f = big[2500].astype(np.float32)
t = time.perf_counter(); OPERATIONS["rolling_ball"](f, radius=50); time.perf_counter() - t   # a few s
```

## 8. Server

Terminal 1:

```bash
pd-xray serve --root data --no-browser       # http://127.0.0.1:8000, only data/ is visible
```

Browser: `http://127.0.0.1:8000/docs` is the interactive API page (try requests with "Try it out").

Terminal 2 (`python`), with `D` = absolute path of `data/`:

```python
import httpx, time, json, asyncio, websockets
D = str(Path("data").resolve()); c = httpx.Client(base_url="http://127.0.0.1:8000", timeout=60)
c.get("/api/files", params={"path": D}).json()["entries"]     # folders first, natural order, `openable` flags
c.get("/api/files", params={"path": "/etc"}).status_code       # 403: outside the roots
c.get("/api/inspect", params={"path": f"{D}/{Path(SIM).name}"}).json()   # both datasets, is_image True
c.post("/api/datasets", json={"path": f"{D}/{Path(SIM).name}"}).json()    # 400: choose one with `dataset=`
big = c.post("/api/datasets", json={"path": f"{D}/{Path(BIG).name}"}).json()
big["frame_shape"], big["levels"], big["warnings"]             # [5616] 4, the two broken-extent warnings
b = f"/api/datasets/{big['id']}/frames"
c.get(f"{b}/2500/stats").json()                                # min/max, auto contrast low/high
```

Open in the browser (replace `ID`): whole frame, then a full-resolution tile, then fixed contrast:

- `http://127.0.0.1:8000/api/datasets/ID/frames/2500/tiles/3/0_0.png`
- `…/frames/2500/tiles/0/4_2.png`, `…/frames/2500/tiles/3/0_0.jpeg?vmin=0&vmax=4000`
- `…/frames/353/tiles/3/0_0.png` (missing frame: black), `…/frames/9999/stats` (400 with a clear message)

```python
for f in range(2600, 2605):                                    # cold frames: ≈ 20–50 ms each
    t = time.perf_counter(); c.get(f"{b}/{f}/tiles/3/0_0.jpeg"); print(f"{(time.perf_counter()-t)*1e3:.0f} ms")
seq = c.post("/api/datasets", json={"path": f"{D}/24_I20231030_1/0042", "pattern": "*_img.tif"}).json(); seq["shape"]

async def watch():                                             # job progress over the WebSocket, then cancel
    async with websockets.connect("ws://127.0.0.1:8000/api/events") as ws:
        job, sent = c.post("/api/jobs/value-range", json={"dataset_id": big["id"]}).json(), False
        while True:
            m = json.loads(await ws.recv())["job"]           # on connect, every kept job is sent first
            if m["id"] != job["id"]: continue
            print(m["state"], m["current"], m["total"], m["eta_seconds"])
            if m["current"] >= 30 and not sent: sent = c.post(f"/api/jobs/{job['id']}/cancel")
            if m["state"] in ("done", "failed", "cancelled"): break
asyncio.run(watch())   # progress lines (~10/s), then "cancelled" a few frames after 30
c.delete(f"/api/datasets/{big['id']}").status_code             # 204
```

While the job runs, tiles from another terminal still load in well under 0.2 s. Stop the server with Ctrl+C.

## 9. Web UI

`pd-xray serve --root data` (opens the browser), or on the shared server `pd-xray serve --root data --host 0.0.0.0
--no-browser` and browse to `http://<server>:8000/`. Check each line:

- Left: `data` → folders first, then files with sizes. Filter box narrows the list.
- Click the 99 GB `.h5`: frame 0 appears in < 0.5 s. Right panel: shape, uint16, the two ⚠ warnings, Metadata.
- Click in the viewer once, then ←/→: one frame per press, ~0.1 s or less. Shift+→ jumps 10, Home/End to the ends.
  Hold →: frames keep coming (~10/s) and stop exactly where you release. "showing N" turns yellow while loading.
- Drag the slider fast: no lag build-up; it settles on the frame you stop at. Type a frame number in the box.
- Scroll to zoom to single pixels (sharp, not blurred), drag to pan, Fit to reset. Stepping keeps the zoom.
- Frame 353 (missing): black. Min/Max: type 0 / 4000 + Enter. Auto, Frame range. Colours: viridis, fire, …
- Dataset range: status bar shows progress + ETA. × cancels it ("cancelled"). Let one finish on the 104 MB file:
  Min/Max are set to the whole-dataset range.
- `24_I20231030_1/0042`: the green `▦ 0042__*_img.tif  13 files` row opens the sequence; `dar/` has `*_dar.tif`.
- `pr_110_proj_0040.h5`: a picker lists `projections` and `projections_E`.
- Stop the server (Ctrl+C) with the page open: red dot, "Reconnecting…", stepping says "Can't reach the server".
  Start it again: green dot, the dataset reopens by itself and stepping works.
- Two browser tabs at once: both work independently.

## 10. Processing panel

Open `24_I20231030_1/0042` → `▦ 0042__*_img.tif`. In the right panel, under Processing:

- `+ Add step… → Flat/dark correction`: the view switches to Processed and the status bar says the step needs a dark
  or flat frame.
- dark → Choose… (opens in `0042/`) → `dar` → "Use this folder (9 files)". flat → Choose… → `flat` → "Select shown"
  → "Use 3 selected files" (each file shows its name and a checkbox). The flat-corrected frame appears (values ~1; contrast is automatic). Step time ~0.05 s.
- Add Median filter: ~1 s; each step shows its time. Hold → while Processed: the slider follows at once, the status
  bar counts seconds, and only the frame you stop on is processed ("showing N" catches up).
- P (or the buttons) switches Original/Processed instantly once both are computed. Each has its own Min/Max.
- Change median size to 5 + Enter: ~3 s. Size 0: the status bar names the step and the allowed range.
- ↑/↓ reorder steps, the checkbox disables one, × removes it. Processed is disabled when no step is enabled.
- Esc or clicking outside closes the picker without changing anything.
- Add Threshold (value 500), Rotate and Flip: a white mask on black, rotated and mirrored. Threshold's default
  value 0 keeps everything, so the frame is all white until you set a value.
- Every operation also works on `MG36212_1_0091_narrowfov_000000.h5` (flat field needs same-sized frames, else a clear
  error).

In Python, a step list from the UI behaves the same:

```python
from pd_xray.processing import Pipeline, Step
p = Pipeline([Step("flat_field", {"dark": f"{SEQ}/dar", "flat": [f"{SEQ}/flat/0042__000010_img.tif"]})])
p.references        # {'…/dar', ('…/0042__000010_img.tif',)}: lists become tuples
```

## 11. Histogram and tone

Open `24_I20231030_1/0042` → `▦ 0042__*_img.tif`. Top of the right panel:

- Histogram of the frame on screen (around Min–Max; "full range" shows min..max, "log" switches the count scale). It
  follows the frame you step to (~0.1 s after it shows) and the processed view.
- Drag the black/white triangles: Min/Max change on release and the frame reloads (~0.35 s).
- Tone (display) sliders: the orange line updates while dragging, the frame on release. Double-click a slider resets it.
  Arrow keys on a focused slider move it, not the frame.
- Click on the blue curve between the triangles and drag: a new point. Double-click a point removes it.
- Histogram "Reset": straight curve and automatic Min/Max (the sliders keep their own Reset).
- "Add as step": a Tone step appears with black/white = Min/Max, the display tone resets, Processed shows and looks the
  same (exact at full zoom; zoomed out, edges can differ slightly because the step works on full-resolution pixels).
- `+ Add step… → Tone`: starts with the current Min/Max, so the frame doesn't go white. The step card has its own curve.
- Works on `MG36212_1_0091_narrowfov_000000.h5` too (histogram ~70 ms, tone adds ~5 ms per tile).

## 12. Python 3.10 (Jetson-like)

```bash
python3 -m venv /tmp/pd310 && /tmp/pd310/bin/pip install ./packages/pd-xray-core ./packages/pd-xray-storage "./packages/pd-xray-io[all]" ./packages/pd-xray-processing ./packages/pd-xray-server fastapi==0.142.2 uvicorn==0.54.0 websockets==16.1.1 httpx==0.28.1 pytest
/tmp/pd310/bin/pip list | grep -iE "numpy|h5py|tifffile|pillow|scipy|scikit|yaml|fastapi|uvicorn|websockets"
# 1.26.4, 3.16.0, 2025.5.10, 12.3.0, 1.15.3, 0.25.2, 6.0.3, 0.142.2, 0.54.0, 16.1.1
cd /tmp && /tmp/pd310/bin/python -m pytest -q --import-mode=importlib ~/pd_xray/packages   # 69 passed
```

Cleanup: `rm -rf /tmp/pd_xray_manual /tmp/pd310`.
