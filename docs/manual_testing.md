# Manual testing guide: `pd_xray.core`, `io`, `processing`, `storage`, `server`, web UI (steps 1.1–1.7a, 2.1–2.2)

Each section is a block to paste into Python, with the expected result next to it. Run them from the repo root.

## 0. Setup

```bash
source ~/miniconda3/bin/activate pd-xray-dev     # conda isn't on PATH in plain shells
cd ~/pd_xray
ruff check packages && ruff format --check packages && mypy && pytest   # expect: all pass, 86 tests
cd frontend && npm ci && npm run format:check && npm test && npm run build && cd ..   # 9 tests; build goes into pd-xray-server
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

## 12. Remote storage: connect and browse (2.1)

Needs your NAS (SFTP enabled: Synology Control Panel › File Services › FTP › SFTP) and/or an S3 bucket. Without them,
use local stand-ins: `moto_server -p 5055` (fake S3, any keys) and any SSH server.

```python
from pd_xray.storage import SFTPStorage, S3Storage, LocalStorage, HostKeyError, AuthenticationError
nas = SFTPStorage("NAS_HOST", "USER", password="…", root="/volume1/SHARE")   # or key_file="~/.ssh/id_ed25519"
nas.check()          # unknown host: HostKeyError with the fingerprint. Check it, then add host_key="SHA256:…"
[(e.name, e.is_dir, e.size) for e in nas.list()]          # the share's top folder
f = nas.open("SOME/FILE.tif"); f.seek(1000); f.read(16)   # reads only that range
SFTPStorage("NAS_HOST", "USER", password="wrong", host_key="SHA256:…").check()   # AuthenticationError, no password in it
SFTPStorage("10.255.255.1", "x", password="x").check()    # StorageConnectionError after 15 s

s3 = S3Storage("BUCKET", access_key="…", secret_key="…", endpoint_url=None)   # or profile="…"; endpoint for non-AWS
s3.check(); s3.list(); s3.list("some/folder")
S3Storage("no-such-bucket-xyz", access_key="…", secret_key="…").check()     # FileNotFoundError: bucket doesn't exist
nas.close(); s3.close()
```

Web UI (`pd-xray serve --root data --port 8765 --no-browser`):

- File browser top level: "+ Add connection (NAS or S3)". NAS: host, user, folder, password (or private key file).
  Connect: first time an amber note shows the server's key fingerprint → "Trust this key and connect".
- The connection appears as 🔌 name at the top level; browse folders, ↑ goes back up and finally to the top. Remote
  files are listed with sizes but are greyed out (opening comes in 2.4). × disconnects.
- Wrong password → "refused the login"; unreachable host → "timed out" after 15 s. The dialog stays open to fix it.
- Reopen the dialog: "Saved" lists earlier connections without passwords; click one to fill the form.
- Another browser (or a private window) sees none of your connections. Restarting the server drops them all.

## 13. Download: remote → local (2.2)

Use a folder of a few GB on the NAS/S3 (REMOTE below) and an empty local folder.

```python
from pd_xray.storage import copy, LocalStorage
from pd_xray.core import CancelToken, OperationCancelledError
dst = LocalStorage(OUT)
t = time.time(); print(copy(nas, "REMOTE", dst, "dl", progress=None), time.time() - t)   # CopyResult(copied=N, …)
copy(nas, "REMOTE", dst, "dl")                    # skipped=N: identical files are not fetched again
stop = CancelToken()
def cut(p):
    if p.current > 500e6: stop.cancel()
copy(nas, "REMOTE", dst, "dl2", progress=cut, cancel=stop)   # OperationCancelledError; hidden *.pd-xray-part kept
copy(nas, "REMOTE", dst, "dl2", verify=True)                 # resumes (log: "Resuming …"), checksummed=N
copy(s3, "FOLDER", dst, "s3dl")                              # checksummed > 0 for single-part S3 uploads (MD5)
```

```bash
pd-xray-storage ls sftp://USER@NAS_HOST/volume1/SHARE --password          # unknown host: prints --host-key to add
pd-xray-storage copy sftp://USER@NAS_HOST/volume1/SHARE/REMOTE /tmp/pd_xray_manual --password --host-key SHA256:…
# progress line with MB/s; Ctrl-C then the same command resumes. S3: s3://BUCKET/KEY (--profile / --endpoint-url)
```

Web UI: in a connection, ⧉ next to a file/folder (or in the path bar for the folder shown) → pick a folder inside the
allowed folders → Copy here. Status bar: "1.2 GB/3.4 GB · MB/s · time left", × cancels; copying again resumes.
Hover a finished job for what it copied. Change a copied file, copy again → error asks for "Overwrite different
files". The dialog reopens in the last folder used.

Measured (local OpenSSH, idle machine): 3 GB at 355–370 MB/s (OpenSSH `sftp` 678 MB/s); more parallel chunks or
connections don't help (Python-bound). Moto's fake S3 is slow (17–19 MB/s, also for boto3 itself): measure real S3.

## 14. Copy anywhere: upload, NAS ↔ S3 (2.3)

Continue from 13 (`nas`, `s3`, `dst` open; LOCAL a local folder of a few GB, e.g. `data/24_I20231030_1/0042`).

```python
src = LocalStorage("LOCAL_PARENT")
copy(src, "LOCAL", nas, "pd_xray_test/up")              # upload; again -> skipped=N
copy(nas, "pd_xray_test/up", s3, "pd_xray_test/up")     # NAS -> S3 (streams, no temp file); again -> skipped=N
copy(s3, "pd_xray_test/up", dst, "back", verify=True)   # S3 -> local, checksummed=N
# compare: `diff -r LOCAL OUT/back` prints nothing; `ls -l --time-style=+%s` shows the same dates
stop = CancelToken()
copy(src, "BIG_FILE", s3, "pd_xray_test/big", progress=cut, cancel=stop)   # cancel mid-way (file > 500 MB)
s3.list("pd_xray_test")                                 # hidden ".big.<size>-<time>.pd-xray-part" marker
copy(src, "BIG_FILE", s3, "pd_xray_test/big")          # resumes (log: "Resuming upload … after N parts")
copy(nas, "pd_xray_test", nas, "pd_xray_test/inner")    # StorageError: can't copy into itself
```

```bash
pd-xray-storage copy LOCAL sftp://USER@NAS_HOST/volume1/SHARE/pd_xray_test/ --password   # trailing /: inside it
```

Web UI: ⧉ is on every row, local ones too. The dialog lists allowed folders and 🔌 connections; pick a folder in a
connection → Copy here (upload). Also try NAS → S3 and within one NAS. Hover the finished job: "into nas: /…".
Check in the Synology/S3 console: files there with the right sizes; no `.pd-xray-part` left after a finished copy.
Clean up `pd_xray_test` on the NAS and S3 afterwards.

Measured (local OpenSSH + moto server, idle machine, 3 GB slice of the 99 GB file): local → NAS 330–357 MB/s, NAS →
local 400 MB/s, cancel at 1 GB → resume sent only the remaining 2.2 GB, byte-identical. 0042 folder (26 files,
559 MB): local → NAS 271 MB/s, NAS → S3 141 MB/s, S3 → local 324 MB/s (verify), NAS → NAS 166 MB/s; identical with
dates kept. Moto upload 130–173 MB/s (moto-bound): measure real S3.

## 15. Save and run pipelines (1.8)

Web UI, `pd-xray serve --root data`. Open `MG36212_1_0091_narrowfov_000000.h5`, add Median filter (3) and Crop (top
100, height 2000).

- Save… → opens in `data/`; name `test_pipeline` → `test_pipeline.yaml` (a name with "/" can't be saved: open the
  folder instead). Save again with the same name: asks to replace. × both steps, Load… → the file → both steps back.
- Run… → Format TIFF, Values uint16 (Range = the processed Min/Max), Frames 0, 999, 1: "1000 of 5616 frames · output
  about 16.4 GB". Run. Status bar: `n/1000 · ~20/s · … left`; scrubbing stays smooth meanwhile. Done in ~50 s; hover
  it: "Wrote 1000 frames … Pipeline: ….pipeline.yaml".
- Run… again, same name → asks to replace → Run → × after a few seconds: cancelled; the previous output is still
  there, complete, and no `.….pd-xray-writing` folder is left (`ls -a data`).
- Format PNG: float32 isn't offered; JPEG: uint8 only.
- `0042` sequence: flat field with `dar` and three flats + median 3 → Save…: the file says `dark: dar`,
  `flat: [flat/…]` (relative). Run… into `0042_corrected.h5` (float32): 14 frames, a few seconds, values ~0–3.

Reproduce a run in Python (docs/processing.md, "Reproducing a web app run"), then compare:

```python
with open_dataset("data/OUT.tif") as a, open_dataset("data/again.tif") as b:
    print(all(np.array_equal(a[i], b[i]) for i in range(a.shape[0])))     # True
    print(a.metadata["pd_xray_run"]["pipeline"], a.metadata["value_scaling"])
```

Delete the test outputs and `test_pipeline.yaml` afterwards.

Measured (idle machine, 112 cores): median 3 + crop on 2304 × 4096 frames, 1000 frames (19 GB in, 16.4 GB uint16
out) in 45–50 s from the UI, 48 s from Python, identical. Threads: 1 → 0.9 frames/s, 16 → 12.8, 32 → 17.4 (Python,
300 frames). New-frame tiles during a run 71 ms (54 ms idle).

## 16. Open datasets on the NAS (2.4a)

Web UI with your real NAS connected (§12). Put (or find) on the NAS a large HDF5 file and a TIFF sequence with
`dar/` and `flat/` folders, e.g. copies of `MG36212_1_0091_narrowfov_000000.h5` and `24_I20231030_1/0042` (⧉).

- In the NAS listing, image files are no longer greyed out (tooltip "Read from … as needed"), and sequences show
  as `▦`. Click the HDF5 file: it opens in about a second; the Dataset panel shows 🔌 and the `sftp://…` location
  on hover. Step with →: note the time per frame (local test server: 0.1–0.4 s; the next frame is prefetched).
  Watch the NAS's network use in DSM: only about one frame per step, not the whole file.
- Open the sequence. Flat/dark correction → Choose… opens in the scan's folder on the NAS → `dar` → Use this
  folder; flat → `flat` → Select shown → Use 3. Add median 3. Save… (opens in the server's folders): the file says
  `dark: dar`, `flat: [flat/…]`. Run… into a server folder; hover the finished job.
- Its `.pipeline.yaml` has `storage: sftp://USER@NAS…` (no password) and `source: /…`. Reproduce in Python:

```python
nas = SFTPStorage("NAS_HOST", "USER", password=getpass.getpass(), root="/volume1/SHARE")
run = yaml.safe_load(open("OUT.tif.pipeline.yaml"))["run"]
pipeline = Pipeline.load("OUT.tif.pipeline.yaml")
with open_sequence(run["source"], run["pattern"], storage=nas) as ds:
    resolved = pipeline.resolve(lambda ref: average_frames(reference_frames(ref, run["source"], nas)))
    convert(map_frames(ds, resolved, workers=8), "again.tif")      # identical to OUT.tif
```

- × Disconnect at the top level while a NAS dataset is open: the panel shows "No dataset open.". Reopening a NAS
  file after a server restart says to connect again.

Measured (local OpenSSH stand-in, machine fully loaded by another job): 99 GB HDF5 inspect 0.38 s, open 0.6 s,
new frame 0.3–0.6 s (whole-frame tile); arrow steps in the UI 81–330 ms; 0042 sequence frame ~0.36 s; flat field
with 12 references from the NAS: first preview 6.4 s (loads them), then 0.1 s per step; 14-frame run 3–11 s. UI run
= Python over SFTP = Python on local copies (identical).

## 17. Packaging and a clean install (1.9)

Release (you): bump nothing for 0.1.0; build wheels as in [development](development.md#releases), tag `v0.1.0`,
create the GitHub release and attach the five `.whl` files.

On a second machine (a colleague's, from the README alone): Quick start with conda → the app opens; open a file,
add a step, Run…. `pip check` only reports the known imagecodecs/NumPy line.

On the Jetson (system Python 3.10, aarch64), in a fresh venv: the pinned `pip install …` line from
[installation](installation.md#python-310--jetson-no-conda), ending with `--find-links $REL pd-xray-server`
instead of the clone paths. `pd-xray serve --host 0.0.0.0 --no-browser --root …`, open it from your computer,
scrub a file, run a pipeline. Library only, in another venv: `pip install --find-links $REL "pd-xray-io[hdf5]"`
installs just pd-xray-core and pd-xray-io (+ h5py, hdf5plugin, NumPy 1.26.4); opening a TIFF then says
`pip install "pd-xray-io[tiff]"`.

Measured here (x86-64): wheels 4–207 KB (the server's holds the built UI); fresh conda env from a clean copy 34 s,
smoke test OK; Python 3.10 venvs from the wheels: library-only and full app (pinned versions), `pip check` clean,
app serves the UI, tiles and previews. pip reads GitHub `expanded_assets` pages as `--find-links` (checked on
another project's release).

## 18. Python 3.10 (Jetson-like)

```bash
python3 -m venv /tmp/pd310 && /tmp/pd310/bin/pip install ./packages/pd-xray-core "./packages/pd-xray-storage[all]" "./packages/pd-xray-io[all]" ./packages/pd-xray-processing ./packages/pd-xray-server fastapi==0.142.2 uvicorn==0.54.0 websockets==16.1.1 httpx==0.28.1 moto==5.2.4 pytest
/tmp/pd310/bin/pip list | grep -iE "numpy|h5py|tifffile|pillow|scipy|scikit|yaml|fastapi|uvicorn|websockets"
# 1.26.4, 3.16.0, 2025.5.10, 12.3.0, 1.15.3, 0.25.2, 6.0.3, 0.142.2, 0.54.0, 16.1.1
cd /tmp && /tmp/pd310/bin/python -m pytest -q --import-mode=importlib ~/pd_xray/packages   # 97 passed
```

Cleanup: `rm -rf /tmp/pd_xray_manual /tmp/pd310`.
