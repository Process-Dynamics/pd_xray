# WORK_PLAN.md

One step at a time, in order. A step is done when it meets "Done" and the user has reviewed it.
`[ ]` todo · `[~]` in progress · `[x]` done. Resolve a step's open decisions before starting it.

**Current:** 2.4a built, awaiting review (user checks the real NAS, manual guide §16). Then 2.4b (S3; user's bucket access pending), then 1.9. Real S3 check
of 2.3 deferred too; run manual guide §14 when fixed.
Phase 2 (2.1–2.3) ran before 1.8; 2.4 runs before 1.9.
Open items: Starlette TestClient warns `httpx` → `httpx2` (harmless); threshold's default value 0 keeps everything.
Manual test guide: `docs/manual_testing.md` (update it per step). User docs: `README.md`, `docs/*.md` (update with
features).

## Phase 0: Foundations

### [x] 0.1 Skeleton and environments
- `packages/*` with `pyproject.toml` (hatchling, src layout, namespace), README and `tests/`.
- `environment.yml`, `environment-dev.yml`. Root `ruff.toml`, `mypy.ini`, `pytest.ini` (separate files to avoid the
  legacy `pyproject.toml`). Update `.gitignore`.
- Done: fresh env installs. All five `pd_xray.*` import. ruff/mypy/pytest pass.

### [x] 0.2 Sample data
- User provides local samples in `data/`: multi-page TIFF, PNG/JPG, large HDF5. Note their paths here.
- Synthetic generator deferred until large-file benchmarks need it.
- Samples:
  - `data/pr_110_proj_0040.h5` (104 MB, simulated projections). Datasets `projections` and `projections_E` (artificial
    contrast). Each is `(500, 512, 512)` float32 `(angle, Y, X)`, chunks `(1, 512, 512)`, gzip-4, so reading one
    frame takes about 4 ms. String attributes: `angular_range_deg`, `n_angles`, `units`, `voxel_size_um`, ….
  - `data/MG36212_1_0091_narrowfov_000000.h5` (99 GB). Dataset `narrowfov`, uint16, 2304×4096, chunks `(1, 2304, 4096)`,
    blosc (stored uncompressed). **Broken extent:** the shape claims about 8.9e18 frames, but there are 5598 chunks
    written up to frame 5615. 18 frames are missing (in a period-910 pattern) and read as fill value 0. Listing takes
    27 ms, and a frame read takes 10–24 ms.
  - `data/24_I20231030_1/0042/0042__NNNNNN_img.tif`: a sequence of 13 single-frame TIFFs (darks/flats now in `dar/`,
    `flat/` subfolders), uint16 with 12-bit values, 2101×5120, uncompressed. Orientation tag = 3 (180°). About 7 ms per
    file.
  - `data/0042_img_multipage.tif`: the 16 `*_img.tif` frames as one multi-page TIFF (made with `convert`).
  - `data/labelled_data/imagesTr/*.tif`: single 2D TIFFs (8 MB), plus label TIFFs. Legacy segmentation data.

## Phase 1: Local files in the web UI

Scope: local/NFS files. TIFF (multi-page), PNG, JPG, HDF5. Axes `(Y, X)`, `(T, Y, X)`.

### [x] 1.1 core
- Exceptions, `ProgressCallback`, thread-safe `CancelToken`, named axes.
- The frame axis isn't always time (the sample is `(angle, y, x)`), so the slider works on any non-spatial axis.
- Done: API reviewed. Small tests.

### [x] 1.2 io: Dataset + HDF5
- `Dataset` base (`shape`, `dtype`, `axes`, `metadata`, `warnings`, lazy `__getitem__`, context manager),
  `ArrayDataset`, `split_key`. `open_dataset()` picks the reader by extension (lazy import).
- h5py reader (extra `[hdf5]`): `inspect_hdf5()` lists datasets, the user picks one. Axes come from dimension labels
  or are user-given (`frame` if unknown). A broken extent on an unlimited axis is trimmed to the written chunks, with
  missing frames in `metadata["missing_frames"]` and `warnings`.
- Done: 99 GB file inspects in 27 ms, opens in 7 ms, reads a frame in 10–24 ms.

### [x] 1.3 io: TIFF, PNG/JPG, sequences, writers, convert
- tifffile reader (extra `[tiff]`): first series, memmapped if contiguous, otherwise page-wise. Orientation tag is
  recorded (`tiff_orientation`), not applied.
- `open_sequence(dir, pattern)` or a list of files: single-frame TIFF/PNG/JPG as one dataset, in natural order (e.g.
  `*_img.tif` and `*_dar.tif` separately).
- Pillow reader (extra `[png]`) for PNG/JPG.
- Writers (`write_frames`, pull frames from an iterable): multi-page (Big)TIFF, HDF5 (per-frame chunks, dimension labels, attrs), PNG/JPG (one file
  per frame, JSON metadata).
- `convert()`: exact when the format allows. Otherwise linear scaling with one global range, recorded in
  `metadata["value_scaling"]`. Progress and cancel supported.
- Done: sequence→multi-page TIFF exact, memmapped (6 ms/frame). uint16→PNG exact. float32→PNG within half a step.
  Py3.10 tested.

### [x] 1.4 processing
- Deps: scipy, scikit-image, PyYAML (env pins 1.15.2 / 0.25.2 / 6.0.3; last py3.10 lines).
- `OPERATIONS` registry: `Operation` (name, title, description, typed `Param`s, frame function). Frames are float32
  (pipelines cast on input); ops return new float32 frames.
- Ops: flat_field, invert, gaussian, median, normalise, clip, threshold, crop, rotate (90° steps), flip, rolling_ball
  (ImageJ-style shrink for radius > 10). `average_frames` (mean/median) builds dark/flat references.
- `Pipeline` of `Step`s, YAML file `pd_xray_pipeline: 1`. Reference params (dark/flat) are strings in the file,
  `resolve(load)` swaps in arrays. `apply_pipeline(pipeline, frames, …)` yields frames lazily; the caller writes them
  with `io.write_frames` (processing doesn't import io).
- Measured (machine at full load): median 3 ≈ 3 s, gaussian ≈ 0.7 s, rolling ball r=10 ≈ 15 s, r=50 ≈ 2.6 s per
  2304×4096 frame. Speed-ups (threads over frames in 1.8, GPU in Phase 3) later.
- Done: numerical test per op. Pipeline file round-trips. Real-data pipeline matches a manual computation.

### [x] 1.5 server
- FastAPI + `pd-xray serve` (host, port, allowed roots, open browser).
- APIs: browse files (within allowed roots), open/inspect datasets, frame tiles (cached), jobs (progress events,
  cancel) (D4, D5, D7).
- Done: endpoints work, and a job reports progress and cancels without blocking other requests.
- Built: `/api/files` (roots only, symlink/`..` escapes refused), `/api/inspect`, `/api/datasets` (file, HDF5 dataset,
  folder + pattern), `…/frames/{i}/stats`, `…/frames/{i}/tiles/{level}/{col}_{row}.{png|jpeg}?vmin&vmax`, jobs
  (`value-range` job, cancel), WebSocket `/api/events`. One byte-bounded LRU cache (frames, levels, tiles), each entry
  computed once under concurrency. Level n = 2×2 mean of level n−1. Grey only; colour maps in 1.6.
- Measured on the 99 GB file (512 px tiles): cold frame, whole-frame tile 20–50 ms; level-1/0 tiles 3–8 ms after; new
  contrast 2 ms. Cancel stops within ~5 frames. Tiles stay < 0.1 s while two jobs run. Py3.10 tested.

### [x] 1.6 frontend: browser, viewer, status bar
- Framework (D6). Layout: file browser, viewer, side panel, status bar.
- Viewer: pan/zoom, frame slider with keys, display contrast/colour map.
- Status bar: loading, jobs with progress/ETA/cancel, errors.
- Done: TIFF and HDF5 scrub smoothly. About 200 ms per 2k×2k uint16 frame (revise after measuring).
- Built: frames load hidden and swap in when complete (screen never mixes frames). Latest-wins loading, next-frame
  prefetch while stepping. Auto-reopen after a server restart. Sequences auto-discovered in folder listings. Colour maps
  on tiles (`cmap`). Built UI served at `/`.
- Measured in headless Chrome, 99 GB file (2304×4096): open 0.19 s; arrow step 67 ms (prefetched; ~115 ms cold);
  holding a key ~10 frames/s; same during a job. 2101×5120 TIFF sequence 67 ms; 512² 23 frames/s.
- Later if needed: playback button, pixel value under the cursor, keeping zoom on reopen.

### [x] 1.7 processing panel
- Ops from the registry, auto-built controls. Add/remove/reorder/toggle steps.
- Live preview on the current frame. Original/processed toggle.
- Done: preview updates quickly and never blocks the slider.
- Built: `GET /api/operations`, `POST …/frames/{i}/preview` (exact, full resolution; a newer preview from the same tab
  cancels the older between steps), `pipeline=` on tiles/stats. Panel: add/remove/reorder/toggle, controls from the
  registry, per-step timings, Original/Processed toggle (P), separate contrast for processed. Reference picker: a folder
  or chosen files. `Pipeline.__call__(frame, progress, cancel)`; reference params may be a list of paths.
- Measured, 2101×5120 sequence: flat field (9 darks folder + 3 flat files) 0.25 s first, 0.05 s per frame; + median 3
  1.0 s, median 5 3.1 s. Slider answers in < 10 ms while processing; fast stepping processes only the last frame.

### [x] 1.7a tone and histogram (user request)
- Tone: exposure, contrast, whites, blacks, highlights, shadows (per pixel), tone curve. One shared function, used as
  a display setting (tiles only) and as a `tone` pipeline step (black..white → 0..1 → adjust → back to black..white).
- Histogram of the frame on screen (linear/log). Drag black/white handles = display Min/Max; the tone curve is drawn
  over it and editable there. "Add as step" turns the display tone into a pipeline step.
- Done: adjustments and curve edits update the view quickly; the step matches the display.
- Built: `tone_map()`, `TONE_PARAMS`, `tone` op and `curve` param kind (processing); `tone=` on tiles,
  `GET …/histogram`, `GET /api/tone-curve` (curves drawn from the same function). UI: histogram with Min/Max handles
  and curve editing, histogram Reset (straight curve + auto Min/Max), tone sliders + Reset, "Add as step", curve
  editor in the Tone step card.
- Measured: histogram 70–85 ms per full frame (5 ms cached); tone +5 ms per tile; tone step 0.15–0.35 s per 10 MP
  frame; slider or curve change to new tiles ~0.3–0.4 s. Display and step match exactly at full resolution.

### [x] 1.8 pipeline save and run
- Save/load the pipeline. Apply it to the dataset: pick output and format, run as a background job, cancel (D8).
- Output records provenance (source, pipeline, version, date).
- Done: a multi-GB dataset processes from the UI, and the saved pipeline reproduces the result from plain Python.
- Built: io `map_frames` (lazy `MappedDataset`; `iter_frames` reads in order and processes `workers` frames in a
  thread pool; `Dataset.iter_frames` is the hook), `reference_frames`/`reference_files`, `convert(dtype=)` (integer
  scaling runs in the mapped source's threads), `write_frames` stages into a hidden `.<name>.pd-xray-writing/` and
  moves into place on success (cancel/failure leaves nothing; existing file kept until then). `Pipeline.__call__`
  refuses unresolved references. Server: `GET/POST /api/pipelines` (references inside the dataset folder saved
  relative), `POST /api/jobs/run` (dtype, value range, frame slice, overwrite; checks before starting; 32 threads),
  `<output>.pipeline.yaml` + `pd_xray_run` metadata. UI: Load…/Save…/Run… under the steps, `FileDialog`
  (open/save in allowed folders), run options (format, values, range from the processed view, frames, size
  estimate), replace prompts, frames/s in the status bar.
- Measured (idle machine, 112 cores): 99 GB file, median 3 + crop, 1000 frames (19 GB in, 16.4 GB uint16 out)
  45–50 s from the UI; plain-Python rerun from the pipeline file identical. Threads 1/16/32: 0.9/12.8/17.4 frames/s
  (uint16 scaling was serial at ~8 frames/s before moving it into the threads). 0042 flat field with relative
  references: identical from Python. Tiles during a run 71 ms (54 idle). Py3.10 tested.

### [ ] 1.9 packaging
- Frontend bundled into `pd-xray-server`. User README. Clean install on a second machine (and the Jetson).
- Done: a colleague installs and uses it from the README alone.

## Phase 2: Remote storage and copying

Done before 1.8/1.9 (user request). Port legacy `src/pd_xray/data/backends/` after re-checking.

### [x] 2.1 Access: connect and browse
- storage: small backend interface (list, stat, exists, seekable range-read `open`). Local (also NFS), SFTP
  (Synology NAS, extra `[sftp]`), S3 (custom endpoint, extra `[s3]`). Connection settings separate from credentials.
- Clear errors: auth failed, host unreachable, bucket/path missing. Credentials never in logs or errors.
- Python credentials: SSH key/agent or password; AWS profile/env or keys.
- Server: add/test/remove connections, per browser, credentials in memory only; settings (no secrets) remembered.
  File browser shows connections as extra roots.
- UI: "Add connection" dialog (NAS / S3) with Test.
- Done: the user's real NAS and S3 bucket browse from Python and the UI.
- Built: `Storage` (`stat`, `list`, `open`, `exists`, `check`, `url`), `LocalStorage`, `SFTPStorage` (asyncssh on a
  private loop thread, reconnects once), `S3Storage` (boto3, path-style for custom endpoints). Unknown SSH host key:
  `HostKeyError.fingerprint`, trusted via `host_key=`. Server: browser-id cookie, `/api/connections` (POST tests
  first), `…/{id}/files`; server never uses its own SSH keys/agent/AWS credentials. UI: dialog with host-key trust
  prompt and saved settings (localStorage, no secrets); 🔌 connections at the browser's top level, remote files greyed.
- Measured (local OpenSSH, machine at full load): connect + list 0.17–0.34 s; S3 (moto server) connect + list 0.75 s;
  10 MB range reads ≈ raw asyncssh speed (150 MB/s loaded). Not yet tried on the real NAS/S3.

### [x] 2.2 Download: remote → local
- `copy(src, dst, progress, cancel)` for files and folders: streamed, bounded buffers, parallel parts, `.part` then
  rename, resume, integrity check (size; optional checksum). `pd-xray-storage copy` CLI.
- Server: copy job (status bar progress, cancel). UI: select files/folder → "Download to…" a server folder.
- Done: a multi-GB file downloads from NAS and S3, from Python and the UI. Cancel and resume work. Speed measured.
- Built: `Storage.read_range` (thread-safe) and `md5` (S3 single-part ETag); `open` shared in the base.
  `copy(source, path, destination, path, overwrite, verify, workers, progress, cancel)` → `CopyResult`; local
  destinations only (2.3 adds others). 4 files at once, 4 × 8 MB reads in flight per file. Hidden
  `.<name>.<size>-<mtime>.pd-xray-part` files, kept on cancel, resumed. `pd-xray-storage ls|copy` (cp-like folder
  target). Server: `POST /api/jobs/download`, jobs carry `unit` and `rate`. UI: ⤓ per remote row and for the folder
  shown, destination dialog (remembers the last folder), status bar shows bytes, MB/s and time left.
- Measured (local OpenSSH, idle machine): 3 GB at 355–370 MB/s via Python, CLI and UI (OpenSSH `sftp` 678 MB/s).
  More chunks or several SSH connections are not faster (Python-bound). Cancel → resume continued from 940 MB;
  result byte-identical. S3 speed not measured (moto's server caps at ~18 MB/s, boto3's own downloader too).

### [x] 2.3 Copy to any destination
- Same `copy` for every pair: local→NAS/S3, NAS↔S3. Remote→remote streams through the server, no temp file. S3
  multipart upload.
- UI: destination picker over all connections.
- Done: local→NAS, NAS→S3, S3→local work and pass the integrity check.
- Built: `Storage.upload(path, size, modified) -> Upload` (`offset`, `part_path`, `write`, `commit`, `discard`,
  `close`) and `Storage.remove`. Local/SFTP: hidden part file, then rename (SFTP posix-rename, fsync if supported).
  S3: one request up to 8 MB, else multipart (4 parts in flight, Content-MD5 each, part size grows above 80 GB); a
  hidden marker object holds the upload id for resume; removing a marker aborts its upload. `copy` lists each
  destination folder once (skip/conflict/stale parts), stats only same-size files; S3 sources are stat'ed for exact
  dates (`list_dates_exact`). Size checked after commit, `verify` re-reads the copy; bad copies removed. Copying a
  folder into itself is refused. CLI destinations can be remote (trailing `/` = into folder). Server:
  `POST /api/jobs/copy` (`source`/`destination` = {connection or null, path}) replaces `/api/jobs/download`. UI: ⧉ on
  every row and the path bar; `CopyDialog` browses allowed folders and connections, remembers the last destination.
- Measured (local OpenSSH + moto server, idle machine): 3 GB local→NAS 330–357 MB/s, NAS→local 400 MB/s; cancel at
  1 GB → resume sent the remaining 2.2 GB only (NAS and S3), byte-identical. 0042 folder (559 MB) local→NAS→S3→local
  identical with dates kept; NAS→S3 141 MB/s, NAS→NAS 166 MB/s. Moto upload 130–173 MB/s (moto-bound). Reviewed on the real NAS; real S3 later. UI upload,
  cancel, resume, into-itself error and download checked in headless Chrome. Py3.10 tested.

### [ ] 2.4 Open remote datasets lazily (user request, before 1.9)
View, process and run pipelines on files on a NAS or S3 without copying them first. Only the bytes read are fetched.

#### [~] 2.4a NAS (SFTP)
- io: `storage=` on `open_dataset`, `open_sequence`, `inspect_hdf5`, `reference_frames` (duck-typed: an object with
  `open`/`list`, so io doesn't import storage). HDF5, TIFF (multi-page, sequences), PNG/JPG from file objects.
- storage: remote files read through a bounded block cache (small metadata reads don't each cost a round trip;
  large chunk reads go straight through, in parallel parts). Readers thread-safe under concurrent tile requests.
- Server/UI: remote files and sequences open from the browser like local ones. Preview, tiles, histogram, runs work;
  run output goes to server folders, `source` recorded as the URL (no credentials). References on the same
  connection (picker browses it), relative to the dataset's folder when inside it. Clear error if the connection is
  gone (e.g. after a server restart).
- Done: the 99 GB HDF5 and the 0042 sequence on the NAS (local sshd stand-in; the user checks the real NAS) open,
  scrub and process; a run with remote darks/flats reproduces from Python with `storage=`. Frame times measured.
- Built: io `FileStore` protocol + `storage=` on `open_dataset`, `open_sequence`, `inspect_hdf5`, `reference_files`/
  `reference_frames` (and the readers); remote TIFF is page-wise (no memmap). storage `RangeReader` is now a
  thread-safe block cache (256 KB × 64 per open file; reads ≥ 1 MB go straight through) instead of a 1 MB
  `BufferedReader` (which re-read whole ranges on every jump back). Server: `OpenRequest.connection`,
  `/api/inspect?connection=`, remote paths start with "/" (from the connection root), `OpenDataset.folder`/`storage`,
  references on the same connection, run record `storage` (URL) + `source`; remote listings openable with
  sequences; disconnect closes its datasets. UI: remote files/sequences open, 🔌 in the Dataset panel, reference
  picker browses the connection (returns "/…" paths), Save/Run dialogs start at the allowed folders.
- Measured (local OpenSSH stand-in, machine fully loaded by another user): 99 GB file inspect 0.38 s, open 0.6 s,
  new frame 0.3–0.6 s via API, UI arrow steps 81–330 ms; SFTP reads ~70 MB/s under load (parallel parts didn't
  help then; re-check on an idle machine). Remote flat field + median run = Python with `storage=` = Python on the
  local files (identical). Py3.10 tested.

#### [ ] 2.4b S3
- The same over S3 (range GETs), tested with moto. The user checks the real bucket when access works.
- Done: as 2.4a, on S3.

## Phase 3: Hosted deployment
- Docker (conda). GPU ops via CuPy with CPU parity tests.
- Sessions, job queue limits, cache limits. Server config file and log files. Verify on the Jetson.

## Phase 4: More formats and operations
- EDF, VTI.
- Histogram, line profile, ROI measurements, frame arithmetic.

## Backlog
Tomography and 3D views, Zarr/remote-friendly layout, segmentation/dendrite analysis, CIL, auth.

## Open decisions

| # | Decision | Step | Leaning |
|---|----------|------|---------|
| D9 | How inter-package deps resolve for outside installs (`pd-xray-storage` alone can't find `pd-xray-core`) | 1.9 | Private index or PyPI vs git direct references |
| D12 | Test targets: NAS host/folder, S3 provider (AWS/MinIO/Synology/…) and bucket | 2.1 | User provides |

## Decisions log

- 2026-10-23: Same repo, new branch, `pd_xray` name. Legacy code is reference only.
- 2026-10-23: Five pip packages in the `pd_xray` namespace. Heavy dependencies are optional extras.
- 2026-10-23: Conda + pip. `uv` dropped.
- 2026-10-23: FastAPI backend. TypeScript frontend bundled into the server package.
- 2026-10-23: Phase 1 is local 2D/time-series data: TIFF, PNG, JPG, HDF5.
- 2026-10-23: HDF5 datasets are auto-discovered and the user picks one. The user can supply the structure.
- 2026-10-23: EDF/VTI kept (Phase 4). Zarr deferred.
- 2026-10-23: Synology NAS via NFS or SFTP.
- 2026-10-23: Hosted on the internal network, ≤ 5 users, Docker, GPU. Auth deferred.
- 2026-10-23: GPU is optional. CPU is the reference.
- 2026-10-23: British spelling, 120-char lines, ruff + mypy, focused tests.
- 2026-10-24: `numpy==1.26.4` to match the Jetson Orin AGX. Dependencies must support it and aarch64. Python ≤ 3.12.
- 2026-10-24: D1: Jetson uses system Python 3.10.12, so code supports 3.10–3.12. Dev envs use 3.12, and ruff/mypy target
  3.10.
- 2026-10-24: Synthetic data generator skipped for now. Develop against `data/pr_110_proj_0040.h5`.
- 2026-10-24: HDF5 before TIFF (1.2/1.3). Format deps are io extras: `[hdf5]` h5py 3.16.0 + hdf5plugin 7.1.0,
  `[tiff]` tifffile 2025.5.10 (last release supporting py3.10 and NumPy 1.x). Env files pin exact versions.
- 2026-10-24: Broken HDF5 extents are trimmed to the written chunks, and missing frames are reported (not hidden).
- 2026-10-24: File sequences moved into 1.3. Flat/dark correction moved into 1.4, with user-selected dark/flat files.
- 2026-10-24: TIFF orientation tag is recorded in metadata, not applied.
- 2026-10-24: Pillow 12.3.0 (`[png]`, last py3.10 release). Conversions keep values and histograms. Unavoidable
  scaling is linear with one global range, and the mapping is stored.
- 2026-10-24: Short, direct markdown. Simple code that reuses existing libraries.
- 2026-10-25: D3: first ops as listed in 1.4. scipy + scikit-image + PyYAML are required processing deps.
- 2026-10-25: D2: YAML pipeline files. Processing works on frame iterables; callers read/write with io.
- 2026-10-25: Pipelines work in float32. Output dtype/scaling is chosen at write time (1.8).
- 2026-10-25: D4: WebSocket for job progress. D5: server-rendered 8-bit tiles first, API open to raw 16-bit.
  D7: jobs run in a thread pool.
- 2026-10-25: Server deps: fastapi 0.142.2 (conda `fastapi-core`), uvicorn 0.54.0, websockets 16.1.1 (last py3.10
  line). httpx 0.28.1 for tests only.
- 2026-10-25: D6: React + OpenSeadragon, plain React + CSS (no UI kit or state library). Reliability first, then speed.
  Mature lines pinned: Node 24 LTS (conda-forge, dev env only), React 19.2, OpenSeadragon 6.1.1, Vite 7.3, TypeScript
  5.9.3, Vitest 4.1. Built assets ship in pd-xray-server, so users never need Node.
- 2026-10-25: Colour maps are built-in 256-entry tables in the server (grey, inverted grey, viridis, magma, inferno,
  cividis, ImageJ fire). No matplotlib.
- 2026-10-25: Frontend formatter: Prettier 3.9.9, defaults with width 120.
- 2026-10-25: References (dark/flat): a folder (all image files directly in it) or one or more files; every frame found
  is averaged (mean). Relative paths resolve against the dataset's folder.
- 2026-10-25: Preview is exact, at full resolution (user didn't choose; recommended option). Outdated previews are
  cancelled.
- 2026-10-25: D8: a cancelled run deletes its partial output.
- 2026-10-25: D10: saved pipelines store reference paths, so the YAML file reproduces the run. Paths inside the
  dataset folder are saved relative to it (the same file then works on other scans with the same layout), others
  absolute.
- 2026-10-25: 1.7a added before 1.8: tone adjustments both as display setting and pipeline step; step output in the
  black..white range; highlights/shadows per pixel (no local masks); histogram of the frame on screen, editable.
- 2026-10-25: Phase 2 (2.1 access, 2.2 download, 2.3 any-to-any copy) before 1.8/1.9. Remote dataset opening is 2.4.
- 2026-10-25: Credentials per browser, in server memory only (lost on restart). Connection settings without secrets
  are remembered.
- 2026-10-25: Libraries: fastest option. S3: boto3 (parallel multipart transfers). SFTP: decided by benchmark (D11).
- 2026-10-25: D11: asyncssh 2.24.0 for SFTP (local OpenSSH, 1 GB: 359 vs 288 MB/s whole file, 372 vs 131 MB/s 10 MB
  range reads vs paramiko). boto3 1.43.114 for S3. moto 5.2.4 for S3 tests (dev only); SFTP tests use asyncssh's server.
- 2026-10-25: Server connections need their own credentials (password/key, access keys); the server account's SSH
  keys, agent and AWS credentials are never used. Unknown host keys are shown to the user to trust (fingerprint).
- 2026-10-25: 2.2 checks: size always, plus S3's stored MD5 when it is a plain MD5; optional SHA-256 `verify`.
  Identical files (size + mtime) are skipped; different ones stop the copy unless overwrite. UI downloads go anywhere
  inside the allowed folders. Copies keep the source mtime. Part files are kept on cancel (resume), unlike D8.
- 2026-10-25: Built UI (`pd-xray-server/.../static/`) is committed (no longer ignored), so users need no Node; rebuild
  and commit it after frontend changes. README is short (quick start, web app); details in `docs/` (installation,
  web app, io, processing, storage, core, development). Verified from a clean copy with a new conda env.
- 2026-10-25: History scan: no credentials ever committed; one old notebook version (commit 99a1f5b, on main) shows a
  real S3 endpoint/bucket/prefix and a file list (repo is public); user decides on purging. Added data ignore rules
  and nbstripout 0.9.1 (dev env; `.gitattributes`, required filter, install once per clone).
- 2026-10-26: 2.3: S3 uploads store the source date as object metadata `mtime` (rclone-compatible), so repeated copies
  skip identical files. Cancelled S3 uploads keep their parts (marker object) for resuming. One "Copy to…" (⧉) for
  local and remote rows. Tested with local stand-ins (sshd, moto); user checks real NAS/S3 (D12).
- 2026-10-26: 1.8: runs write float32 by default, or uint16/uint8 scaled from a value range (default: the processed
  view's Min/Max; clipped; mapping in `value_scaling`). Pipelines save/load as YAML in the allowed folders; every run
  also writes `<output>.pipeline.yaml` (pipeline + provenance). `pd_xray.io.reference_frames` loads references for
  Python and the server alike. Outputs go to server folders only (copy to NAS/S3 with ⧉ afterwards).
- 2026-10-26: 1.8 reviewed. 2.4 (lazy remote datasets) moved before 1.9, split into 2.4a NAS and 2.4b S3 (S3 later:
  access issue). io takes `storage=` (duck-typed open/list; no storage import). Remote datasets' references live on
  the same connection (relative to the dataset's folder when inside it).
