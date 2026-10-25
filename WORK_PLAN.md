# WORK_PLAN.md

One step at a time, in order. A step is done when it meets "Done" and the user has reviewed it.
`[ ]` todo · `[~]` in progress · `[x]` done. Resolve a step's open decisions before starting it.

**Current:** Step 1.4 done, awaiting user review. Next: 1.5. Before starting 1.5, ask the user about D4, D5 and D7.
Manual test guide: `docs/manual_testing.md` (update it per step).

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
  - `data/24_I20231030_1/0042/0042__NNNNNN_{dar,img}.tif`: a sequence of 25 single-frame TIFFs (9 dark, 16 image),
    uint16 with 12-bit values, 2101×5120, uncompressed. Orientation tag = 3 (180°). About 7 ms per file.
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

### [~] 1.4 processing
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

### [ ] 1.5 server
- FastAPI + `pd-xray serve` (host, port, allowed roots, open browser).
- APIs: browse files (within allowed roots), open/inspect datasets, frame tiles (cached), jobs (progress events,
  cancel) (D4, D5, D7).
- Done: endpoints work, and a job reports progress and cancels without blocking other requests.

### [ ] 1.6 frontend: browser, viewer, status bar
- Framework (D6). Layout: file browser, viewer, side panel, status bar.
- Viewer: pan/zoom, frame slider with keys, display contrast/colour map.
- Status bar: loading, jobs with progress/ETA/cancel, errors.
- Done: TIFF and HDF5 scrub smoothly. About 200 ms per 2k×2k uint16 frame (revise after measuring).

### [ ] 1.7 processing panel
- Ops from the registry, auto-built controls. Add/remove/reorder/toggle steps.
- Live preview on the current frame. Original/processed toggle.
- Done: preview updates quickly and never blocks the slider.

### [ ] 1.8 pipeline save and run
- Save/load the pipeline. Apply it to the dataset: pick output and format, run as a background job, cancel (D8).
- Output records provenance (source, pipeline, version, date).
- Done: a multi-GB dataset processes from the UI, and the saved pipeline reproduces the result from plain Python.

### [ ] 1.9 packaging
- Frontend bundled into `pd-xray-server`. User README. Clean install on a second machine (and the Jetson).
- Done: a colleague installs and uses it from the README alone.

## Phase 2: Remote storage and copying
- `StorageBackend`: Local (also NFS), SFTP, S3. Port from legacy.
- Transfers between backends: progress, cancel, resume, integrity check. `pd-xray-storage` CLI.
- io reads remote files lazily (range requests), falling back to download.
- UI: connections (in-memory credentials), remote browser, copy dialog.

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
| D4 | Progress transport | 1.5 | WebSocket |
| D5 | Tiles: 8-bit rendered vs raw 16-bit + WebGL | 1.5 | 8-bit first, API open to 16-bit |
| D6 | Frontend framework and viewer | 1.6 | React + OpenSeadragon |
| D7 | Job workers: threads vs processes | 1.5 | Threads first |
| D8 | Partial output on cancel | 1.8 | Delete it |
| D9 | How inter-package deps resolve for outside installs (`pd-xray-storage` alone can't find `pd-xray-core`) | 1.9 | Private index or PyPI vs git direct references |

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
