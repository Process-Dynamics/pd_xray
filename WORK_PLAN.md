# WORK_PLAN.md

One step at a time, in order. A step is done when it meets "Done" and the user has reviewed it.
`[ ]` todo · `[~]` in progress · `[x]` done. Resolve a step's open decisions before starting it.

**Current:** Step 0.1, waiting for plan approval.

## Phase 0: Foundations

### [ ] 0.1 Skeleton and environments
- `packages/*` with `pyproject.toml` (hatchling, src layout, namespace), README and `tests/`.
- `environment.yml`, `environment-dev.yml`. Root `ruff.toml`, `mypy.ini`, `pytest.ini` (separate files to avoid the
  legacy `pyproject.toml`). Update `.gitignore`.
- Done: fresh env installs. All five `pd_xray.*` import. ruff/mypy/pytest pass.

### [ ] 0.2 Sample data
- User provides local samples in `data/`: multi-page TIFF, PNG/JPG, large HDF5. Note their paths here.
- `scripts/` generator for synthetic test files.

## Phase 1: Local files in the web UI

Scope: local/NFS files. TIFF (multi-page), PNG, JPG, HDF5. Axes `(Y, X)`, `(T, Y, X)`.

### [ ] 1.1 core
- Exceptions, `ProgressCallback`, thread-safe `CancelToken`, named axes.
- Done: API reviewed. Small tests.

### [ ] 1.2 io: Dataset + TIFF
- `Dataset` protocol (`shape`, `dtype`, `axes`, `metadata`, lazy `__getitem__`, close/context manager).
- `open_dataset()` picks the reader.
- tifffile reader: page-wise, memmap when possible, native dtype. User can override the axes.
- Done: a large TIFF opens in < 1 s, and reading one frame reads only that frame.

### [ ] 1.3 io: PNG/JPG, HDF5, writers
- PNG/JPG reader.
- h5py reader: `inspect()` lists datasets (shape/dtype/chunks), the user picks one. Use hdf5plugin.
- Frame-wise writers: (Big)TIFF and HDF5.
- Done: a 100 GB HDF5 inspects and reads a frame quickly. Writer round-trip tests.

### [ ] 1.4 processing
- Operation registry: name plus typed params (default/range/description, so the UI auto-builds controls) and a frame
  function.
- First ops (D3): invert, gaussian, median, normalise, clip, threshold, crop, rotate/flip, rolling ball.
- Versioned `Pipeline` file (D2). `run_pipeline(dataset, pipeline, writer, progress, cancel)` with constant memory.
- Done: numerical test per op. Pipeline file round-trips.

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
- EDF, VTI. Image sequences (folders of images) as one dataset.
- Flat/dark correction, histogram, line profile, ROI measurements, frame arithmetic.

## Backlog
Tomography and 3D views, Zarr/remote-friendly layout, segmentation/dendrite analysis, CIL, auth.

## Open decisions

| # | Decision | Step | Leaning |
|---|----------|------|---------|
| D2 | Pipeline file format | 1.4 | YAML |
| D3 | First ops list | 1.4 | User confirms the list in 1.4 |
| D4 | Progress transport | 1.5 | WebSocket |
| D5 | Tiles: 8-bit rendered vs raw 16-bit + WebGL | 1.5 | 8-bit first, API open to 16-bit |
| D6 | Frontend framework and viewer | 1.6 | React + OpenSeadragon |
| D7 | Job workers: threads vs processes | 1.5 | Threads first |
| D8 | Partial output on cancel | 1.8 | Delete it |

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
- 2026-10-24: Short, direct markdown. Simple code that reuses existing libraries.
