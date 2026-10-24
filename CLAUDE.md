# CLAUDE.md

## Project

`pd_xray`: X-ray radiography (later tomography) toolkit, being rewritten from scratch as:

1. Small, independently pip-installable Python libraries (storage, formats, processing). Other projects import them
   directly, so each must be light and work without the web app.
2. A web app on top: runs on localhost, or on a shared internal lab server (≤ 5 users, Docker, GPU).

Workflow: browse/copy files (local, NFS, Synology SFTP, S3) → open dataset → scrub frames with a slider → apply
ImageJ-style processing from a side panel → save as pipeline → run on the whole dataset in the background → write to a
new location.

Plan, current step and decisions: `WORK_PLAN.md`. Read it before starting work.

## Priorities

1. Reliability and longevity. Use mature dependencies and pin versions.
2. Speed. Files can be 100 GB+. The UI never freezes and always shows backend activity.
3. Readability. Mainly maintained by AI.

## Working rules

- Git: the user does all git operations. Only read-only git commands are allowed.
- One `WORK_PLAN.md` step at a time. Don't build ahead.
- Ask before adding dependencies, changing the layout or public APIs, or making architectural decisions. Record
  decisions in `WORK_PLAN.md`.
- Legacy code (`src/pd_xray/`, `tests/`, `notebooks/`, `download_timesteps.py`, root `pyproject.toml`, `uv.lock`) is
  reference only. Don't modify it. Port useful parts after re-checking them.
- Tests: focused, not exhaustive. Cover easy-to-get-wrong logic (readers, slicing, dtypes, numerical output).
- No large files in the repo. Tests use small synthetic data. Real data is in untracked `data/`.
- **Markdown files are short and direct**: they are for the AI, to keep token use low. No lengthy explanations.
- **Simple code.** Use existing libraries (NumPy, SciPy, scikit-image, tifffile, h5py…) instead of reimplementing.
  Don't write duplicate functions or add needless abstraction. Never at the cost of functionality or speed.

## Layout

```
packages/
  pd-xray-core/        pd_xray.core: types, errors, progress/cancel protocol
  pd-xray-storage/     pd_xray.storage: local/NFS, SFTP, S3, transfers, CLI
  pd-xray-io/          pd_xray.io: HDF5, TIFF, PNG, JPG, EDF, VTI readers/writers, lazy datasets
  pd-xray-processing/  pd_xray.processing: operations, pipelines, optional GPU
  pd-xray-server/      pd_xray.server: FastAPI, jobs, tiles, `pd-xray serve`
frontend/              TypeScript UI, built assets bundled into pd-xray-server
docker/
environment.yml, environment-dev.yml
```

- PEP 420 namespace: no `src/pd_xray/__init__.py` in any package. Don't install the legacy root package alongside.
- Dependency direction: core ← storage, io, processing ← server. `processing` works on arrays only. Libraries never
  import `server`.
- `core` depends only on NumPy and the standard library.
- Backend-specific dependencies are optional extras (`[sftp]` paramiko, `[s3]` boto3, `[gpu]` CuPy). Import them
  lazily, with a clear error if missing.

## Engineering rules

- Never assume data fits in memory. Use lazy, sliceable datasets and frame/chunk-wise processing.
- Keep the native dtype (usually uint16). Conversions are explicit.
- For remote data, prefer range requests/streaming with bounded buffers.
- Display settings (contrast, colour map, downsampling) never modify data.
- Named axes, e.g. `(T, Y, X)`. Structure is auto-discovered or user-supplied, never silently guessed.
- Long library operations take an optional progress callback and cancel token (from `core`). Libraries never print or
  use tqdm.
- Server: long work runs as background jobs pushing progress to the UI status bar. Never block the event loop.
- GPU is optional. The CPU path is the reference, and GPU results must match it within tolerance.
- Libraries use `logging.getLogger(__name__)` and never configure handlers.
- No global mutable state.
- Never write credentials to configs, logs or error messages.
- Public API is defined in each subpackage's `__all__`. Raise specific exceptions with actionable messages.

## Style

Code must run on Python 3.10–3.12 (`requires-python = ">=3.10"`; ruff `target-version` and mypy `python_version`
set to 3.10). No 3.11+ features: `typing.Self`, `StrEnum`, `tomllib`, `except*`, PEP 695 generics. Type hints, ruff +
mypy clean, 120-char lines. British spelling (`normalise`, `colour`). Google-style
docstrings stating array axes and dtype.

## Environment

- Conda env, packages installed with pip into it. No `uv`.
- NumPy matches the Jetson Orin AGX (linux-aarch64): `environment*.yml` pin `numpy==1.26.4`, package `pyproject.toml`
  files use `numpy>=1.26.4,<2` (so importing projects aren't forced to one exact version). Every dependency must support NumPy
  1.26.4, Python 3.10 and linux-aarch64. The Jetson uses system Python 3.10.12, and dev envs use 3.12.

```bash
conda env create -f environment-dev.yml && conda activate pd-xray-dev
pip install -e packages/pd-xray-core -e packages/pd-xray-storage[all] ...
ruff check packages && ruff format --check packages && mypy packages && pytest packages
```

Commands are set up in step 0.1. Update this section if they change.
