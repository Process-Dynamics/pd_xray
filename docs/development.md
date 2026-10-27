# Development

Plan and decisions: [`WORK_PLAN.md`](../WORK_PLAN.md). Rules for code and AI assistants: [`CLAUDE.md`](../CLAUDE.md).

## Environment

```bash
conda env create -f environment-dev.yml      # editable installs, tools (ruff, mypy, pytest), Node 24, moto
conda activate pd-xray-dev
nbstripout --install --attributes .gitattributes   # once per clone: see "Notebooks and data" below
```

## Notebooks and data

- **Notebook outputs are never committed.** `.gitattributes` sends every `.ipynb` through nbstripout, which removes
  outputs and execution counts when git stores the file (your local copy keeps them). Outputs can contain bucket
  names, file listings, paths and images. The git filter itself lives in `.git/config`, so run
  `nbstripout --install --attributes .gitattributes` once in every new clone; `nbstripout --status` checks it.
  **Without it, notebooks are committed with their outputs** (git silently skips a filter it doesn't know).
- **Data never goes into git**: `data/`, `output/`, `notebooks/experiments/`, `notebooks/visualisations/` and data
  files (`*.h5`, `*.tif`, `*.npy`, `*.joblib`, `*.vti`, …) are ignored. Tests make small synthetic data at run time.
  Use `git add -f` only for a deliberate, small exception.
- Never put credentials in notebooks or code: use `getpass`, SSH keys or AWS profiles (see [storage](storage.md)).

## Checks (all must pass before committing)

```bash
ruff check packages && ruff format --check packages && mypy && pytest
cd frontend && npm ci && npm run format:check && npm test && npm run build && cd ..
```

Tests use small synthetic data. SFTP tests run an in-process SFTP server (asyncssh); S3 tests use moto.

## Web UI

The frontend (`frontend/`, React + TypeScript) builds into `packages/pd-xray-server/src/pd_xray/server/static/`,
which is committed so users don't need Node. **After any frontend change, run `npm run build` and commit the
`static/` folder with it.** Old asset files are removed by the build; commit their deletion too.

```bash
cd frontend
npm run dev        # http://localhost:5173 with live reload; API proxied to `pd-xray serve` on port 8000
```

## Python 3.10

The Jetson runs Python 3.10, so code must too (no 3.11+ features; ruff and mypy target 3.10). Test it in a venv as in
[installation](installation.md#python-310--jetson-no-conda) plus `pip install pytest httpx==0.28.1 moto==5.2.4`, then
`python -m pytest -q --import-mode=importlib packages` from the repository root.

## Releases

The five packages are released together, with the same version, each requiring its siblings at `~=` that version
(`test_versions.py` checks this). To release:

1. Set `version` in all five `packages/*/pyproject.toml` and the `~=X.Y.Z` pins between them; build the UI.
2. Run all checks, then build the wheels (pure Python, so one set works on x86-64 and the Jetson's aarch64):

   ```bash
   rm -rf dist && pip wheel --no-deps -w dist ./packages/pd-xray-*
   ```

3. Tag the commit (`vX.Y.Z`), create a GitHub release for the tag, and attach the five `dist/*.whl` files.

Users then install with `--find-links` (see [installation](installation.md#from-a-release-other-projects)).

## Manual checks on real data

[`manual_testing.md`](manual_testing.md) walks through every feature with the sample files in `data/` (not in git).
