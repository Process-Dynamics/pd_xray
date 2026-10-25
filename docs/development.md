# Development

Plan and decisions: [`WORK_PLAN.md`](../WORK_PLAN.md). Rules for code and AI assistants: [`CLAUDE.md`](../CLAUDE.md).

## Environment

```bash
conda env create -f environment-dev.yml      # editable installs, tools (ruff, mypy, pytest), Node 24, moto
conda activate pd-xray-dev
```

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

## Manual checks on real data

[`manual_testing.md`](manual_testing.md) walks through every feature with the sample files in `data/` (not in git).
