# Installation

## Conda (recommended)

Needs git and [Miniforge](https://conda-forge.org/download/) (any conda works; use the conda-forge channel).

```bash
git clone https://github.com/Process-Dynamics/pd_xray.git
cd pd_xray
conda env create -f environment.yml
conda activate pd-xray
pd-xray serve --root /path/to/data
```

`environment.yml` pins every dependency to an exact version (NumPy 1.26.4, matching the lab's Jetson) and installs
all five packages from the `packages/` folder. The web UI is already built; Node isn't needed.

Check the installation:

```bash
python -c "import pd_xray.core, pd_xray.io, pd_xray.processing, pd_xray.storage, pd_xray.server; print('ok')"
pd-xray serve --help
```

> Don't run `pip install .` in the repository root. The root `pyproject.toml` belongs to the old `pd_xray` package,
> which conflicts with the new ones.

### Updating

```bash
cd pd_xray
git pull
conda env update -f environment.yml --prune
pip install --force-reinstall --no-deps ./packages/pd-xray-core "./packages/pd-xray-storage[all]" \
    "./packages/pd-xray-io[all]" ./packages/pd-xray-processing ./packages/pd-xray-server
```

The last line is needed because `environment.yml` installs copies of the packages: `git pull` alone doesn't update
them.

## Only the libraries, with pip

For using `pd_xray.io` etc. from another project, install just what you need into that project's environment. The
packages aren't on PyPI yet, so install from a clone (core first, because the others depend on it):

```bash
pip install ./packages/pd-xray-core "./packages/pd-xray-io[hdf5,tiff]"     # e.g. read HDF5 and TIFF only
pip install ./packages/pd-xray-processing
pip install "./packages/pd-xray-storage[sftp]"                            # NAS access
```

Extras: `pd-xray-io[hdf5]` (h5py, hdf5plugin), `[tiff]` (tifffile), `[png]` (Pillow, also JPG), `[all]`;
`pd-xray-storage[sftp]` (asyncssh), `[s3]` (boto3), `[all]`. A missing extra gives an error that names the command
to install it.

## Python 3.10 / Jetson (no conda)

Everything runs on Python 3.10 to 3.12 with NumPy 1.x. On the Jetson (system Python 3.10), use a virtual
environment and the same versions as `environment.yml`:

```bash
python3 -m venv ~/pd-xray-venv
source ~/pd-xray-venv/bin/activate
pip install numpy==1.26.4 h5py==3.16.0 hdf5plugin==7.1.0 tifffile==2025.5.10 pillow==12.3.0 scipy==1.15.3 \
    scikit-image==0.25.2 pyyaml==6.0.3 asyncssh==2.24.0 boto3==1.43.114 fastapi==0.142.2 uvicorn==0.54.0 \
    websockets==16.1.1
pip install ./packages/pd-xray-core "./packages/pd-xray-storage[all]" "./packages/pd-xray-io[all]" \
    ./packages/pd-xray-processing ./packages/pd-xray-server
```

(SciPy 1.15.3 is the pip release of conda's 1.15.2; both are the last line for Python 3.10.)

## A shared lab server

One server can serve the app to a few people (up to about 5). There is no login yet, so only do this on a trusted
internal network.

```bash
pd-xray serve --host 0.0.0.0 --port 8000 --root /data/scans --root /mnt/nas --no-browser --cache-gb 8 --workers 4
```

Then open `http://SERVER_NAME:8000` on your own computer.

Safer alternative, no open port: keep the default `--host 127.0.0.1` on the server and use an SSH tunnel from your
computer, then open `http://localhost:8000`:

```bash
ssh -N -L 8000:localhost:8000 you@SERVER_NAME
```

| Option | Meaning |
|---|---|
| `--root FOLDER` | A folder users may browse, open and download into. Repeat for several. Default: the current folder. |
| `--host` | `127.0.0.1` (default): this computer only. `0.0.0.0`: every network interface. |
| `--port` | Default 8000. |
| `--cache-gb` | Memory for cached frames and tiles (default 2). More makes revisiting frames faster. |
| `--workers` | Background jobs (downloads, dataset range) running at once (default 2). Others wait in a queue. |
| `--no-browser` | Don't open a browser on start (for servers). |

Each browser's NAS/S3 connections are private to that browser and kept only in the server's memory: restarting the
server drops them, and passwords are never written to disk. The server never uses its own SSH keys or AWS
credentials for users' connections.
