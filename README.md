# pd_xray

X-ray radiography toolkit from the Process Dynamics group. You can use it in two ways:

- **Web app**: browse and scrub through large image stacks (HDF5, TIFF, PNG, JPG; 100 GB+ files are fine), try
  ImageJ-style processing live, run it on whole datasets, and copy data between your computer, a NAS and S3.
- **Python packages**: the same reading, processing and storage functions, in your own scripts and notebooks.

Runs on Linux, macOS and Windows (WSL), with Python 3.10–3.12. Current release:
[v0.2.0](https://github.com/Process-Dynamics/pd_xray/releases/tag/v0.2.0).

## 1. Web app

You need [Miniforge](https://conda-forge.org/download/) (or another conda) and git.

```bash
git clone https://github.com/Process-Dynamics/pd_xray.git
cd pd_xray
conda env create -f environment.yml      # once; takes a few minutes
conda activate pd-xray
pd-xray serve --root /path/to/your/data  # opens http://127.0.0.1:8000 in your browser
```

Next time, only the last two lines. Stop the server with Ctrl-C. To update to a new release:
[installation](docs/installation.md#updating).

1. **Open data**: the left panel lists the folders you gave with `--root`. Click a file. For an HDF5 file with several
   datasets, pick one. A folder of numbered single images (`scan_0001.tif`, …) shows as one `▦` entry.
2. **Scrub frames**: drag the slider or use ← / → (Shift: 10 frames, Home/End: first/last). Scroll to zoom, drag to
   pan.
3. **Contrast**: right panel. Drag the black/white handles on the histogram, or use Auto. Display settings never
   change your data.
4. **Process**: pick from the "+ Add step…" menu (flat/dark correction, filters, crop, rotate, background
   subtraction, …). The current frame shows the result; press P to compare with the original.
   **Save…/Load…** keep the steps in a file; **Run…** processes the whole dataset into a new file in the
   background.
5. **NAS or S3**: in the left panel's top level, "+ Add connection". Files there open directly (read as needed, not
   copied). ⧉ next to any file or folder copies it to a folder on the server, a NAS or S3. Progress and Cancel are in
   the bottom bar; copying again resumes.

More: [web app guide](docs/web_app.md), and running it
[on a shared lab server](docs/installation.md#a-shared-lab-server) or
[on the Jetson](docs/installation.md#python-310--jetson-no-conda).

## 2. Python packages

Five packages share the `pd_xray` namespace. Install only what you need; each pulls in `pd-xray-core` and its own
dependencies from PyPI.

| Package | Import | For | Extras |
|---|---|---|---|
| `pd-xray-io` | `pd_xray.io` | Reading HDF5/TIFF/PNG/JPG lazily, writing, converting | `[hdf5]`, `[tiff]`, `[png]`, `[all]` |
| `pd-xray-processing` | `pd_xray.processing` | Operations and pipelines on NumPy frames | |
| `pd-xray-storage` | `pd_xray.storage` | Local, NAS (SFTP) and S3 storage, copying | `[sftp]`, `[s3]`, `[all]` |
| `pd-xray-core` | `pd_xray.core` | Axis names, progress, cancelling, errors | |
| `pd-xray-server` | `pd_xray.server` | The web app (`pd-xray serve`) | |

The packages aren't on PyPI. Point pip at a release with `--find-links`:

```bash
REL=https://github.com/Process-Dynamics/pd_xray/releases/expanded_assets/v0.2.0
pip install --find-links $REL "pd-xray-io[hdf5,tiff]" pd-xray-processing
```

In a `requirements.txt`, put `--find-links https://github.com/Process-Dynamics/pd_xray/releases/expanded_assets/v0.2.0`
on the first line, then e.g. `pd-xray-io[hdf5]~=0.2.0`.

```python
from pd_xray.io import convert, map_frames, open_dataset
from pd_xray.processing import Pipeline, Step

ds = open_dataset("scan.h5", dataset="projections")   # lazy: reads only what you index
frame = ds[100]                                        # one frame as a NumPy array, in its stored dtype

pipeline = Pipeline([Step("median", {"size": 3}), Step("crop", {"top": 100, "height": 1800})])
processed = pipeline(frame)                            # float32
pipeline.save("recipe.yaml")                           # also loads in the web app

convert(map_frames(ds, pipeline, workers=8), "out/processed.tif", progress=print)   # the whole dataset
```

Guides: [`pd_xray.io`](docs/io.md), [`pd_xray.processing`](docs/processing.md), [`pd_xray.storage`](docs/storage.md),
[`pd_xray.core`](docs/core.md). Installing from a clone: [installation](docs/installation.md).

## More

- [docs/](docs/README.md): all guides.
- [Development](docs/development.md): tests, rebuilding the web UI, making a release.
- Problems or questions: open an issue on GitHub.
