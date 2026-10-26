# pd_xray

X-ray radiography toolkit from the Process Dynamics group. Browse and scrub through large image stacks (HDF5, TIFF,
PNG, JPG; 100 GB+ files are fine), try ImageJ-style processing live, run it on whole datasets, and copy data between
your computer, a NAS and S3, all in a web browser. The same functions are available as Python packages.

## Quick start (your own computer)

You need [Miniforge](https://conda-forge.org/download/) (or another conda) and git. Linux, macOS or Windows (WSL).

```bash
git clone https://github.com/Process-Dynamics/pd_xray.git
cd pd_xray
conda env create -f environment.yml      # once; takes a few minutes
conda activate pd-xray
pd-xray serve --root /path/to/your/data  # opens http://127.0.0.1:8000 in your browser
```

Next time, only the last two lines. Stop the server with Ctrl-C.

## Using the web app

1. **Open data**: the left panel lists the folders you gave with `--root`. Click a file. For an HDF5 file with several
   datasets, pick one. A folder of numbered single images (`scan_0001.tif`, …) shows as one `▦` entry.
2. **Scrub frames**: drag the slider or use ← / → (Shift: 10 frames, Home/End: first/last). Scroll to zoom, drag to
   pan.
3. **Contrast**: right panel. Drag the black/white handles on the histogram, or use Auto. Display settings never
   change your data.
4. **Process**: pick from the "+ Add step…" menu (flat/dark correction, filters, crop, rotate, background
   subtraction, …). The current frame shows the result; press P to compare with the original.
   **Save…/Load…** keep the steps in a file; **Run…** processes the whole dataset into a new file in the background.
5. **NAS or S3**: in the left panel's top level, "+ Add connection". Files there open directly (read as needed, not
   copied). ⧉ next to any file or folder copies it to a folder on the server, a NAS or S3. Progress and Cancel are in
   the bottom bar; copying again resumes.

## Python

```python
from pd_xray.io import open_dataset
from pd_xray.processing import Pipeline, Step

ds = open_dataset("scan.h5", dataset="projections")   # lazy: reads only what you index
frame = ds[100]                                        # one frame as a NumPy array
result = Pipeline([Step("median", {"size": 3})])(frame)
```

## More

- [docs/](docs/index.md): installation options (shared server, Jetson, pip), the web app in detail, and how to use each
  Python package.
- Problems or questions: open an issue on GitHub.
