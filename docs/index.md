# pd_xray documentation

Start with the [README](../README.md) for a quick start.

| Page | For |
|---|---|
| [Installation](installation.md) | Conda, pip only, Python 3.10 / Jetson, a shared lab server, updating |
| [Web app](web_app.md) | Every panel of the browser app, keyboard shortcuts, NAS/S3, downloads, sharing the server |
| [`pd_xray.io`](io.md) | Opening HDF5/TIFF/PNG/JPG files and folders lazily, writing and converting |
| [`pd_xray.processing`](processing.md) | Operations, pipelines, pipeline files, processing whole datasets |
| [`pd_xray.storage`](storage.md) | Local, NAS (SFTP) and S3 storage, copying, the `pd-xray-storage` command |
| [`pd_xray.core`](core.md) | Axis names, progress reporting and cancelling, errors |
| [Development](development.md) | Tests, checks, rebuilding the web UI |
| [Manual testing](manual_testing.md) | Step-by-step checks on real data (for maintainers) |

## The packages

pd_xray is five pip packages that share the `pd_xray` namespace. Use the libraries on their own in scripts and
notebooks; the web app is built on top of them.

| Package | Import | Needs |
|---|---|---|
| `pd-xray-core` | `pd_xray.core` | NumPy |
| `pd-xray-io` | `pd_xray.io` | core; extras `[hdf5]`, `[tiff]`, `[png]` or `[all]` |
| `pd-xray-processing` | `pd_xray.processing` | core, SciPy, scikit-image, PyYAML |
| `pd-xray-storage` | `pd_xray.storage` | core; extras `[sftp]`, `[s3]` or `[all]` |
| `pd-xray-server` | `pd_xray.server`, `pd-xray` command | all of the above, FastAPI, uvicorn |

Conventions used everywhere:

- **Lazy data.** Files are never loaded whole. Indexing a dataset reads only that part, so 100 GB files work.
- **Native dtype.** Data comes back as stored (usually `uint16`). Processing works in `float32`; conversions are
  explicit.
- **Named axes.** Arrays end with `("y", "x")` (plus `"channel"` for colour). Frame axes in front are named, e.g.
  `("angle", "y", "x")` or `("time", "y", "x")`.
- **Long operations** accept `progress=` (a function called with progress updates) and `cancel=` (a `CancelToken`).
