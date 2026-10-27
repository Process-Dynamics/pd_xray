# pd-xray-io

Lazy readers and writers for HDF5, TIFF, PNG and JPG, folders of single images, and format conversion.

Install from a GitHub release with `pip install --find-links …`: see the [main
README](../../README.md#2-python-packages). Extras: `[hdf5]`, `[tiff]`, `[png]` or `[all]`.

```python
from pd_xray.io import open_dataset

ds = open_dataset("scan.h5", dataset="projections")
frame = ds[100]
```

Guide: [docs/io.md](../../docs/io.md).
