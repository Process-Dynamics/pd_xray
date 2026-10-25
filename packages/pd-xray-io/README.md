# pd-xray-io

Lazy readers and writers for HDF5, TIFF, PNG and JPG, folders of single images, and format conversion.

```bash
pip install "pd-xray-io[all]"      # or [hdf5], [tiff], [png]
```

```python
from pd_xray.io import open_dataset

ds = open_dataset("scan.h5", dataset="projections")
frame = ds[100]
```

Guide: [docs/io.md](../../docs/io.md).
