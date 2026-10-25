# pd-xray-storage

Storage backends for pd_xray: local folders (also NFS/SMB mounts), SFTP (e.g. Synology NAS) and S3 (AWS or
S3-compatible). Paths are relative to the storage root; files open as seekable, range-reading binary files.

```bash
pip install "pd-xray-storage[sftp]"   # asyncssh
pip install "pd-xray-storage[s3]"     # boto3
```

```python
from pd_xray.storage import LocalStorage, S3Storage, SFTPStorage, copy

with SFTPStorage("nas.lab", "me", password="…", root="/volume1/xray", host_key="SHA256:…") as nas:
    print([e.name for e in nas.list("scans")])
    with nas.open("scans/0042/a.tif") as f:
        header = f.read(8)

with S3Storage("bucket", prefix="scans", endpoint_url="https://minio.lab:9000", access_key="…", secret_key="…") as s3:
    print(s3.stat("0042/a.tif").size)
    copy(s3, "0042", LocalStorage("/data"), "0042")  # parallel, resumable; skips identical files; checks size/MD5
```

Command line: `pd-xray-storage ls sftp://me@nas.lab/volume1/xray --password` and
`pd-xray-storage copy s3://bucket/scans/0042 /data --profile lab` (Ctrl-C, then the same command resumes).

Guide: [docs/storage.md](../../docs/storage.md).
