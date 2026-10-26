# `pd_xray.storage`: local, NAS (SFTP) and S3 storage

```python
from pd_xray.storage import LocalStorage, SFTPStorage, S3Storage, copy
```

Install `pd-xray-storage[sftp]` for a NAS, `[s3]` for S3, or `[all]`. A storage is a place that holds files. All
three kinds work the same way.

## Opening a storage

```python
local = LocalStorage("/data/scans")                       # also NFS/SMB mounts

nas = SFTPStorage("nas.lab", "alice", password="…", root="/volume1/xray")
nas = SFTPStorage("nas.lab", "alice", key_file="~/.ssh/id_ed25519", root="/volume1/xray")
nas = SFTPStorage("nas.lab", "alice", root="/volume1/xray")   # SSH agent or default keys in ~/.ssh

s3 = S3Storage("my-bucket", prefix="scans", access_key="…", secret_key="…")
s3 = S3Storage("my-bucket", profile="lab")                     # a profile from ~/.aws
s3 = S3Storage("my-bucket")                                    # AWS environment variables / ~/.aws defaults
s3 = S3Storage("xray", endpoint_url="https://minio.lab:9000", access_key="…", secret_key="…")   # non-AWS service
```

Avoid typing passwords into notebooks that you share: use `getpass.getpass()`, SSH keys, or AWS profiles.

Connections open on first use and reopen once by themselves if dropped. Close them when done, or use `with`:

```python
with SFTPStorage("nas.lab", "alice", password=getpass.getpass()) as nas:
    ...
```

### First connection to a NAS: host keys

The NAS's identity is checked against `~/.ssh/known_hosts`. For a NAS you've never connected to with `ssh`, you get
a `HostKeyError` that shows its fingerprint:

```python
from pd_xray.storage import HostKeyError
try:
    nas.check()
except HostKeyError as error:
    print(error.fingerprint)          # SHA256:…  compare it with your admin's / the NAS's
nas = SFTPStorage("nas.lab", "alice", password="…", host_key="SHA256:…")   # once you trust it
```

(Or connect once with `ssh alice@nas.lab` and accept the key; it's then in `known_hosts`.)

## Browsing and reading

Paths are relative to the storage's root and use `/`. `""` is the root.

```python
nas.check()                           # connects and reads the root; raises a clear error if that fails
for entry in nas.list("2026/scan_0042"):
    print(entry.name, entry.is_dir, entry.size, entry.modified)   # size in bytes, modified as a POSIX time
nas.stat("2026/scan_0042/a.tif")      # one Entry
nas.exists("2026/scan_0042")

with nas.open("2026/scan_0042/a.tif") as f:   # seekable: reads only the bytes you ask for
    f.seek(1_000_000)
    header = f.read(4096)
nas.read_range("2026/scan_0042/a.tif", offset=0, size=8)   # thread-safe range read
```

## Copying

`copy` copies a file, or a folder with everything below it, between any two storages (also within one).
Remote-to-remote copies stream through your computer without temporary files.

```python
result = copy(nas, "2026/scan_0042", LocalStorage("/data"), "scan_0042")   # download
copy(LocalStorage("/data"), "scan_0042", s3, "2026/scan_0042")                # upload
copy(nas, "2026/scan_0042", s3, "2026/scan_0042")                             # NAS -> S3
result.copied, result.skipped, result.bytes_copied, result.checksummed
```

- **Destination path** is the path of the copy itself: here `/data/scan_0042/…`.
- **Run it again** and only what is missing is copied: files with the same size and modification time are skipped,
  and an interrupted file continues from where it stopped (its partial copy is a hidden
  `.name.<size>-<time>.pd-xray-part` file next to it; on S3, an unfinished multipart upload marked by a hidden object
  of that name, which keeps using space until the copy is run again).
- **Existing different files** stop the copy before anything is copied (`FileExistsError`), unless `overwrite=True`.
- **Checks**: every file's size; its MD5 too when the source stores one (S3 objects uploaded in one part).
  `verify=True` re-reads source and copy and compares SHA-256 (about twice the time). A mismatch raises
  `IntegrityError` and deletes the bad copy.
- Copies keep the source's modification time. S3 can't set an object's date, so it is stored in the object's
  `mtime` metadata (as rclone does); `S3Storage.stat` reports it, `list` shows the upload time.
- **S3 uploads** send files over 8 MB in parts (4 at once), each checked by S3 with Content-MD5.
- **Speed**: 4 files at a time (`workers=`) with several parallel reads per file. One SSH connection reaches about
  350 MB/s both ways; a 1 Gb network carries about 110 MB/s.
- **Progress and cancel**: `progress=` receives bytes done/total and the current file name; `cancel=` a
  `CancelToken` (see [core](core.md)). Cancelling keeps the partial files, so the next run resumes.

```python
from pd_xray.core import CancelToken
stop = CancelToken()
copy(s3, "run7", LocalStorage("/data"), "run7", verify=True, progress=lambda p: print(p.current, p.total, p.message),
     cancel=stop)
```

## Errors

| Error | Meaning |
|---|---|
| `AuthenticationError` | Wrong user/password/key, missing or refused S3 keys |
| `HostKeyError` | The NAS's key is unknown or changed (see above) |
| `StorageConnectionError` | Host or endpoint unreachable, or timeout (15 s) |
| `FileNotFoundError`, `NotADirectoryError`, `IsADirectoryError`, `PermissionError` | As usual, for paths |
| `InvalidPathError` | A path with `..`, or a symlink leading out of a local root |
| `IntegrityError` | A copy didn't match its source |

All storage errors derive from `StorageError` (and `pd_xray.core.PdXrayError`). Messages never contain passwords
or keys.

## Command line: `pd-xray-storage`

Locations (source and destination) are local paths, `sftp://user@host[:port]/absolute/path` or `s3://bucket/key`.

```bash
pd-xray-storage ls sftp://alice@nas.lab/volume1/xray --password          # asks for the password
pd-xray-storage ls s3://my-bucket/scans --profile lab
pd-xray-storage copy sftp://alice@nas.lab/volume1/xray/scan_0042 /data --password
pd-xray-storage copy s3://xray/run7 /data/run7 --endpoint-url https://minio.lab:9000 --verify
pd-xray-storage copy /data/scan_0042 sftp://alice@nas.lab/volume1/xray/ --key-file ~/.ssh/id_ed25519
```

- `copy` into an existing folder, or one ending in `/`, puts the copy inside it (like `cp`). Ctrl-C stops; the same
  command resumes. Connection options apply to both locations.
- SFTP login: SSH agent or `~/.ssh` keys by default, `--key-file FILE`, or `--password` (asked for, never typed on
  the command line). An unknown NAS prints the `--host-key SHA256:…` option to add once you trust it.
- S3 login: `--profile`, or the `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables.
- Other options: `--overwrite`, `--verify`, `--workers N`, `--region`. `pd-xray-storage copy --help` lists them.
