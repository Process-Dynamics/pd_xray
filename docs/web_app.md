# Web app

Start it with `pd-xray serve --root /path/to/data` (see [Installation](installation.md) for all options). The page
has three columns: the file browser (left), the viewer (middle) and the panels (right), with a status bar at the
bottom.

## File browser (left)

- The top level lists the `--root` folders, your NAS/S3 connections (🔌) and "+ Add connection".
- Click a folder to enter it, ↑ to go up. "Filter" narrows the list as you type (big folders show the first 2000
  entries; the filter finds the rest).
- Click a file to open it. Supported: HDF5 (`.h5 .hdf5 .hdf .nxs`), TIFF (single or multi-page), PNG, JPG.
- **HDF5**: a file with one image dataset opens directly; with several, a list appears. Datasets that aren't images
  are greyed out. ⚠ marks problems found in the file (hover for details), e.g. frames that were never written.
- **Image sequences**: numbered single-frame files (`0042__002015_img.tif`, `0042__002020_img.tif`, …) show as one
  `▦ 0042__*_img.tif` entry. Opening it gives one dataset with a frame per file, in natural number order.

## Viewer (middle)

- Scroll or double-click to zoom, drag to pan, "Fit" to see the whole frame.
- Frames are drawn as tiles at the resolution needed, so large frames show quickly and sharpen as you zoom in.
- The new frame appears only once it's complete, so the screen never mixes two frames.

### Frame slider and keys

One slider per frame axis (e.g. `angle`, or `time` and `angle`). The number next to it is the frame on screen; it
greys out while the next frame loads.

| Key | Action |
|---|---|
| ← / → | Previous / next frame on the last axis |
| ↓ / ↑ | Previous / next on the axis before it (datasets with two frame axes) |
| Shift + arrows | 10 frames at a time |
| Home / End | First / last frame |
| P | Switch between original and processed (when there are processing steps) |

Keys work anywhere except while typing in a text field. Holding an arrow key plays through the frames.

## Display panel (right)

Display settings only change how the data looks on screen, never the data.

- **Histogram** of the frame on screen. Drag the black and white triangles to set Min/Max. "log" switches to
  logarithmic counts (shows rare values); "full range" shows the frame's whole range. "Reset" gives a straight curve
  and automatic Min/Max.
- **Tone curve** (blue line on the histogram): click to add a point and drag it; double-click a point to remove it.
- **Min / Max**: type exact values. "Auto" uses the 0.1–99.9 percentiles of the frame on screen, "Frame range" its
  min and max, "Dataset range" the min and max over every frame (a background job that reads the whole dataset).
- **Colours**: grey, inverted grey, viridis, magma, inferno, cividis, fire.
- **Image**: PNG shows exact 8-bit values; JPEG is smaller, which helps over slow networks.
- **Tone (display)**: exposure, contrast, highlights, shadows, whites, blacks, like a photo editor. Double-click a
  slider to reset it, or "Reset" for all. "Add as step" turns the current Min/Max and tone into a Tone processing
  step, so it can be applied to the data.
- **Dataset**: shape, type, size, warnings and the file's metadata.

## Processing panel (right)

Build a chain of steps and see the result on the current frame at full resolution.

- "+ Add step…" adds an operation. Each step card has its settings, a checkbox to switch it off without removing it,
  ↑ ↓ to reorder and × to remove. The time each step took on the last frame is shown.
- **Original / Processed** (or P) switches the view. The processed view has its own Min/Max.
- While you step through frames, only the frame you stop on is processed, so the slider stays responsive.

| Step | What it does |
|---|---|
| Flat/dark correction | (frame − dark) / (flat − dark). Dark or flat: "Choose…" a folder (every image in it is averaged) or one or more files. Paths inside the dataset's folder are kept relative. |
| Invert | high + low − value (e.g. high 4095 for 12-bit data) |
| Gaussian blur | Gaussian filter, sigma in pixels |
| Median filter | Square window; removes hot pixels and zingers |
| Normalise | This frame's low/high percentiles → 0..1 (each frame scaled separately) |
| Clip | Limit values to low..high |
| Threshold | 1 above (or below) a value, 0 elsewhere |
| Tone | Exposure, contrast, curve, … between a black and a white value; output stays in black..white |
| Crop | Keep a rectangle (top, left, height, width; 0 = to the edge) |
| Rotate | 90°, 180° or 270° clockwise |
| Flip | Horizontal or vertical mirror |
| Subtract background (rolling ball) | ImageJ's rolling ball; tick "light background" for bright backgrounds |

Saving the pipeline and running it on the whole dataset come in a later version. In Python, see
[processing](processing.md).

## NAS and S3 connections

"+ Add connection" at the browser's top level.

- **NAS (SFTP)**: host (name or IP), port (22), user name, optionally a folder to start in (e.g. `/volume1/xray`),
  and a password or a private key file (plus its passphrase if it has one). On a Synology, an admin must enable SFTP
  (Control Panel › File Services › FTP › SFTP).
- **S3 bucket**: endpoint URL (empty for AWS; e.g. `https://minio.lab:9000` for others), region (optional), bucket,
  folder (optional), access key and secret key.
- **First connection to a NAS**: you'll see the server's key fingerprint (`SHA256:…`). Check it with your admin (or
  `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` on the NAS) and click "Trust this key and connect". This
  protects your password from a machine pretending to be the NAS.
- Clear errors: wrong password ("refused the login"), wrong host ("timed out" after 15 s), missing bucket, …
- Connections are private to your browser. Passwords and keys stay in the server's memory only, so after a server
  restart you connect again. The dialog remembers host, bucket etc. (never passwords): click a saved entry to fill
  the form. × next to a connection disconnects it.
- Opening files directly from a NAS or S3 comes later; for now, download them.

## Downloading

In a connection, ⤓ next to a file or folder (or next to the path, for the folder you're in) → choose a folder inside
the allowed folders → "Download here".

- The download runs in the background. The status bar shows size done, speed and time left; × cancels.
- Downloading the same thing again **resumes**: completed files with the same size and date are skipped, and a
  partly downloaded file continues where it stopped.
- A file that already exists with a different size or date stops the download (nothing is overwritten), unless you
  tick "Overwrite different files".
- Every file's size is checked; files from S3 are also checked against the MD5 that S3 stores (when it has one).
  "Verify (slower)" re-reads both copies and compares SHA-256.
- Downloaded files keep their original modification date. Partial files are hidden (`.name.….pd-xray-part`) until
  complete.
- Hover a finished job in the status bar to see what it copied.

## Status bar (bottom)

- The dot is green while connected to the server. If the server restarts, the page reconnects and reopens your
  dataset.
- What is loading right now, with seconds elapsed.
- Background jobs with progress, time left and × to cancel.
- The last error (× to dismiss).
