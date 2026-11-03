# ACCESS phase X-ray viewer

A standalone local browser UI for inspecting one phase-field dendrite. Choose one of seven native time steps, drag the sampled 3D surface to choose an angle, and render a 512 × 512 parallel-beam X-ray projection from the corresponding 511³ phase field. The six nonempty frames are used directly; there is no interpolation in time. This example is separate from the main `pd-xray serve` app.

## Requirements

- Python 3.10+ with `numpy`, `scipy`, `h5py`, and `Pillow` (`python -m pip install numpy scipy h5py Pillow`).
- Seven HDF5 files, `phase_0000.h5` through `phase_0006.h5`, under `DATA/downsampled_511/phase/`. These are **generated data**, not included in Git. Each file must contain the `phase_field` dataset and its `time_s` attribute.

The [`pd-tomo`](https://github.com/Process-Dynamics/tomo_simulation/tree/main/packages/pd-tomo) CLI can create the required data from source files you are permitted to read:

```bash
pd-tomo external --source SOURCE --out-dir DATA --size downsampled --fields frac1 --frames all
```

An optional `DATA/downsampled_63/phase/` cache speeds up the later 3D previews. The viewer builds a small preview from the 511³ frames if that cache is absent. It never modifies the source HDF5 files.

## Start and stop

On Windows, place the generated files under `pd_xray/data` (ignored by Git), then double-click [start_access_phase_viewer.cmd](start_access_phase_viewer.cmd). The launcher looks for `pd_xray/.venv`, an active conda environment, or a user Anaconda installation before falling back to `python` on `PATH`. It opens <http://127.0.0.1:8765/> in your default browser.

For another data location, run the launcher from a terminal:

```powershell
.\start_access_phase_viewer.cmd --data-dir "D:\path\to\DATA"
```

You can also run `python access_phase_viewer.py --data-dir DATA --open-browser`. Set `PD_TOMO_VIEWER_DATA` to the same directory if you prefer an environment variable. When the repositories are side by side in one workspace, an existing `access_phase_data/cache` is detected automatically.

The service runs **only while the launcher is open**. Click **Stop server** in the page or close the launcher console to stop it. Closing only the browser tab leaves the service running. The server listens on `127.0.0.1`, so it is available only from this computer.

## Use

1. Select a **Time step**. Step 0 (0.00 s) is almost empty; steps 1–6 cover 0.45–2.50 s.
2. Drag the left preview for yaw and pitch; hold **Shift** while dragging for roll. **Front 0°** and **Side 45°** are quick presets.
3. Adjust the grain and background attenuation coefficients if desired, then click **Render X-ray**. The selected time step and angle are used together.
4. Switch between **Raw (windowed)** and **Enhanced**. Click **Save PNG** only when you want to save the displayed view; its filename includes the time step and angles. Nothing is saved automatically.

The calculated intensity is `I/I₀ = exp[−(τ_background + τ_grain)]`, where `τ_grain` integrates the phase fraction along the beam. **Raw (windowed)** shows `1 − I/I₀` through a fixed 0.88–1.00 display window. **Enhanced** removes the constant background brightness baseline and applies a square-root stretch to reveal thin arms. Both start from the same projection. The 0.20 µm display voxel scale and default attenuation coefficients are illustrative, not physically calibrated to the original simulation.
