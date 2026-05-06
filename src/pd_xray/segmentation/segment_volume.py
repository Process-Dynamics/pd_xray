"""Parallel volume segmentation using a saved RFSegmenter model.

Segments a directory of 2D TIFF slices using a pre-trained RFSegmenter,
writing predicted label maps to an output directory.

When workers > 1, each worker process loads the model once (via initializer)
and processes slices independently. The RFSegmenter's internal n_jobs is
overridden to 1 in worker processes to avoid CPU over-subscription.

When workers == 1, the RF uses all cores internally (n_jobs=-1 from the
saved model), which is optimal for single-slice throughput.

CLI usage::

    python -m pd_xray.segmentation.segment_volume \\
        --model rf_segmenter.joblib \\
        --input_dir /path/to/tiff_slices/ \\
        --output_dir /path/to/masks/ \\
        --workers 4

Python usage::

    from pd_xray.segmentation.segment_volume import segment_volume

    results = segment_volume(
        model_path="rf_segmenter.joblib",
        input_dir="/data/scan_001/",
        output_dir="/data/scan_001_masks/",
        workers=4,
        postprocess_steps=[{"name": "close", "kernel_size": 3}],
    )
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from pd_xray.core.logging import get_logger
from pd_xray.processing.image_processor import postprocess_segmentation_mask
from pd_xray.segmentation import RFSegmenter

logger = get_logger(__name__)

# Per-worker globals populated by _worker_init — never set in the main process.
_model: Optional[RFSegmenter] = None


def _worker_init(model_path: str, n_jobs: int) -> None:
    """Load the model once per worker process and patch RF n_jobs."""
    global _model
    _model = RFSegmenter.load(model_path)
    _model._rf.n_jobs = n_jobs
    logger.debug("Worker PID %d ready (n_jobs=%d)", os.getpid(), n_jobs)


def _process_slice(
    task: tuple[str, str, Optional[list[dict]], Optional[list[dict]]]
) -> tuple[str, bool, Optional[str]]:
    """Segment one slice. Runs inside a worker process.

    Args:
        task: (input_path, output_path, preprocessing_steps, postprocess_steps)

    Returns:
        (output_path, success, error_message)
    """
    input_path, output_path, preprocessing_steps, postprocess_steps = task
    try:
        data = tifffile.imread(input_path).astype(np.float32)
        image = data[0] if data.ndim == 3 else data

        mask = _model.predict(image, preprocessing_steps=preprocessing_steps)

        if postprocess_steps:
            mask = postprocess_segmentation_mask(
                mask, steps=postprocess_steps, background_class=0
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(out), mask.astype(np.int32))
        return output_path, True, None
    except Exception as exc:
        return output_path, False, str(exc)


def segment_volume(
    model_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    glob_pattern: str = "*.tif",
    workers: int = 1,
    preprocessing_steps: Optional[list[dict]] = None,
    postprocess_steps: Optional[list[dict]] = None,
    skip_existing: bool = False,
) -> dict[str, bool]:
    """Segment all TIFF slices in a directory using a saved RFSegmenter.

    Args:
        model_path:          Path to a ``.joblib`` file saved by ``RFSegmenter.save()``.
        input_dir:           Directory containing 2D TIFF slices.
        output_dir:          Directory where predicted label TIFFs are written.
                             Mirror filenames are used (same stem, ``.tif`` extension).
        glob_pattern:        Glob used to collect input files. Defaults to ``"*.tif"``.
        workers:             Number of parallel worker processes.

                             - ``1``  — sequential; RF uses all cores internally.
                             - ``N>1`` — N worker processes; RF in each is limited to
                               ``max(1, total_cores // N)`` threads to avoid overload.
        preprocessing_steps: Per-call preprocessing override passed to
                             ``RFSegmenter.predict()``. ``None`` uses the pipeline
                             saved inside the model.
        postprocess_steps:   Morphological post-processing applied after prediction
                             (see ``postprocess_segmentation_mask``). ``None`` skips.
        skip_existing:       Skip slices whose output file already exists.

    Returns:
        Dict mapping each output path (str) to ``True`` (success) or ``False`` (failed).
    """
    model_path = Path(model_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    slices = sorted(input_dir.glob(glob_pattern))
    if not slices:
        raise FileNotFoundError(
            f"No files matching '{glob_pattern}' found in {input_dir}"
        )

    tasks: list[tuple[str, str, Optional[list[dict]], Optional[list[dict]]]] = []
    for src in slices:
        dst = output_dir / src.name
        if skip_existing and dst.exists():
            logger.info("Skipping existing: %s", dst)
            continue
        tasks.append((str(src), str(dst), preprocessing_steps, postprocess_steps))

    if not tasks:
        logger.info("All %d slices already exist; nothing to do.", len(slices))
        return {}

    logger.info(
        "Segmenting %d/%d slices  |  workers=%d  |  model=%s",
        len(tasks),
        len(slices),
        workers,
        model_path,
    )

    results: dict[str, bool] = {}

    if workers == 1:
        # Sequential: RF uses n_jobs=-1 (all cores) internally.
        model = RFSegmenter.load(model_path)
        for i, (src, dst, pre, post) in enumerate(tasks, 1):
            logger.info("Slice %d/%d: %s", i, len(tasks), Path(src).name)
            try:
                data = tifffile.imread(src).astype(np.float32)
                image = data[0] if data.ndim == 3 else data
                mask = model.predict(image, preprocessing_steps=pre)
                if post:
                    mask = postprocess_segmentation_mask(
                        mask, steps=post, background_class=0
                    )
                out = Path(dst)
                out.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(str(out), mask.astype(np.int32))
                results[dst] = True
            except Exception as exc:
                logger.error("Failed %s: %s", Path(src).name, exc)
                results[dst] = False
    else:
        n_jobs_per_worker = max(1, os.cpu_count() // workers)
        completed = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(str(model_path), n_jobs_per_worker),
        ) as pool:
            futures = {pool.submit(_process_slice, t): t for t in tasks}
            for future in as_completed(futures):
                output_path, success, err = future.result()
                completed += 1
                if success:
                    logger.info(
                        "[%d/%d] Done: %s",
                        completed,
                        len(tasks),
                        Path(output_path).name,
                    )
                else:
                    logger.error(
                        "[%d/%d] Failed: %s — %s",
                        completed,
                        len(tasks),
                        Path(output_path).name,
                        err,
                    )
                results[output_path] = success

    n_ok = sum(results.values())
    logger.info("Complete: %d/%d slices succeeded.", n_ok, len(results))
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Segment a directory of TIFF slices using a saved RFSegmenter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Path to .joblib model file.")
    p.add_argument("--input_dir", required=True, help="Directory of input TIFF slices.")
    p.add_argument("--output_dir", required=True, help="Directory for output label TIFFs.")
    p.add_argument("--glob", default="*.tif", help="Glob pattern to match input files.")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes. "
            "Use 1 for sequential (RF uses all cores). "
            "Use >1 for slice-level parallelism."
        ),
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip slices whose output file already exists.",
    )
    return p


if __name__ == "__main__":
    from pd_xray.core.logging import setup_logging

    setup_logging(level="INFO")
    args = _build_parser().parse_args()
    segment_volume(
        model_path=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        glob_pattern=args.glob,
        workers=args.workers,
        skip_existing=args.skip_existing,
    )
