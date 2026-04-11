import contextlib
import os
import sys
from typing import Any

from pd_xray.core.constants import DEFAULT_MAX_WORKERS
from pd_xray.core.exceptions import CPUDetectionError, GPUTorchError
from pd_xray.core.logging import get_logger

logger = get_logger(__name__)


def detect_cpus() -> int:
    """Return the number of usable CPU cores.

    Resolution order:
    1. SLURM_CPUS_PER_TASK environment variable (HPC)
    2. PBS_NP or PBS_NUM_PPN (PBS/Torque HPC)
    3. os.cpu_count() (local machine)
    4. Fallback: DEFAULT_MAX_WORKERS from constants

    Note: On HPC, os.cpu_count() returns ALL cores on the node, not the cores allocated to your job. Always prefer
    SLURM/PBS vars.
    """
    try:
        cpu_core_count = os.cpu_count() or 0
        if cpu_core_count > 0:
            return cpu_core_count
        else:
            return DEFAULT_MAX_WORKERS
    except CPUDetectionError as e:
        logger.warning(e)
        return DEFAULT_MAX_WORKERS


def _parse_nvidia_smi_csv(csv_text: str) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in csv_text.strip().splitlines() if ln.strip()]
    gpus: list[dict[str, Any]] = []

    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        with contextlib.suppress(IndexError, ValueError):
            idx = int(parts[0])
        name = parts[1] if len(parts) > 1 else "Unknown"

        def _safe_int(i: int, _parts: list[str] = parts) -> int | None:
            try:
                return int(_parts[i])
            except (IndexError, ValueError):
                return None

        total_mib = _safe_int(2)
        free_mib = _safe_int(3)
        util_pct = _safe_int(4)

        entry = {
            "id": idx,
            "name": name,
            "total_memory_GB": round(total_mib / 1024, 2) if total_mib is not None else None,
            "free_memory_GB": round(free_mib / 1024, 2) if free_mib is not None else None,
            "utilization_precent": util_pct,
            "source": "nvidia-smi",
        }
        gpus.append(entry)
    return gpus


def detect_gpus() -> list[dict[str, Any]]:
    """Detect available GPUs.

    Returns a list of dicts, one per GPU:
        [{"id": 0, "name": "NVIDIA RTX A6000", "memory_bytes": }, ...]

    Resolution:
    1. If CUDA_VISIBLE_DEVICES is set, only those GPUs are visible.
    2. Use torch.cuda if PyTorch is installed.
    3. Use subprocess nvidia-smi as fallback.
    4. If no GPUs found, return empty list (never raise).

    Note: PyTorch and nvidia-smi are optional. If neither is available, return empty list. pd-core does NOT depend
    on PyTorch
    """
    gpus: list[dict[str, Any]] = []
    try:
        import torch
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "id": i,
                        "name": props.name,
                        "total_memory_GB": round(props.total_memory / 1024**3, 2),
                        "multi_processor_count": props.multi_processor_count,
                        "compute_capability": f"{props.major}.{props.minor}",
                    })
            return gpus
        except GPUTorchError as e:
            logger.warning(e)
    except ModuleNotFoundError:
        logger.warning(
            "PyTorch is not installed. Trying with 'nvidia-smi' instead.")
        pass

    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return []

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    ]

    try:
        nvidia_smi_command = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if nvidia_smi_command.returncode != 0:
            return []
        return _parse_nvidia_smi_csv(nvidia_smi_command.stdout)
    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []


def _linux_mem_available() -> int | None:
    """Read MemAvailable from /proc/meminfo (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    value_kb = int(parts[1])
                    return int(value_kb / 1024**2)  # in GB
    except Exception as e:
        logger.warning(f"Failed reading /proc/meminfo: {e}")
    return None


def _macos_mem_available() -> int | None:
    """Use psutil if available, otherwise fall back to sysconf."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        logger.warning("psutil not available, falling back to sysconf.")
    except Exception as e:
        logger.debug(f"psutil failed: {e}")

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size
    except (ValueError, OSError, AttributeError) as e:
        logger.warning(f"sysconf fallback failed: {e}")

    return None


def detect_ram() -> int:
    """Return total available RAM in bytes.

    On Linux: read from /proc/meminfo (MemAvailable, not MemTotal).
    On macOS: use os.sysconf or psutil.
    Fallback: return 0 and log a warning.

    Note: "available" me32ans what the OS can allocate right now, not total physical RAM. This accounts for other
    processes using memeory.
    """
    try:
        result: int | None = None
        if sys.platform.startswith("linux"):
            result = _linux_mem_available()
        elif sys.platform == "darwin":
            result = _macos_mem_available()
        else:
            logger.warning(
                f"Unsupported platform for RAM deteciton {sys.platform}")

        if result is not None:
            return result
    except Exception as e:
        logger.warning(f"Unexpected error detecting RAM: {e}")
    logger.warning("Could not determine available RAM.")
    return 0


def detect_resources() -> dict[str, Any]:
    """One-call summary of all available resources.

    Returns:
        {
            "cpus": 16,
            "gpus": [{"id": 0, "name": "...", "memory_bytes": ...}, ...],
            "ram_bytes": ...,
            "scheduler": "slurm" | "pbs" | "local",
            "hostname": "workstation01",
        }
    """
    return {
        "cpus": detect_cpus(),
        "gpus": detect_gpus(),
        "ram_bytes": detect_ram(),
        "scheduler": "local",
        "hostname": os.uname()[1],
    }


def suggest_workers(
    task_type: str = "io_bound",
    available_cpus: int | None = None,
) -> int:
    """Suggest number of worker threads/processes.

    task_type:
    - "io_bound": More workers than CPUs is fine (threads). Suggest min(32, cpus * 2).
    - "cpu_bound": One worker per CPU. Suggest cpus.
    - "gpu_bound": One worker per GPU. Suggest len(gpus).

    Cap at available_cpus if provided (allows user override).
    """
    if available_cpus is None:
        available_cpus = detect_cpus()
    if task_type == "io_bound":
        return min(32, detect_cpus() * 2)
    elif task_type == "cpu_bound":
        return detect_cpus()
    elif task_type == "gpu_bound":
        return len(detect_gpus())
    else:
        logger.warning(
            f"Wrong task_type option: {task_type}. "
            "Available options are ['io_bound', 'cpu_bound', 'gpu_bound']"
        )
        return available_cpus


def suggest_chunk_size(
    available_ram_bytes: int,
    dtype: str,
    ndim: int,
    max_fraction: float = 0.1,
) -> int:
    """Suggest chunk edge length for Zarr arrays.

    Tries to make each chunk use roughly max_fraction of available RAM divided by expected concurrent reads.

    Returns edge length in pixels (same of all dimensions).

    Example: 750GB RAM, float32, 3D, 10% -> chunks of ~300 per edge.
    """
    # TODO
    raise NotImplementedError
