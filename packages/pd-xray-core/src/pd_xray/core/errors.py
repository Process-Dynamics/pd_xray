"""Base exceptions shared by all pd_xray packages."""

import importlib
from types import ModuleType


class PdXrayError(Exception):
    """Base class for all pd_xray errors."""


class OperationCancelledError(PdXrayError):
    """Raised when a long operation stops because its `CancelToken` was cancelled."""


class MissingDependencyError(PdXrayError, ImportError):
    """Raised when an optional dependency (a package extra) is not installed."""


def import_optional(module: str, extra: str, package: str) -> ModuleType:
    """Import an optional dependency, or raise a `MissingDependencyError` that says how to install it.

    Args:
        module: Module to import, e.g. "paramiko".
        extra: Extra that provides it, e.g. "sftp".
        package: Distribution that defines the extra, e.g. "pd-xray-storage".

    Returns:
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"'{module}' is required for this feature. Install it with: pip install \"{package}[{extra}]\""
        ) from exc
