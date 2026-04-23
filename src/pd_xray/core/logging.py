import logging
import sys


def setup_logging(
    level: str | int = logging.INFO,
    format_string: str | None = None,
    date_format: str | None = None,
) -> None:
    """
    Configure global logging settings for the entire application.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string
        date_format: Custom date format
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"

    if date_format is None:
        date_format = "%H:%M:%S"

    # Remove any existing handlers to avoid duplicates
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(level=level, format=format_string,
                        datefmt=date_format, stream=sys.stdout)


def get_logger(name: str) -> logging.Logger:
    # Ensure logging is configured
    if not logging.getLogger().handlers:
        setup_logging()

    return logging.getLogger(name)


# Optional: Pre-configured loggers for specific modules
def get_reconstruction_logger() -> logging.Logger:
    """Get logger specifically for reconstruction operations."""
    return get_logger("tomography.reconstruction")


def get_preprocessing_logger() -> logging.Logger:
    """Get logger specifically for preprocessing operations."""
    return get_logger("tomography.preprocessing")


def get_parameter_logger() -> logging.Logger:
    """Get logger specifically for parameter operations."""
    return get_logger("tomography.parameters")
