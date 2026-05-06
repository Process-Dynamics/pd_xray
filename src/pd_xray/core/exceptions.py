class PDCoreError(Exception):
    """Main core exception class."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class ConfigError(PDCoreError):
    """Raised for configuration loading or access errors"""
    pass


class ConfigKeyError(ConfigError, KeyError):
    """Raised when a required key is missing.

    Inherits from KeyError so callers can catch either.
    """
    pass


class ConfigTypeError(ConfigError, TypeError):
    """Raised when a config value has an unexpected type.

    E.g. a mid-path key is a scalar rather than a dict, or require_type() finds a type mismatch
    """
    pass

class ConfigFileNotFoundError(ConfigError, FileNotFoundError):
    """Raised when a config file is not found."""
    pass

class ConfigValidationError(ConfigError):
    """Raised when a config value fails validation."""
    pass

class EnvVarNotFoundError(PDCoreError):
    """Raised when an expected environment variable is not found."""
    pass

class ResourceDetectionError(PDCoreError):
    pass

class CPUDetectionError(PDCoreError):
    pass


class GPUError(PDCoreError):
    """Raised for GPU related errors"""
    pass


class GPUTorchError(GPUError):
    """Raised when torch fails"""
    pass


class SchemaError(PDCoreError):
    pass
