"""Configuration management

Provides a Config class for loading, validating, and accessing YAML configuration files using dot-notation keys.
Also exposes a module-level load() function.

Usage:
    # Class interface (preferred):
    config = Config("config.yaml")
    energy = config.get("reconstruction.paganin.energy")
    gpu_ids = config["hardware.gpu_ids"]

    # Check existence before accessing
    if config.has("segmentation.model"):
        model = config.require("segmentation.model")

    # Backward compatible function:
    cfg = load("config.yaml")  # Returns raw dict
"""

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Union

import yaml

from pd_xray.core.exceptions import ConfigError, ConfigKeyError, ConfigTypeError
from pd_xray.core.logging import get_logger

logger = get_logger(__name__)


class Config:
    """Configuration manager.

    Loads a YAML file and provides:
    - Dot-notation key access (e.g. 'reconstruction.paganin.energy')
    - Optional type validation on access
    - Required-key enforcement
    - In-memory key override (without touching the file)
    - Merging of a second config on top of this one
    - Iteration over all flattened dot-notation keys
    - A read-only raw-dict view for interop with legacy code

    Attributes:
        config_path: Path to the loaded YAML file.

    Example:
        >>> config = Config("config.yaml")
        >>> energy =  config.get("reconstruction.paganin.energy", default=25.0)
        >>> gpu_ids = config["hardware.gpu_ids"]
        >>> config.set("reconstruction.paganin.energy", 30.0)
        >>> for key, value in config.items():
                print(key, value)
    """

    def __init__(self, config_path: str | Path) -> None:
        """Load YAML configuration from disk.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ConfigError: If the file is not valid YAML or does not parse to a dict at the top level.
        """
        self.config_path = Path(config_path)
        self._config = self._load()

    def _load(self) -> dict[str, Any]:
        """Read and parse the YAML file.

        Returns:
            Top-level configuration dictionary.

        Raises:
            FileNotFoundError: Config file not found.
            ConfigError: YAML parse error or non-dict top level.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}"
            )
        try:
            with open(self.config_path) as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Failed to parse YAML config '{self.config_path}': {e}",
            ) from e

        if raw is None:
            raise ConfigError(
                f"Config file is empty: {self.config_path}",
            )

        if not isinstance(raw, dict):
            raise ConfigError(
                f"Config file must be a YAML mapping at the top level, "
                f" got {type(raw).__name__}: {self.config_path}",
            )
        logger.debug(f"Loaded config from {self.config_path} "
                     f"({len(raw)} top-level keys)")
        return raw

    def reload(self) -> None:
        """Re-read the YAML file from disk, discarding any in-memory overrides.

        Useful when the file has been edited externally during a long session.
        """
        self._config = self._load()
        logger.info(f"Reloaded config from {self.config_path}")

    def _resolve(
        self,
        key: str,
        raise_on_missing: bool = False,
    ) -> tuple[bool, Any]:
        """Walk the config dict following dot-notation key.

        Args:
            key: Dot-separated key, e.g. 'reconstruction.paganin.energy'.
            raise_on_missing: If True, raise ConfigKeyError when the key is not found. If False, return (False, None).

        Returns:
             (found: bool, value: Any)

        Raises:
            ConfigKeyError: If raise_on_missing is True and key not found.
            ConfigTypeError: If a mid-path segment is not a dict
                (e.g. 'a.b.c' but config['a']['b'] is an integer).
        """
        if not key:
            raise ConfigKeyError("Config key must not be empty")

        parts = key.split(".")
        current: Any = self._config

        for depth, part in enumerate(parts):
            if not isinstance(current, dict):
                traversed = ".".join(parts[:depth])
                raise ConfigTypeError(
                    f"Cannot traverse '{key}': segment '{traversed}' "
                    f"is {type(current).__name__}, not a dict.",
                )

            if part not in current:
                if raise_on_missing:
                    raise ConfigKeyError(
                        f"Required config key '{key}' not found. "
                        f"Missing segment: '{part}'.",
                    )
                return False, None
            current = current[part]
        return True, current

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """Get a configuration value by dot-notation key.

        Args:
            key: Dot-separated key, e.g. "reconstruction.paganin.energy".
            default: Value to return when the key is not present. Defaults to None.

        Returns:
            Configuration value, or default if not found.

        Raises:
            ConfigTypeError: If a mid-path key is not a dict.

        Example:
            >>> energy = config.get("reconstruction.paganin.energy", 25.0)
            >>> gpu_ids = config.get("hardware.gpu_ids", [0])
        """
        found, value = self._resolve(key, raise_on_missing=False)
        return value if found else default

    def require(self, key: str) -> Any:  # noqa: ANN401
        """Get a config value, raising an error if the key is absent.

        Use this for keys that are truly non-negotiable.

        Args:
            key: Dot-separated key.

        Returns:
            Configuration value.

        Raises:
            ConfigKeyError: If the key is not present.
            ConfigTypeError: If a mid-path key is not a dict

        Example:
            >>> base_path = config.require("data.raw_edf.base_path")
        """
        _, value = self._resolve(key, raise_on_missing=True)
        return value

    def get_typed(
        self,
        key: str,
        expected_type: type,
        default: Any = None  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Get a config value and validate its type.

        Args:
            key: Dot-separated key.
            expected_type: The Python type the value must be an instance of.
            default: Returned when the key is absent. Not type-checked.

        Returns:
            Configuration value (type-validated), or default.

        Raises:
            ConfigTypeError: If the value exists but is the wrong type.
            ConfigTypeError: If a mid-path key is not a dict.

        Example:
            >>> energy = config.get_typed("reconstruction.paganin.energy", float)
            >>> gpu_ids = config.get_typed("hardware.gpu_ids", list)
        """
        found, value = self._resolve(key, raise_on_missing=False)

        if not found:
            return default

        if not isinstance(value, expected_type):
            raise ConfigTypeError(
                f"Config key '{key}' expected {expected_type.__name__}, "
                f"got {type(value).__name__} = {value!r}.",
            )

        return value

    def has(self, key: str) -> bool:
        """Check whether a key exists in the configuration.

        Args:
            key: Dot-separated key.

        Returns:
            True if the key exists, False otherwise.

        Example:
            >>> if config.has("segmentation.model"):
            ...    model = config.require("segmentation.model")
        """
        found, _ = self._resolve(key, raise_on_missing=False)
        return found

    def get_section(self, key: str) -> "Config":
        """Return a sub-tree of the config as a new Config-like view.

        Useful for passing just a relevant section to a subsystem without giving it the whole config.

        Args:
            key: Dot-separated key pointing to a dict-valued node.

        Returns:
            A new Config instance wrapping the sub-dict. Changes to the sub-config are isolated (deep copy).

        Raises:
            ConfigKeyError: If the key does not exist.
            ConfigTypeError: If the resolved value is not a dict.

        Example:
            >>> recon_cfg = config.get_section("reconstruction")
            >>> energy = recon_cfg.get("paganin.energy")
        """
        _, value = self._resolve(key, raise_on_missing=True)

        if not isinstance(value, dict):
            raise ConfigTypeError(
                f"Config key '{key}' does not point to a section (dict), "
                f"got {type(value).__name__}.",
            )

        # Build a minimal Config wrapper around the sub-dict
        sub = object.__new__(Config)
        sub.config_path = self.config_path  # Keep origin for error messages
        sub._config = copy.deepcopy(value)
        return sub

    def set(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Set a configuration value in memory (does not modify the file).

        Intermediate dicts are created automatically if they don't exist.

        Args:
            key: Dot-separated key.
            value: New value to store.

        Example:
            >>> config.set("reconstruction.paganin.energy", 30.0)
            >>> config.set("new.nested.key", "hello")
        """
        parts = key.split(".")
        current = self._config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            if not isinstance(current[part], dict):
                raise ConfigTypeError(
                    f"Cannot set '{key}': intermediate key '{part}' "
                    f"is {type(current[part]).__name__}, not a dict.",
                )
            current = current[part]
        current[parts[-1]] = value
        logger.debug(f"Config override: {key} = {value!r}")

    def merge(self, other: Union["Config", dict[str, Any]]) -> None:
        """Deep-merge another Config or plain dict into this one.

        Values in 'other' take precedence. Nested dicts are merged recursively; scalars and lists are replaced
        wholesale.

        Useful for applying experiment-specific overrides on top of a base configuration without touching the YAML file

        Args:
            other: Config instance or plain dict to merge from.

        Example:
            >>> base = Config("config.yaml")
            >>> overrides = {"reconstruction": {"paganin": {"energy": 30.0}}}
            >>> base.merge(overrides)
        """
        source = other._config if isinstance(other, Config) else other
        self._deep_merge(self._config, source)
        logger.debug("Merged additional config on top of base config")

    @staticmethod
    def _deep_merge(
        base: dict[str, Any],
            override: dict[str, Any],
    ) -> None:
        """Recursively merge override into base in-place."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value

    def keys(self, prefix: str = "") -> list[str]:
        """Return all dot-notation keys in the config (flattened).

        Args:
            prefix: If given, only keys under this prefix are returned.

        Returns:
            Sorted list of dot-notation key strings.

        Example:
            >>> config.keys("reconstruction.paganin")
            ['reconstruction.paganin.delta_beta_ratio',
             'reconstruction.paganin.distance',
             'reconstruction.paganin.energy',
             'reconstruction.paganin.pixel_size']
        """
        if prefix:
            found, subtree = self._resolve(prefix, raise_on_missing=False)
            if not found or not isinstance(subtree, dict):
                return []
            flat = list(self._flatten(subtree, parent=prefix))
        else:
            flat = list(self._flatten(self._config))

        return sorted(flat)

    def items(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        """Iterate over (dot-notation-key, value) pairs for all leaf nodes.
        Args:
            prefix: If given, only keys under this prefix are yielded.

        Yields:
            (key, value) tuples for each leaf in the config tree.

        Example:
            >>> for key, value in config.items("hardware"):
            ...     print(f"{key}: {value}")
        """
        for key in self.keys(prefix=prefix):
            _, value = self._resolve(key)
            yield key, value

    @staticmethod
    def _flatten(
        d: dict[str, Any],
        parent: str = ""
    ) -> Iterator[str]:
        """Yield flattened dot-notation keys for all leaf nodes."""
        for key, value in d.items():
            full_key = f"{parent}.{key}" if parent else key
            if isinstance(value, dict):
                yield from Config._flatten(value, parent=full_key)
            else:
                yield full_key

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Dict-like access. Raises ConfigKeyError if key is absent.

        Prefer .get() for optional keys, .require() for required keys.
        This method behaves like .require() to avoid silent None bugs.

        Example:
            >>> gpu_ids = config["hardware.gpu_ids"]
        """
        return self.require(key)

    def __contains__(self, key: object) -> bool:
        """Support `'key' in config` checks.

        Example:
            >>> if "segmentation.model" in config:
            ...     pass
        """
        if not isinstance(key, str):
            return False
        return self.has(key)

    def __repr__(self) -> str:
        top_keys = list(self._config.keys())
        return (
            f"Config(path='{self.config_path}', "
            f"top_level_keys={top_keys})"
        )

    @property
    def as_dict(self) -> dict[str, Any]:
        """Return a deep copy of the raw configuration dictionary.

        Use when passing config to a library that expects a plain dict.
        Changes to the returned dict do NOT affect this Config instance.

        Returns:
            Deep copy of the full configuration.
        """
        return copy.deepcopy(self._config)


def load(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return the raw dictionary.

    Backward-compatible function for code that predates the Config class.
    New code should use Config() directly.

    Args:
        config_path: Path to YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If the file is not valid YAML.
    """
    return Config(config_path).as_dict
