"""Server settings, fixed when the app is created."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Server settings.

    Attributes:
        roots: Directories the server may browse and open. Paths outside them are refused. Stored resolved.
        cache_bytes: Memory limit for cached frames, pyramid levels and encoded tiles.
        job_workers: Background jobs that may run at once. Further jobs wait in a queue.
        tile_size: Tile edge length in pixels.

    Raises:
        NotADirectoryError: If a root isn't an existing directory.
        ValueError: For no roots or out-of-range numbers.
    """

    roots: tuple[Path, ...]
    cache_bytes: int = 2 * 1024**3
    job_workers: int = 2
    tile_size: int = 512

    def __post_init__(self) -> None:
        roots = tuple(Path(root).expanduser().resolve() for root in self.roots)
        if not roots:
            raise ValueError("Give at least one root directory.")
        for root in roots:
            if not root.is_dir():
                raise NotADirectoryError(f"Root is not a directory: {root}")
        if self.job_workers < 1 or self.tile_size < 64 or self.cache_bytes < 0:
            raise ValueError("job_workers must be ≥ 1, tile_size ≥ 64 and cache_bytes ≥ 0.")
        object.__setattr__(self, "roots", roots)
