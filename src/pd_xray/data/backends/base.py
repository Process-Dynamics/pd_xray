from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract base class for storage systems."""

    @abstractmethod
    def list_files(
        self,
        prefix: str = "",
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[str]:
        """List files matching a glob pattern under prefix.

        Args:
            prefix    : Directory or S3 prefix to search under.
            pattern   : Glob pattern (e.g., "proj_*.edf"). Applied to filenames only.
            recursive : Whether to search subdirectories.
        
        Returns:
            Sorted list of full paths/keys matching the pattern.
        """
        ...

    @abstractmethod
    def read_bytes(
        self,
        path: str,
    ) -> bytes:
        """Read entire file contents as bytes.

        Use only for small files (metadata, configs). For large files, use read_file_to_local() and let the format
        reader handle streaming.
        """
        ...

    @abstractmethod
    def read_file_to_local(
        self,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Download a single file to local filesystem.

        Args:
            remote_path : Path/key on the remote storage.
            local_path  : Destination path on local filesystem.
        """
        ...

    @abstractmethod
    def write_file_from_local(
        self,
        local_path: str,
        remote_path: str,
    ) -> None:
        """Upload a single file from local filesystem.

        Args:
            local_path  : Source path on local filesystem.
            remote_path : Destination path/key on the remote storage.
        """
        ...
    
    @abstractmethod
    def exists(
        self,
        path: str,
    ) -> bool:
        """Check if a file or prefix exists."""
        ...
    
    @abstractmethod
    def file_size(
        self,
        path: str
    ) -> int:
        """Return file size in bytes."""
        ...
