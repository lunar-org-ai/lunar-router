"""
Lunar Router Hub Manager - Download and manage artifacts.

Inspired by NLTK, spaCy, and HuggingFace Hub patterns.
Uses HuggingFace Hub as the default source for weights.
"""

import os
import sys
import json
import shutil
import hashlib
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import zipfile
import tarfile

logger = logging.getLogger(__name__)

# HuggingFace Hub configuration
HF_REPO_ID = "pureai-ecosystem/lunar-router-weights"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_ID}"

# ============================================================================
# Configuration
# ============================================================================

# Default data directory (like ~/.nltk_data or ~/.cache/huggingface)
def _get_data_home() -> Path:
    """Get the data home directory, respecting environment variables."""
    # Check environment variable first (like HF_HOME, NLTK_DATA)
    env_path = os.environ.get("LUNAR_DATA_HOME")
    if env_path:
        return Path(env_path)

    # Platform-specific defaults
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
        return base / "lunar_router"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "lunar_router"
    else:
        # Linux/Unix - follow XDG spec
        xdg_data = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        return Path(xdg_data) / "lunar_router"


LUNAR_DATA_HOME = _get_data_home()

# Package index URL (like NLTK's index)
DEFAULT_INDEX_URL = "https://raw.githubusercontent.com/pureai-ecosystem/lunar-router/main/packages/index.json"

# Fallback to bundled index
BUNDLED_INDEX_PATH = Path(__file__).parent / "index.json"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Package:
    """Represents a downloadable package."""

    id: str
    name: str
    version: str
    description: str
    category: str  # "weights", "models", "data", etc.

    # Download source
    source_type: str = "huggingface"  # "huggingface", "url", "s3"
    url: str = ""  # URL for direct download
    hf_repo_id: str = HF_REPO_ID  # HuggingFace repo
    hf_filename: str = ""  # Filename in HF repo
    size_bytes: int = 0
    sha256: Optional[str] = None

    # Archive info
    archive_type: str = "zip"  # "zip", "tar.gz", "tar", "none"

    # Metadata
    author: str = ""
    license: str = "MIT"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # For weights packages
    num_clusters: Optional[int] = None
    embedding_model: Optional[str] = None
    models_profiled: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Package":
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            category=data.get("category", "weights"),
            source_type=data.get("source_type", "huggingface"),
            url=data.get("url", ""),
            hf_repo_id=data.get("hf_repo_id", HF_REPO_ID),
            hf_filename=data.get("hf_filename", f"{data['id']}.zip"),
            size_bytes=data.get("size_bytes", 0),
            sha256=data.get("sha256"),
            archive_type=data.get("archive_type", "zip"),
            author=data.get("author", ""),
            license=data.get("license", "MIT"),
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            num_clusters=data.get("num_clusters"),
            embedding_model=data.get("embedding_model"),
            models_profiled=data.get("models_profiled", []),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "source_type": self.source_type,
            "url": self.url,
            "hf_repo_id": self.hf_repo_id,
            "hf_filename": self.hf_filename,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "archive_type": self.archive_type,
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "num_clusters": self.num_clusters,
            "embedding_model": self.embedding_model,
            "models_profiled": self.models_profiled,
        }

    @property
    def size_human(self) -> str:
        """Human-readable size."""
        if self.size_bytes == 0:
            return "Unknown"
        for unit in ["B", "KB", "MB", "GB"]:
            if self.size_bytes < 1024:
                return f"{self.size_bytes:.1f} {unit}"
            self.size_bytes /= 1024
        return f"{self.size_bytes:.1f} TB"


@dataclass
class PackageIndex:
    """Index of available packages."""

    version: str
    updated: str
    packages: Dict[str, Package]

    @classmethod
    def from_dict(cls, data: dict) -> "PackageIndex":
        packages = {}
        for pkg_data in data.get("packages", []):
            pkg = Package.from_dict(pkg_data)
            packages[pkg.id] = pkg
        return cls(
            version=data.get("version", "1.0.0"),
            updated=data.get("updated", ""),
            packages=packages,
        )

    @classmethod
    def load_bundled(cls) -> "PackageIndex":
        """Load bundled index file."""
        if BUNDLED_INDEX_PATH.exists():
            with open(BUNDLED_INDEX_PATH) as f:
                return cls.from_dict(json.load(f))
        return cls(version="1.0.0", updated="", packages={})

    def get(self, package_id: str) -> Optional[Package]:
        return self.packages.get(package_id)

    def list_by_category(self, category: str) -> List[Package]:
        return [p for p in self.packages.values() if p.category == category]


# ============================================================================
# Hub Class
# ============================================================================

class Hub:
    """
    Lunar Router Hub - manages artifact downloads.

    Similar to NLTK's downloader or HuggingFace Hub.
    """

    def __init__(
        self,
        data_home: Optional[Path] = None,
        index_url: Optional[str] = None,
    ):
        self.data_home = Path(data_home) if data_home else LUNAR_DATA_HOME
        self.index_url = index_url or DEFAULT_INDEX_URL
        self._index: Optional[PackageIndex] = None

    @property
    def index(self) -> PackageIndex:
        """Get package index, loading from remote or bundled."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _load_index(self) -> PackageIndex:
        """Load package index from remote URL or bundled file."""
        # Try remote first
        try:
            req = Request(self.index_url, headers={"User-Agent": "lunar-router"})
            with urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return PackageIndex.from_dict(data)
        except (URLError, HTTPError, json.JSONDecodeError) as e:
            logger.debug(f"Could not fetch remote index: {e}")

        # Fallback to bundled
        return PackageIndex.load_bundled()

    def refresh_index(self) -> PackageIndex:
        """Force refresh the package index."""
        self._index = self._load_index()
        return self._index

    def list_packages(
        self,
        category: Optional[str] = None,
        installed_only: bool = False,
    ) -> List[Package]:
        """List available packages."""
        packages = list(self.index.packages.values())

        if category:
            packages = [p for p in packages if p.category == category]

        if installed_only:
            packages = [p for p in packages if self.is_installed(p.id)]

        return sorted(packages, key=lambda p: p.id)

    def get_package(self, package_id: str) -> Optional[Package]:
        """Get package info by ID."""
        return self.index.get(package_id)

    def path(self, package_id: str) -> Path:
        """Get local path for a package."""
        return self.data_home / package_id

    def is_installed(self, package_id: str) -> bool:
        """Check if package is installed."""
        pkg_path = self.path(package_id)
        manifest = pkg_path / "manifest.json"
        return manifest.exists()

    def download(
        self,
        package_id: str,
        force: bool = False,
        quiet: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Download a package.

        Args:
            package_id: Package ID to download.
            force: Force re-download even if installed.
            quiet: Suppress output.
            progress_callback: Callback for progress updates (bytes_downloaded, total_bytes).

        Returns:
            Path to installed package.
        """
        pkg = self.get_package(package_id)
        if pkg is None:
            available = ", ".join(self.index.packages.keys())
            raise ValueError(f"Unknown package '{package_id}'. Available: {available}")

        pkg_path = self.path(package_id)

        # Check if already installed
        if self.is_installed(package_id) and not force:
            if not quiet:
                print(f"Package '{package_id}' is already installed at {pkg_path}")
            return pkg_path

        # Download dependencies first
        for dep_id in pkg.dependencies:
            if not self.is_installed(dep_id):
                if not quiet:
                    print(f"Installing dependency: {dep_id}")
                self.download(dep_id, quiet=quiet)

        if not quiet:
            print(f"Downloading {pkg.name} ({pkg.version})...")
            if pkg.size_bytes > 0:
                print(f"  Size: {pkg.size_human}")
            print(f"  Source: {pkg.source_type}")

        # Create temp directory for download
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Download file based on source type
            archive_path = tmp_path / f"package.{pkg.archive_type}"

            if pkg.source_type == "huggingface":
                self._download_from_huggingface(
                    pkg.hf_repo_id,
                    pkg.hf_filename,
                    archive_path,
                    quiet=quiet,
                )
            elif pkg.source_type == "url":
                self._download_file(
                    pkg.url,
                    archive_path,
                    expected_sha256=pkg.sha256,
                    quiet=quiet,
                    progress_callback=progress_callback,
                )
            else:
                raise ValueError(f"Unknown source type: {pkg.source_type}")

            # Extract
            if pkg.archive_type != "none":
                if not quiet:
                    print("  Extracting...")
                extract_path = tmp_path / "extracted"
                self._extract_archive(archive_path, extract_path, pkg.archive_type)
            else:
                extract_path = archive_path

            # Move to final location
            if pkg_path.exists():
                shutil.rmtree(pkg_path)
            pkg_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle single directory in archive
            extracted_items = list(extract_path.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                shutil.move(str(extracted_items[0]), str(pkg_path))
            else:
                shutil.move(str(extract_path), str(pkg_path))

            # Save manifest
            manifest = {
                "package": pkg.to_dict(),
                "installed_at": str(pkg_path),
            }
            with open(pkg_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

        if not quiet:
            print(f"✓ Installed {pkg.name} to {pkg_path}")

        return pkg_path

    def _download_file(
        self,
        url: str,
        dest: Path,
        expected_sha256: Optional[str] = None,
        quiet: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Download a file with progress bar."""
        req = Request(url, headers={"User-Agent": "lunar-router"})

        try:
            with urlopen(req, timeout=60) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192

                hasher = hashlib.sha256() if expected_sha256 else None

                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if hasher:
                            hasher.update(chunk)

                        if progress_callback:
                            progress_callback(downloaded, total_size)
                        elif not quiet and total_size > 0:
                            pct = (downloaded / total_size) * 100
                            bar_len = 30
                            filled = int(bar_len * downloaded / total_size)
                            bar = "=" * filled + "-" * (bar_len - filled)
                            print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)

                if not quiet and total_size > 0:
                    print()  # New line after progress bar

                # Verify checksum
                if hasher and expected_sha256:
                    actual_sha256 = hasher.hexdigest()
                    if actual_sha256 != expected_sha256:
                        raise ValueError(
                            f"Checksum mismatch! Expected {expected_sha256}, got {actual_sha256}"
                        )

        except HTTPError as e:
            raise RuntimeError(f"Download failed: HTTP {e.code} - {e.reason}")
        except URLError as e:
            raise RuntimeError(f"Download failed: {e.reason}")

    def _download_from_huggingface(
        self,
        repo_id: str,
        filename: str,
        dest: Path,
        quiet: bool = False,
    ) -> None:
        """Download a file from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub package required for HuggingFace downloads. "
                "Install with: pip install huggingface_hub"
            )

        if not quiet:
            print(f"  From: huggingface.co/{repo_id}/{filename}")

        try:
            # Download to HF cache first
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
            )

            # Copy to destination
            shutil.copy2(cached_path, dest)

            if not quiet:
                print("  ✓ Downloaded from HuggingFace Hub")

        except Exception as e:
            raise RuntimeError(f"HuggingFace download failed: {e}")

    def _extract_archive(
        self,
        archive_path: Path,
        dest_path: Path,
        archive_type: str,
    ) -> None:
        """Extract an archive."""
        dest_path.mkdir(parents=True, exist_ok=True)

        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_path)
        elif archive_type in ("tar.gz", "tgz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest_path)
        elif archive_type == "tar":
            with tarfile.open(archive_path, "r") as tf:
                tf.extractall(dest_path)
        else:
            raise ValueError(f"Unknown archive type: {archive_type}")

    def remove(self, package_id: str, quiet: bool = False) -> bool:
        """Remove an installed package."""
        pkg_path = self.path(package_id)

        if not pkg_path.exists():
            if not quiet:
                print(f"Package '{package_id}' is not installed.")
            return False

        shutil.rmtree(pkg_path)

        if not quiet:
            print(f"✓ Removed {package_id}")

        return True

    def verify(self, package_id: str) -> bool:
        """Verify package integrity."""
        pkg = self.get_package(package_id)
        pkg_path = self.path(package_id)

        if not pkg_path.exists():
            return False

        # Basic structure check
        manifest_path = pkg_path / "manifest.json"
        if not manifest_path.exists():
            return False

        # For weights packages, check required files
        if pkg and pkg.category == "weights":
            clusters_dir = pkg_path / "clusters"
            profiles_dir = pkg_path / "profiles"
            if not clusters_dir.exists() or not profiles_dir.exists():
                return False

        return True

    def info(self, package_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a package."""
        pkg = self.get_package(package_id)
        if pkg is None:
            return None

        info = pkg.to_dict()
        info["installed"] = self.is_installed(package_id)
        info["local_path"] = str(self.path(package_id)) if info["installed"] else None
        info["verified"] = self.verify(package_id) if info["installed"] else None

        return info


# ============================================================================
# Module-level API (like NLTK)
# ============================================================================

# Global hub instance
_hub: Optional[Hub] = None


def _get_hub() -> Hub:
    """Get the global hub instance."""
    global _hub
    if _hub is None:
        _hub = Hub()
    return _hub


def download(
    package_id: str,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """
    Download a package.

    Args:
        package_id: Package ID (e.g., "weights-mmlu-v1").
        force: Force re-download.
        quiet: Suppress output.

    Returns:
        Path to installed package.

    Example:
        >>> import lunar_router
        >>> lunar_router.download("weights-mmlu-v1")
        Downloading weights-mmlu-v1 (1.0.0)...
          Size: 1.2 MB
          [==============================] 100.0%
          Extracting...
        ✓ Installed weights-mmlu-v1 to ~/.local/share/lunar_router/weights-mmlu-v1
    """
    return _get_hub().download(package_id, force=force, quiet=quiet)


def list_packages(
    category: Optional[str] = None,
    installed_only: bool = False,
) -> List[Package]:
    """
    List available packages.

    Args:
        category: Filter by category ("weights", "models", etc.).
        installed_only: Only show installed packages.

    Returns:
        List of packages.

    Example:
        >>> import lunar_router
        >>> for pkg in lunar_router.list_packages():
        ...     status = "✓" if pkg.installed else " "
        ...     print(f"[{status}] {pkg.id}: {pkg.description}")
    """
    return _get_hub().list_packages(category=category, installed_only=installed_only)


def info(package_id: str) -> Optional[Dict[str, Any]]:
    """
    Get info about a package.

    Example:
        >>> lunar_router.info("weights-mmlu-v1")
        {'id': 'weights-mmlu-v1', 'version': '1.0.0', 'installed': True, ...}
    """
    return _get_hub().info(package_id)


def remove(package_id: str, quiet: bool = False) -> bool:
    """Remove an installed package."""
    return _get_hub().remove(package_id, quiet=quiet)


def path(package_id: str) -> Path:
    """Get path to a package (installed or not)."""
    return _get_hub().path(package_id)


def verify(package_id: str) -> bool:
    """Verify package integrity."""
    return _get_hub().verify(package_id)
