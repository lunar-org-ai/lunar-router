"""
Weights management for OpenTracy.

Provides functions to download, locate, and manage pre-trained routing weights.
Delegates to the Hub module for actual download operations.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .hub import Hub, OPENTRACY_DATA_HOME, download as hub_download, path as hub_path
from .hub.manager import _get_hub

logger = logging.getLogger(__name__)

# Name mappings: short name -> hub package ID
_WEIGHTS_ALIASES = {
    "default": "weights-default",
    "mmlu": "weights-mmlu-v1",
    "mmlu-v1": "weights-mmlu-v1",
}

# Package IDs that resolve to the same bundled weights directory.
_BUNDLED_PACKAGE_ALIASES = {
    "weights-default": "weights-mmlu-v1",
}

_BUNDLED_WEIGHTS_ROOT = Path(__file__).parent / "_bundled_weights"


def _resolve_package_id(name: str) -> str:
    """Resolve a weights name to a hub package ID."""
    if name in _WEIGHTS_ALIASES:
        return _WEIGHTS_ALIASES[name]
    # If already a valid package ID, use as-is
    if name.startswith("weights-"):
        return name
    # Try prefixing with "weights-"
    return f"weights-{name}"


def _bundled_path(package_id: str) -> Optional[Path]:
    """Return the bundled weights path for a package, if shipped in the wheel."""
    candidate = _BUNDLED_PACKAGE_ALIASES.get(package_id, package_id)
    path = _BUNDLED_WEIGHTS_ROOT / candidate
    if path.exists() and (path / "clusters").exists():
        return path
    return None


@dataclass
class WeightsConfig:
    """Configuration for a weights package."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    num_clusters: int = 100
    embedding_model: str = "all-MiniLM-L6-v2"
    models: List[str] = field(default_factory=list)
    source_type: str = "huggingface"
    url: str = ""
    hf_repo_id: str = ""
    hf_filename: str = ""
    sha256: Optional[str] = None


# Registry of known weights configurations
WEIGHTS_REGISTRY: Dict[str, WeightsConfig] = {
    "weights-mmlu-v1": WeightsConfig(
        name="MMLU Weights v1",
        version="1.0.0",
        description="Pre-trained routing weights on MMLU benchmark with 100 clusters",
        num_clusters=100,
        embedding_model="all-MiniLM-L6-v2",
        models=[
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
            "mistral-large-latest", "mistral-small-latest", "codestral-latest",
            "ministral-8b-latest", "ministral-3b-latest", "pixtral-12b-2409",
        ],
        source_type="huggingface",
        hf_repo_id="diogovieira/opentracy-weights",
        hf_filename="weights-mmlu-v1.zip",
    ),
    "weights-default": WeightsConfig(
        name="Default Weights",
        version="1.0.0",
        description="Default routing weights (alias for weights-mmlu-v1)",
        num_clusters=100,
        embedding_model="all-MiniLM-L6-v2",
        models=[
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
            "mistral-large-latest", "mistral-small-latest",
        ],
        source_type="huggingface",
        hf_repo_id="diogovieira/opentracy-weights",
        hf_filename="weights-mmlu-v1.zip",
    ),
}


def download_weights(
    name: str = "default",
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download pre-trained weights.

    Args:
        name: Weights name (e.g., "default", "mmlu-v1", "weights-mmlu-v1").
        force: Force re-download even if already installed.
        verbose: Print progress information.

    Returns:
        Path to downloaded weights directory.

    Example:
        >>> from opentracy.weights import download_weights
        >>> path = download_weights("default")
        >>> print(path)
    """
    package_id = _resolve_package_id(name)

    # Prefer bundled weights shipped in the wheel — zero network, zero auth.
    if not force:
        bundled = _bundled_path(package_id)
        if bundled is not None:
            if verbose:
                print(f"Using bundled weights at {bundled}")
            return bundled

    return hub_download(package_id, force=force, quiet=not verbose)


def download_from_huggingface(
    repo_id: str,
    filename: str,
    dest: Optional[Path] = None,
    quiet: bool = False,
) -> Path:
    """
    Download weights directly from a HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID (e.g., "diogovieira/opentracy-weights").
        filename: Filename in the repository.
        dest: Destination path. If None, uses default data directory.
        quiet: Suppress output.

    Returns:
        Path to downloaded file.
    """
    hub = _get_hub()
    if dest is None:
        dest = hub.data_home / "custom" / filename

    dest.parent.mkdir(parents=True, exist_ok=True)
    hub._download_from_huggingface(repo_id, filename, dest, quiet=quiet)
    return dest


def download_from_url(
    url: str,
    dest: Optional[Path] = None,
    sha256: Optional[str] = None,
    quiet: bool = False,
) -> Path:
    """
    Download weights from a direct URL.

    Args:
        url: URL to download from.
        dest: Destination path. If None, uses default data directory.
        sha256: Expected SHA256 checksum for verification.
        quiet: Suppress output.

    Returns:
        Path to downloaded file.
    """
    hub = _get_hub()
    if dest is None:
        filename = url.rsplit("/", 1)[-1]
        dest = hub.data_home / "custom" / filename

    dest.parent.mkdir(parents=True, exist_ok=True)
    hub._download_file(url, dest, expected_sha256=sha256, quiet=quiet)
    return dest


def download_from_s3(
    bucket: str,
    key: str,
    dest: Optional[Path] = None,
    quiet: bool = False,
) -> Path:
    """
    Download weights from Amazon S3.

    Args:
        bucket: S3 bucket name.
        key: S3 object key.
        dest: Destination path. If None, uses default data directory.
        quiet: Suppress output.

    Returns:
        Path to downloaded file.

    Raises:
        ImportError: If boto3 is not installed.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 downloads. "
            "Install with: pip install opentracy[s3]"
        )

    if dest is None:
        filename = key.rsplit("/", 1)[-1]
        dest = OPENTRACY_DATA_HOME / "custom" / filename

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"Downloading from s3://{bucket}/{key}...")

    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(dest))

    if not quiet:
        print(f"Downloaded to {dest}")

    return dest


def get_weights_path(name: str = "default") -> Path:
    """
    Get the local path for a weights package.

    Args:
        name: Weights name (e.g., "default", "mmlu-v1").

    Returns:
        Path to the weights directory (may not exist if not downloaded).
    """
    package_id = _resolve_package_id(name)
    bundled = _bundled_path(package_id)
    if bundled is not None:
        return bundled
    return hub_path(package_id)


def list_available_weights() -> List[WeightsConfig]:
    """
    List all available weights configurations.

    Returns:
        List of WeightsConfig objects.
    """
    return list(WEIGHTS_REGISTRY.values())
