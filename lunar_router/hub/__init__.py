"""
Lunar Router Hub - Artifact Download Manager.

Download and manage pre-trained weights like NLTK, spaCy, and HuggingFace.

Usage:
    # CLI
    $ lunar-router download weights-mmlu-v1
    $ lunar-router list
    $ lunar-router info weights-mmlu-v1

    # Python API
    >>> import lunar_router
    >>> lunar_router.download("weights-mmlu-v1")
    >>>
    >>> # Or auto-download on first use
    >>> router = lunar_router.load_router()  # Downloads if missing
    >>>
    >>> # List available packages
    >>> lunar_router.hub.list_packages()
    >>>
    >>> # Get info about a package
    >>> lunar_router.hub.info("weights-mmlu-v1")
"""

from .manager import (
    download,
    list_packages,
    info,
    remove,
    path,
    verify,
    Hub,
    LUNAR_DATA_HOME,
    Package,
    PackageIndex,
)

__all__ = [
    # Main functions
    "download",
    "list_packages",
    "info",
    "remove",
    "path",
    "verify",
    # Classes
    "Hub",
    "Package",
    "PackageIndex",
    # Constants
    "LUNAR_DATA_HOME",
]
