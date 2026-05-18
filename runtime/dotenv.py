"""Tiny .env loader — sets os.environ from a KEY=VAL file.

Called explicitly from server startup + CLI entry, NOT at package
import time, so tests stay isolated. Quotes around values are
stripped. Lines starting with ``#`` are comments. Existing env vars
take precedence (so a shell-exported value beats the .env file).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_env(path: Optional[Path] = None) -> dict[str, str]:
    """Load KEY=VAL pairs from .env. Returns the dict of values set.

    Does not raise if the file is missing. Existing env vars are
    preserved (.env can't override what's already exported).
    """
    p = Path(path) if path else Path(".env")
    if not p.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded
