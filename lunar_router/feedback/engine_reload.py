"""
Trigger Go engine to reload weights after auto-training.

Calls POST /v1/weights/reload on the Go engine, which atomically
swaps in the new Psi vectors with zero-downtime (in-flight requests
continue on old weights, new requests use updated weights).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_ENGINE_URL = "http://localhost:8080"


@dataclass
class ReloadResult:
    """Result of a weights reload request."""

    success: bool
    message: str
    old_generation: Optional[int] = None
    new_generation: Optional[int] = None
    num_clusters: Optional[int] = None
    num_models: Optional[int] = None


def reload_engine_weights(
    weights_path: Optional[str] = None,
    engine_url: Optional[str] = None,
    timeout: float = 30.0,
) -> ReloadResult:
    """
    Tell the Go engine to reload weights from disk.

    Args:
        weights_path: Path to weights dir. If None, engine uses its configured path.
        engine_url: Go engine URL. Defaults to LUNAR_ENGINE_URL or localhost:8080.
        timeout: HTTP request timeout in seconds.

    Returns:
        ReloadResult with generation info.
    """
    import urllib.request
    import urllib.error
    import json

    url = engine_url or os.environ.get("LUNAR_ENGINE_URL", DEFAULT_ENGINE_URL)
    endpoint = f"{url}/v1/weights/reload"

    body = {}
    if weights_path:
        body["weights_path"] = str(weights_path)

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())

        logger.info(
            f"Engine weights reloaded: generation {result.get('old_generation')} → "
            f"{result.get('new_generation')}, {result.get('num_models')} models"
        )

        return ReloadResult(
            success=True,
            message=result.get("message", "ok"),
            old_generation=result.get("old_generation"),
            new_generation=result.get("new_generation"),
            num_clusters=result.get("num_clusters"),
            num_models=result.get("num_models"),
        )

    except urllib.error.URLError as e:
        msg = f"Failed to reach Go engine at {endpoint}: {e}"
        logger.warning(msg)
        return ReloadResult(success=False, message=msg)

    except Exception as e:
        msg = f"Engine reload failed: {e}"
        logger.error(msg)
        return ReloadResult(success=False, message=msg)


def check_engine_health(engine_url: Optional[str] = None) -> bool:
    """Check if the Go engine is running and healthy."""
    import urllib.request
    import urllib.error

    url = engine_url or os.environ.get("LUNAR_ENGINE_URL", DEFAULT_ENGINE_URL)

    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False
