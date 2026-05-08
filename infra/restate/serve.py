"""Run the Restate Python service container.

  uv run --extra infra python -m infra.restate.serve

Convention:
  9080  this service (Restate connects here to invoke handlers)
  8080  Restate ingress (you call this to invoke handlers from outside)
  9070  Restate admin (registration, listing invocations, etc.)
"""

from __future__ import annotations

import asyncio
import logging
import os

from hypercorn import Config
from hypercorn.asyncio import serve

from infra.restate.services import app

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    host = os.environ.get("RESTATE_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("RESTATE_SERVICE_PORT", "9080"))

    config = Config()
    config.bind = [f"{host}:{port}"]
    # HTTP/2 cleartext — Restate uses gRPC-style bidi streams.
    config.h2_max_concurrent_streams = 1024

    logger.info("restate compactor service listening on http://%s:%d", host, port)
    logger.info(
        "register with: curl http://localhost:9070/deployments "
        "--json '{\"uri\": \"http://host.docker.internal:%d\"}'",
        port,
    )
    asyncio.run(serve(app, config))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
