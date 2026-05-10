"""Router-specific error types.

These are the only failure modes router/uniroute.py + router/config_io.py
raise. Callers (FastAPI handlers, harness proposer, MCP tools) convert
them to user-facing errors (503, 409, "blocked: not_enough_data", etc).
"""


class RouterError(Exception):
    """Base for all router errors."""


class RouterColdStartError(RouterError):
    """Raised when a routing decision is requested before a config exists.

    Two ways to hit this:
    - LLMRegistry has no profiles (no models known to the router).
    - ClusterAssigner has zero centroids (no fit yet).

    Caller decides what to do — typically fall back to agent.models.default
    or surface a 503 router_cold_start to the API client.
    """


class RouterConfigNotFoundError(RouterError):
    """Raised when config_io can't find a router_config artifact.

    Either versions/router_config_current pointer is missing, or the file
    it references doesn't exist on disk.
    """


class RouterConfigInvalidError(RouterError):
    """Raised when a router_config_<n>.json fails schema validation."""


class NotEnoughDataError(RouterError):
    """Corpus too small to fit a cluster model. Wait for more traces.

    Raised by router/training/gate.py:check_first_fit_eligibility before any
    expensive embedding work, and by KMeansTrainer.train() when the caller
    bypassed the gate.
    """


class KMeansFitError(RouterError):
    """KMeans failed to converge or produced degenerate clusters."""
