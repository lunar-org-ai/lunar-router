"""Regression test for the ``GET /v1/datasets/{dataset_id}`` default params.

The dataset detail page in the UI relies on the endpoint hydrating samples
when called without a query string. A prior state of this code defaulted
``include_samples`` to ``False`` and ``samples_limit`` to 50, so the UI
silently showed "No samples yet" even for populated datasets. These tests
inspect the FastAPI route's signature directly — no DB required — so
regressions are caught in every CI run.

If you intentionally change the defaults, update these assertions and call
out the semantic change in CHANGELOG.md so clients can adapt.
"""

from __future__ import annotations

import inspect


def _route(path: str, method: str = "GET"):
    from opentracy.api.server import app

    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route {method} {path} not registered on app")


def test_get_dataset_defaults_include_samples_true() -> None:
    """Callers that omit ``include_samples`` must get samples."""
    sig = inspect.signature(_route("/v1/datasets/{dataset_id}").endpoint)
    param = sig.parameters["include_samples"]
    assert param.default is True, (
        "GET /v1/datasets/{dataset_id} must default include_samples=True so "
        "dataset detail pages render without explicit opt-in. See CHANGELOG."
    )


def test_get_dataset_samples_limit_is_reasonable() -> None:
    """Default sample page size must cover the smallest useful datasets."""
    sig = inspect.signature(_route("/v1/datasets/{dataset_id}").endpoint)
    param = sig.parameters["samples_limit"]
    assert param.default >= 100, (
        f"samples_limit default is {param.default}; pages for datasets with >50 "
        "samples would truncate silently. Expected >= 100."
    )


def test_get_dataset_route_is_not_shadowed() -> None:
    """Exactly one GET handler must be registered for the detail path."""
    from opentracy.api.server import app

    handlers = [
        r
        for r in app.routes
        if getattr(r, "path", None) == "/v1/datasets/{dataset_id}"
        and "GET" in getattr(r, "methods", set())
    ]
    assert len(handlers) == 1, (
        f"expected exactly one GET handler for /v1/datasets/{{dataset_id}}, "
        f"found {len(handlers)}. Duplicate routes with different defaults "
        "caused a production bug — do not re-introduce them."
    )
