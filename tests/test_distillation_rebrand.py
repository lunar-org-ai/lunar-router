"""Regression tests for the distillation pipeline after the rebrand.

The distillation module — `Distiller`, `TrainingClient`, the FastAPI route
registrations, and the module-level env-var reads — crosses almost every
surface that the `lunar_router -> opentracy` rename touched (imports,
`OPENTRACY_*` / `LUNAR_*` fallback, package-data paths, FastAPI routing).
Before this file there was zero automated coverage for any of it, so a
silent regression (e.g. a stray SyntaxError like the one that broke
`/v1/datasets` at commit 9512b5e, or a forgotten `include_samples=False`
default) would escape CI unnoticed.

The tests here are deliberately pure-static: they import modules, inspect
classes and FastAPI route signatures, and poke env-var resolution — no
live HTTP, no DB, no GPU. That keeps them fast enough for every PR while
still catching the class of bugs the rebrand actually produced.
"""

from __future__ import annotations

import importlib
import inspect
import os
import warnings

import pytest


# ---------------------------------------------------------------------------
# 1. Public SDK surface — catches any SyntaxError or missing export.
# ---------------------------------------------------------------------------


def test_distillation_subpackage_imports_cleanly() -> None:
    """Every distillation submodule parses and loads without side effects.

    This is what would have caught the broken `from .schemas import (...)`
    injection in `distillation/router.py` at import time.
    """
    for name in [
        "opentracy.distillation",
        "opentracy.distillation.client",
        "opentracy.distillation.schemas",
        "opentracy.distillation.serialization",
        "opentracy.distillation.repository",
        "opentracy.distillation.pipeline",
        "opentracy.distillation.export",
        "opentracy.distillation.curation",
        "opentracy.distillation.trainer",
    ]:
        importlib.import_module(name)


def test_top_level_exports_distiller_and_training_client() -> None:
    """`from opentracy import Distiller, TrainingClient, DistillerError` stays
    part of the public API. Notebooks and user code depend on the top-level
    re-export; if it disappears the shim won't catch it (it redirects by name,
    not by re-import)."""
    import opentracy

    for attr in ("Distiller", "TrainingClient", "DistillerError"):
        assert hasattr(opentracy, attr), f"opentracy.{attr} must remain exported"


# ---------------------------------------------------------------------------
# 2. Distiller — env-var resolution and constructor contract.
# ---------------------------------------------------------------------------


def test_distiller_constructor_accepts_legacy_kwargs() -> None:
    """Signature of Distiller.__init__ is part of the public API — users pass
    base_url, api_key, tenant_id, timeout positionally in notebooks."""
    from opentracy.distillation.client import Distiller

    sig = inspect.signature(Distiller.__init__)
    for param in ("base_url", "tenant_id", "api_key", "timeout"):
        assert param in sig.parameters, f"Distiller() lost parameter {param!r}"


def test_distiller_api_key_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distiller reads API key via env(): OPENTRACY_API_KEY wins over LUNAR_API_KEY,
    and LUNAR_API_KEY alone still works (with a DeprecationWarning)."""
    from opentracy.distillation.client import Distiller

    monkeypatch.delenv("OPENTRACY_API_KEY", raising=False)
    monkeypatch.delenv("LUNAR_API_KEY", raising=False)
    monkeypatch.setenv("LUNAR_API_KEY", "legacy-token")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        d = Distiller(base_url="http://unused:0")
        dep = [w for w in captured if issubclass(w.category, DeprecationWarning)
               and "LUNAR_API_KEY" in str(w.message)]
        assert dep, "LUNAR_API_KEY fallback must emit a DeprecationWarning"

    assert d._http.headers.get("Authorization") == "Bearer legacy-token", (
        "Distiller must inject the legacy LUNAR_API_KEY as a Bearer token"
    )
    d.close()

    monkeypatch.setenv("OPENTRACY_API_KEY", "new-token")
    d2 = Distiller(base_url="http://unused:0")
    assert d2._http.headers.get("Authorization") == "Bearer new-token", (
        "OPENTRACY_API_KEY must take precedence over LUNAR_API_KEY"
    )
    d2.close()


def test_distiller_explicit_api_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit api_key= beats any env var — this was true pre-rebrand too."""
    from opentracy.distillation.client import Distiller

    monkeypatch.setenv("OPENTRACY_API_KEY", "env-token")
    d = Distiller(base_url="http://unused:0", api_key="explicit-token")
    assert d._http.headers.get("Authorization") == "Bearer explicit-token"
    d.close()


# ---------------------------------------------------------------------------
# 3. FastAPI live surface — guard against the datasets-style shadow-route bug.
# ---------------------------------------------------------------------------


_EXPECTED_ROUTES = [
    ("GET", "/v1/distillation"),
    ("POST", "/v1/distillation"),
    ("POST", "/v1/distillation/estimate"),
    ("GET", "/v1/distillation/teacher-models"),
    ("GET", "/v1/distillation/student-models"),
    ("GET", "/v1/distillation/{job_id}"),
    ("DELETE", "/v1/distillation/{job_id}"),
    ("POST", "/v1/distillation/{job_id}/cancel"),
    ("GET", "/v1/distillation/{job_id}/logs"),
    ("GET", "/v1/distillation/{job_id}/candidates"),
    ("GET", "/v1/distillation/{job_id}/artifacts"),
    ("POST", "/v1/distillation/{job_id}/deploy"),
    ("GET", "/v1/distillation/{job_id}/metrics"),
]


def _handlers_for(path: str, method: str) -> list:
    from opentracy.api.server import app

    return [
        r
        for r in app.routes
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set())
    ]


@pytest.mark.parametrize("method,path", _EXPECTED_ROUTES)
def test_distillation_route_registered_exactly_once(method: str, path: str) -> None:
    """Every public distillation endpoint is registered once and only once.

    If someone copies the endpoint definition into `distillation/router.py`
    AND mounts that router too, FastAPI will keep both — and whichever was
    registered first wins, exactly the pattern that gave `/v1/datasets/{id}`
    the wrong `include_samples=False` default. Catch the shadow before it
    ships.
    """
    handlers = _handlers_for(path, method)
    assert len(handlers) == 1, (
        f"{method} {path}: found {len(handlers)} handlers, want 1. "
        "Duplicate registration has already caused one production bug — "
        "see test_api_dataset_defaults for context."
    )


# ---------------------------------------------------------------------------
# 4. Module-level constants must resolve through the env-var helper.
# ---------------------------------------------------------------------------


def test_distiller_default_base_url_uses_env_helper() -> None:
    """DEFAULT_BASE_URL goes through opentracy._env.env (not os.environ.get
    directly). Verifying this catches anyone who replaces the helper call
    with a raw os.environ.get and inadvertently drops the LUNAR_ fallback."""
    import opentracy.distillation.client as client

    src = inspect.getsource(client)
    assert 'env("API_URL"' in src, (
        "distillation/client.py must read API_URL via opentracy._env.env() "
        "so LUNAR_API_URL stays as a deprecation-warning fallback"
    )
    assert "os.environ.get(\"LUNAR_" not in src and "os.getenv(\"LUNAR_" not in src, (
        "distillation/client.py must not read LUNAR_* directly — use env() "
        "so the deprecation warning fires and OPENTRACY_* wins"
    )
