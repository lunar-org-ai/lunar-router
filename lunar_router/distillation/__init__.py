"""Distillation Module — BOND (Best-of-N Distillation) Pipeline.

Local-first model distillation: data generation → curation → training → export.
All state persisted to ClickHouse; artifacts stored on local filesystem.

The SDK client (``Distiller``) is cheap to import. The FastAPI router
(``lunar_router.distillation.router``) is NOT re-exported here — it pulls
FastAPI / uvicorn / rich / click and only the API server needs it.
Import it explicitly: ``from lunar_router.distillation.router import router``.
"""
from .client import Distiller, TrainingClient, DistillerError

__all__ = ["Distiller", "TrainingClient", "DistillerError"]
