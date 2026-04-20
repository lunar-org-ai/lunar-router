"""Distillation Module — BOND (Best-of-N Distillation) Pipeline.

Local-first model distillation: data generation → curation → training → export.
All state persisted to ClickHouse; artifacts stored on local filesystem.

The SDK client (``Distiller``) is cheap to import; the live FastAPI
endpoints for /v1/distillation/* are registered inline in
``opentracy/api/server.py``. The response serializer shared by those
handlers lives at ``opentracy.distillation.serialization.serialize_job``.
"""
from .client import Distiller, TrainingClient, DistillerError

__all__ = ["Distiller", "TrainingClient", "DistillerError"]
