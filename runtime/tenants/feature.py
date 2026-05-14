"""Multi-tenant feature gate (P16.1).

Opentracy ships in TWO modes:

- **OSS local** — single-tenant, no infra. Everything stays at the
  project root (``agents/``, ``ledger/``, ``traces/``, ``corpora/``).
  No migration runs on boot, no admin auth split, no per-request
  tenant context. The default mode — a user who clones the repo and
  runs locally never has to touch anything.

- **Hosted/infra** — multi-tenant via Cloud Run + GCS fuse + KMS.
  Drives the bootstrap migration, the admin API split, the tenant
  token middleware. Lives mostly in the sibling private ``opentracy-
  infra/`` repo.

Switching modes is a single env var:

    OPENTRACY_MULTI_TENANT=1   → hosted/infra behavior on
                                otherwise               → OSS local

OSS users never have to set this. Hosted deploys set it in the Cloud
Run env. The flag is read fresh on every call so tests can flip it
via ``monkeypatch.setenv`` without restarting the process.
"""

from __future__ import annotations

import os


_ENV_FLAG = "OPENTRACY_MULTI_TENANT"


def is_multi_tenant_enabled() -> bool:
    """True iff the hosted multi-tenant code paths should activate."""
    return os.environ.get(_ENV_FLAG, "").strip() == "1"
