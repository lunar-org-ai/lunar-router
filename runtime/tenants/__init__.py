"""Tenant registry — the layer above agents (P16.1).

Mirrors ``runtime/agents/`` one level up: where the agents package
manages ``agents/<id>/...`` for a single tenant, this package manages
``tenants/<tid>/...`` and treats each tenant as its own private agent
catalog + ledger + traces + corpora root.

Public surface:
  - :mod:`runtime.tenants.registry` — CRUD on ``tenants/`` + the
    ``tenants/_registry.json`` catalog.
  - :mod:`runtime.tenants.tokens` — mint / resolve / revoke per-tenant
    Bearer tokens of the form ``otrcy_live_<base32>``.
  - :mod:`runtime.tenants.bootstrap` — first-boot migration that moves
    the legacy single-tenant layout under ``tenants/_default/``.
"""

from runtime.tenants.registry import (  # noqa: F401 (re-export)
    create_tenant,
    delete_tenant,
    ensure_bootstrapped,
    get_tenant,
    get_tenant_dir,
    list_tenants,
)
from runtime.tenants.tokens import (  # noqa: F401
    list_tokens,
    mint_token,
    rebuild_index,
    resolve_token,
    revoke_token,
)
from runtime.tenants.types import (  # noqa: F401
    TenantMetadata,
    TenantRegistry,
    TokenRecord,
)
