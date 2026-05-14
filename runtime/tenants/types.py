"""Tenant metadata + token record types (P16.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TenantMetadata:
    """One entry in ``tenants/_registry.json``.

    The lightweight identity card the registry tracks. Heavyweight
    state (agents, ledger, traces, corpora) lives in
    ``tenants/<id>/`` on disk.
    """

    id: str                              # slug, also the directory name
    name: str = ""
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TenantMetadata":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )


@dataclass
class TenantRegistry:
    """The on-disk shape of ``tenants/_registry.json``."""

    tenants: list[TenantMetadata] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"tenants": [t.to_dict() for t in self.tenants]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TenantRegistry":
        return cls(
            tenants=[TenantMetadata.from_dict(t) for t in (data.get("tenants") or [])],
        )

    def get(self, tenant_id: str) -> Optional[TenantMetadata]:
        for t in self.tenants:
            if t.id == tenant_id:
                return t
        return None


@dataclass
class TokenRecord:
    """One entry in ``tenants/<tid>/tokens.json``.

    Plaintext tokens are never stored — only ``hash`` is. The plaintext
    is returned once at mint time and the operator is expected to
    capture it.
    """

    hash: str                  # sha256(token).hexdigest() — 64 hex chars
    label: str = ""
    created_at: str = ""
    last_used_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "label": self.label,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenRecord":
        return cls(
            hash=str(data["hash"]),
            label=str(data.get("label", "")),
            created_at=str(data.get("created_at", "")),
            last_used_at=data.get("last_used_at"),
        )

    @property
    def hash_prefix(self) -> str:
        """First 12 hex chars — enough entropy for revocation identifiers
        without exposing the full lookup key."""
        return self.hash[:12]
