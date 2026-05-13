"""Agent metadata types (P2.0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentMetadata:
    """One entry in ``agents/registry.json``.

    The lightweight identity card the registry tracks. Heavyweight
    state (prompt body, pipeline yamls, version snapshots) lives in
    ``agents/<id>/`` on disk.
    """

    id: str                              # slug, also the directory name
    name: str = ""                       # operator-chosen display name
    template: Optional[str] = None       # 'support' | 'sales' | ... | None
    model: str = "claude-sonnet-4-6"
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    onboarding_completed_at: Optional[str] = None
    # Onboarding config snapshot — handy for explaining "what did the
    # operator say when this agent was created?" in the Evolution timeline.
    onboarding: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "model": self.model,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "onboarding_completed_at": self.onboarding_completed_at,
            "onboarding": self.onboarding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMetadata":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            template=data.get("template"),
            model=str(data.get("model", "claude-sonnet-4-6")),
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            onboarding_completed_at=data.get("onboarding_completed_at"),
            onboarding=data.get("onboarding"),
        )


@dataclass
class Registry:
    """The on-disk shape of ``agents/registry.json``."""

    agents: list[AgentMetadata] = field(default_factory=list)
    active: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.to_dict() for a in self.agents],
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Registry":
        return cls(
            agents=[AgentMetadata.from_dict(a) for a in (data.get("agents") or [])],
            active=data.get("active"),
        )

    def get(self, agent_id: str) -> Optional[AgentMetadata]:
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None
