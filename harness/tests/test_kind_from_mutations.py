"""Confirm kind_from_mutations distinguishes UniRoute router_config from
the legacy pipeline/route.yaml router kind."""

from __future__ import annotations

from harness.types import kind_from_mutations


def test_router_config_takes_precedence_over_legacy_router():
    """versions/router_config_v3.json must NOT match the legacy 'route' bucket."""
    assert kind_from_mutations(["versions/router_config_v3.json:x=1"]) == "router_config"


def test_legacy_route_yaml_still_matches_router():
    """pipeline/route.yaml continues to map to the legacy 'router' kind."""
    assert kind_from_mutations(["pipeline/route.yaml:knobs.k=1"]) == "router"


def test_other_kinds_unchanged():
    assert kind_from_mutations(["pipeline/retrieve.yaml:knobs.k=1"]) == "rag"
    assert kind_from_mutations(["pipeline/rerank.yaml:knobs.k=1"]) == "rerank"
    assert kind_from_mutations(["agent/prompts/system.md:body=x"]) == "prompt"
    assert kind_from_mutations(["pipeline/generate.yaml:knobs.t=1"]) == "prompt"
    assert kind_from_mutations(["pipeline/memory.yaml:knobs.window=10"]) == "memory"
    assert kind_from_mutations(["random/file.yaml:x=1"]) == "other"


def test_mixed_mutations_router_config_wins():
    """A mutation set that touches both router_config and other files
    surfaces as router_config (highest priority)."""
    muts = [
        "versions/router_config_v5.json:<inline>=...",
        "agent/prompts/system.md:body=updated",
    ]
    assert kind_from_mutations(muts) == "router_config"
