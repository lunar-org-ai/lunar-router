"""Version snapshots — the rollback primitive.

When a candidate is promoted to live, the prior agent/ tree is snapshotted
into ledger/versions/<version_id>/agent/. Rollback copies that snapshot
back over agent/.

Snapshots are content-addressed by the version string in agent.yaml. We
trust that version is bumped on every promotion (executor's responsibility).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

LIVE_AGENT = Path("agent")
VERSIONS_DIR = Path("ledger/versions")


def read_version(agent_dir: Path | str = LIVE_AGENT) -> str:
    """Read the current `version` from agent.yaml."""
    cfg_path = Path(agent_dir) / "agent.yaml"
    with cfg_path.open() as f:
        doc = yaml.safe_load(f) or {}
    return doc.get("agent", {}).get("version", "unknown")


def _read_version(agent_dir: Path) -> str:  # backward-compat alias
    return read_version(agent_dir)


def snapshot_path(version: str, versions_dir: Path | str = VERSIONS_DIR) -> Path:
    return Path(versions_dir) / version / "agent"


def snapshot_agent(
    agent_dir: Path | str = LIVE_AGENT,
    versions_dir: Path | str = VERSIONS_DIR,
) -> tuple[str, Path]:
    """Copy live agent/ into ledger/versions/<version>/agent. Returns (version, path)."""
    agent_dir = Path(agent_dir)
    version = _read_version(agent_dir)
    target = snapshot_path(version, versions_dir)
    if target.exists():
        # Already snapshotted — idempotent. Don't overwrite (would lose the
        # original snapshot if version got duplicated by mistake).
        return version, target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(agent_dir, target)
    return version, target


def list_snapshots(versions_dir: Path | str = VERSIONS_DIR) -> list[str]:
    root = Path(versions_dir)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "agent" / "agent.yaml").exists())


def restore_version(
    version: str,
    agent_dir: Path | str = LIVE_AGENT,
    versions_dir: Path | str = VERSIONS_DIR,
) -> Path:
    """Replace live agent/ with the snapshot for `version`."""
    agent_dir = Path(agent_dir)
    snap = snapshot_path(version, versions_dir)
    if not snap.exists():
        raise FileNotFoundError(f"no snapshot for version {version!r}: {snap}")

    # Snapshot the *current* live agent first (so we can rollback the rollback).
    current_version, _ = snapshot_agent(agent_dir, versions_dir)
    if current_version == version:
        return agent_dir  # nothing to do

    # Replace
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    shutil.copytree(snap, agent_dir)
    return agent_dir
