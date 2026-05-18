"""Rollback — restore live agent/ from a prior version snapshot."""

from __future__ import annotations

from pathlib import Path

from ledger.versioning import LIVE_AGENT, list_snapshots, read_version, restore_version
from ledger.writer import write_entry


def rollback_to(
    version: str,
    *,
    agent_dir: Path | str = LIVE_AGENT,
    reason: str = "manual rollback",
) -> str:
    """Restore live agent/ to a prior `version`. Records the rollback in the ledger.

    Returns the version now live (= `version`, except no-op if already there).
    """
    agent_dir = Path(agent_dir)
    old_version = read_version(agent_dir)
    if old_version == version:
        return version

    if version not in list_snapshots():
        raise FileNotFoundError(
            f"no snapshot for version {version!r}; available: {list_snapshots()}"
        )

    restore_version(version, agent_dir)

    write_entry(
        kind="rollback",
        agent_version_before=old_version,
        agent_version_after=version,
        summary=f"rolled back: {old_version} → {version} ({reason})",
        payload={"reason": reason},
    )
    return version
