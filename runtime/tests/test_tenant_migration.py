"""Tests for the legacy → multi-tenant migration (P16.1)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from runtime.tenants.bootstrap import migrate_legacy_to_default


def _seed_legacy_project(root: Path) -> None:
    """Plant the pre-P16.1 layout under ``root``."""
    (root / "agents" / "_default" / "prompts").mkdir(parents=True)
    (root / "agents" / "_default" / "prompts" / "system.md").write_text(
        "You are helpful.", encoding="utf-8"
    )
    (root / "agents" / "registry.json").write_text(
        '{"agents": [{"id": "_default", "name": "Default agent"}], "active": "_default"}',
        encoding="utf-8",
    )
    (root / "ledger" / "_default" / "entries").mkdir(parents=True)
    (root / "ledger" / "_default" / "entries" / "demo.jsonl").write_text(
        '{"event": "demo"}\n', encoding="utf-8"
    )
    (root / "ledger" / "versions").mkdir(parents=True)
    (root / "traces" / "_default" / "raw").mkdir(parents=True)
    (root / "traces" / "_default" / "raw" / "2026-05-13.jsonl").write_text(
        '{"trace": "demo"}\n', encoding="utf-8"
    )
    (root / "corpora" / "indexed").mkdir(parents=True)
    (root / "corpora" / "indexed" / "manifest.jsonl").write_text(
        "", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Migration happy path
# ---------------------------------------------------------------------------


def test_migration_moves_every_legacy_dir(tmp_path):
    _seed_legacy_project(tmp_path)
    moved = migrate_legacy_to_default(tmp_path, create_symlinks=False)

    assert moved is True
    # Originals gone
    for legacy in ("agents", "ledger", "traces", "corpora"):
        assert not (tmp_path / legacy).is_dir() or (tmp_path / legacy).is_symlink()
    # Moved under tenants/_default/
    base = tmp_path / "tenants" / "_default"
    assert (base / "agents" / "registry.json").is_file()
    assert (base / "agents" / "_default" / "prompts" / "system.md").is_file()
    assert (base / "ledger" / "_default" / "entries" / "demo.jsonl").is_file()
    assert (base / "ledger" / "versions").is_dir()
    assert (base / "traces" / "_default" / "raw" / "2026-05-13.jsonl").is_file()
    assert (base / "corpora" / "indexed" / "manifest.jsonl").is_file()


def test_migration_writes_log(tmp_path):
    _seed_legacy_project(tmp_path)
    migrate_legacy_to_default(tmp_path, create_symlinks=False)
    log = json.loads(
        (tmp_path / "tenants" / "_default" / "migration.log.json").read_text(
            encoding="utf-8"
        )
    )
    assert log["phase"] == "P16.1"
    moves = {m["from"].rsplit("/", 1)[-1] for m in log["moves"]}
    assert moves == {"agents", "ledger", "traces", "corpora"}


def test_migration_is_idempotent(tmp_path):
    """Second invocation is a no-op."""
    _seed_legacy_project(tmp_path)
    first = migrate_legacy_to_default(tmp_path, create_symlinks=False)
    second = migrate_legacy_to_default(tmp_path, create_symlinks=False)
    assert first is True
    assert second is False


def test_migration_on_fresh_install_creates_default_but_no_log(tmp_path):
    """No legacy dirs → tenants/_default/ exists but no migration log."""
    moved = migrate_legacy_to_default(tmp_path, create_symlinks=False)
    assert moved is False
    assert (tmp_path / "tenants" / "_default").is_dir()
    assert not (tmp_path / "tenants" / "_default" / "migration.log.json").is_file()


# ---------------------------------------------------------------------------
# Symlinks
# ---------------------------------------------------------------------------


def test_migration_creates_symlinks_when_requested(tmp_path):
    _seed_legacy_project(tmp_path)
    migrate_legacy_to_default(tmp_path, create_symlinks=True)

    for name in ("agents", "ledger", "traces", "corpora"):
        link = tmp_path / name
        assert link.is_symlink(), f"{name} should be a symlink"
        target = os.readlink(link)
        # Relative target, points to tenants/_default/<name>
        assert target == os.path.join("tenants", "_default", name)


def test_migration_omits_symlinks_when_disabled(tmp_path):
    _seed_legacy_project(tmp_path)
    migrate_legacy_to_default(tmp_path, create_symlinks=False)
    # Without the convenience symlinks the legacy roots simply don't exist
    for name in ("agents", "ledger", "traces", "corpora"):
        assert not (tmp_path / name).is_symlink()
        assert not (tmp_path / name).exists()


def test_existing_symlink_is_left_alone(tmp_path):
    """If the operator already created a symlink at the legacy root,
    don't try to outsmart them."""
    _seed_legacy_project(tmp_path)
    # Stand in a fake symlink at /agents (overwriting the seeded dir)
    (tmp_path / "agents_fake_target").mkdir()
    (tmp_path / "agents").rename(tmp_path / "tenants_pre_agents")  # park
    os.symlink("agents_fake_target", tmp_path / "agents", target_is_directory=True)

    migrate_legacy_to_default(tmp_path, create_symlinks=True)
    # Symlink not overwritten
    assert (tmp_path / "agents").is_symlink()
    assert os.readlink(tmp_path / "agents") == "agents_fake_target"


# ---------------------------------------------------------------------------
# Partial / odd states
# ---------------------------------------------------------------------------


def test_migration_skips_dir_if_target_already_populated(tmp_path):
    """If tenants/_default/agents/ already exists, the legacy agents/
    stays in place (we won't blow over existing tenant data)."""
    _seed_legacy_project(tmp_path)
    pre = tmp_path / "tenants" / "_default" / "agents"
    pre.mkdir(parents=True)
    (pre / "marker.txt").write_text("preexisting", encoding="utf-8")

    migrate_legacy_to_default(tmp_path, create_symlinks=False)

    # Legacy agents/ still present, target unchanged
    assert (tmp_path / "agents").is_dir()
    assert (pre / "marker.txt").read_text(encoding="utf-8") == "preexisting"
    # But ledger/traces/corpora moved fine
    assert (tmp_path / "tenants" / "_default" / "ledger" / "_default").is_dir()


def test_migration_then_create_new_tenant_does_not_disturb_default(tmp_path):
    """Smoke: after migration, adding a fresh tenant leaves _default
    untouched."""
    from runtime.tenants.registry import create_tenant, ensure_bootstrapped

    _seed_legacy_project(tmp_path)
    migrate_legacy_to_default(tmp_path, create_symlinks=False)
    ensure_bootstrapped(root=tmp_path / "tenants")
    create_tenant("Acme", slug="acme", root=tmp_path / "tenants")

    assert (
        tmp_path / "tenants" / "_default" / "ledger" / "_default" / "entries" / "demo.jsonl"
    ).is_file()
    assert (tmp_path / "tenants" / "acme" / "agents").is_dir()
