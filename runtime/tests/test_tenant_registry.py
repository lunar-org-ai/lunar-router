"""Tests for the tenant registry (P16.1)."""

from __future__ import annotations

import json

import pytest

from runtime.tenants.registry import (
    create_tenant,
    delete_tenant,
    ensure_bootstrapped,
    get_tenant,
    get_tenant_dir,
    list_tenants,
)


@pytest.fixture
def tenants_root(tmp_path):
    """Tmp tenants/ root with no pre-existing registry."""
    return tmp_path / "tenants"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_creates_default_when_missing(tenants_root):
    reg = ensure_bootstrapped(root=tenants_root)
    assert len(reg.tenants) == 1
    assert reg.tenants[0].id == "_default"
    assert (tenants_root / "_default").is_dir()
    assert (tenants_root / "_registry.json").is_file()


def test_bootstrap_is_idempotent(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    # Add a custom tenant between runs to verify the second call is a no-op
    create_tenant("Acme", root=tenants_root)
    reg2 = ensure_bootstrapped(root=tenants_root)
    assert len(reg2.tenants) == 2
    ids = {t.id for t in reg2.tenants}
    assert "_default" in ids
    assert "acme" in ids


def test_bootstrap_preserves_existing_registry_contents(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    create_tenant("Beta", slug="beta", root=tenants_root)
    # Re-run bootstrap; beta should survive
    reg = ensure_bootstrapped(root=tenants_root)
    assert reg.get("beta") is not None


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_with_name_slugifies(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    meta = create_tenant("Acme Corp", root=tenants_root)
    assert meta.id == "acme-corp"
    assert (tenants_root / "acme-corp").is_dir()
    # Sub-dirs pre-created so writers don't need to mkdir
    for sub in ("agents", "ledger", "traces", "corpora"):
        assert (tenants_root / "acme-corp" / sub).is_dir()


def test_create_with_explicit_slug(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    meta = create_tenant("Beta Co", slug="beta-co", root=tenants_root)
    assert meta.id == "beta-co"


def test_create_collision_appends_suffix(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    first = create_tenant("Acme", root=tenants_root)
    second = create_tenant("Acme", root=tenants_root)
    assert first.id == "acme"
    assert second.id != "acme"
    assert second.id.startswith("acme-")


def test_create_rejects_reserved_slugs(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    for reserved in ("_default", "_deleted", "_registry", "_tokens_index"):
        with pytest.raises(ValueError, match="invalid slug|reserved"):
            create_tenant("X", slug=reserved, root=tenants_root)


def test_create_rejects_invalid_slug(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    for bad in ("UPPER", "with space", "x", "with_underscore", "-leading-hyphen"):
        with pytest.raises(ValueError, match="invalid slug"):
            create_tenant("X", slug=bad, root=tenants_root)


def test_create_rejects_duplicate_explicit_slug(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    create_tenant("Acme", slug="acme", root=tenants_root)
    with pytest.raises(ValueError, match="already exists"):
        create_tenant("Different", slug="acme", root=tenants_root)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_soft_moves_to_deleted_bucket(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    create_tenant("Acme", root=tenants_root)
    assert (tenants_root / "acme").is_dir()
    delete_tenant("acme", root=tenants_root)
    assert not (tenants_root / "acme").exists()
    # Soft-deleted with random suffix under _deleted/
    deleted = list((tenants_root / "_deleted").iterdir())
    assert len(deleted) == 1
    assert deleted[0].name.startswith("acme-")


def test_delete_default_is_refused(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    with pytest.raises(ValueError, match="_default"):
        delete_tenant("_default", root=tenants_root)


def test_delete_unknown_raises_keyerror(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    with pytest.raises(KeyError):
        delete_tenant("does-not-exist", root=tenants_root)


def test_delete_prunes_tokens_index(tenants_root):
    """Deleting a tenant should remove its tokens from the global
    lookup so resolves against revoked tenants 401 immediately."""
    from runtime.tenants.tokens import mint_token, resolve_token

    ensure_bootstrapped(root=tenants_root)
    create_tenant("Acme", slug="acme", root=tenants_root)
    plaintext, _ = mint_token("acme", "prod", root=tenants_root)
    assert resolve_token(plaintext, root=tenants_root) == "acme"

    delete_tenant("acme", root=tenants_root)
    # Token should no longer resolve to a tenant
    assert resolve_token(plaintext, root=tenants_root) is None


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def test_get_tenant_dir_does_not_require_existence(tenants_root):
    """Helper is path math — works for not-yet-created tenants."""
    p = get_tenant_dir("future", root=tenants_root)
    assert p == tenants_root / "future"
    assert not p.exists()


def test_list_then_get_round_trip(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    create_tenant("Acme", slug="acme", root=tenants_root)
    reg = list_tenants(root=tenants_root)
    ids = {t.id for t in reg.tenants}
    assert ids == {"_default", "acme"}
    assert get_tenant("acme", root=tenants_root).name == "Acme"
    assert get_tenant("nope", root=tenants_root) is None


def test_registry_on_disk_is_pretty_printed(tenants_root):
    ensure_bootstrapped(root=tenants_root)
    body = (tenants_root / "_registry.json").read_text(encoding="utf-8")
    # Multi-line indented JSON for git-friendly diffs
    assert "\n" in body
    parsed = json.loads(body)
    assert "tenants" in parsed
