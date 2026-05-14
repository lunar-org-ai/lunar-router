"""Tests for per-tenant Bearer tokens (P16.1)."""

from __future__ import annotations

import json

import pytest

from runtime.tenants.registry import create_tenant, ensure_bootstrapped
from runtime.tenants.tokens import (
    list_tokens,
    mint_token,
    rebuild_index,
    resolve_token,
    revoke_token,
)


@pytest.fixture
def tenants_root(tmp_path):
    root = tmp_path / "tenants"
    ensure_bootstrapped(root=root)
    create_tenant("Acme", slug="acme", root=root)
    create_tenant("Beta", slug="beta", root=root)
    return root


# ---------------------------------------------------------------------------
# Mint
# ---------------------------------------------------------------------------


def test_mint_returns_plaintext_with_expected_shape(tenants_root):
    plaintext, record = mint_token("acme", "prod", root=tenants_root)
    assert plaintext.startswith("otrcy_live_")
    assert len(plaintext) > len("otrcy_live_")
    assert record.label == "prod"
    assert len(record.hash) == 64        # sha256 hex
    assert record.created_at
    assert record.last_used_at is None


def test_mint_persists_only_the_hash_never_plaintext(tenants_root):
    plaintext, record = mint_token("acme", "prod", root=tenants_root)
    on_disk = (tenants_root / "acme" / "tokens.json").read_text(encoding="utf-8")
    parsed = json.loads(on_disk)
    assert plaintext not in on_disk
    assert parsed["tokens"][0]["hash"] == record.hash


def test_mint_updates_global_index(tenants_root):
    plaintext, record = mint_token("acme", "prod", root=tenants_root)
    index = json.loads(
        (tenants_root / "_tokens_index.json").read_text(encoding="utf-8")
    )
    assert index[record.hash] == "acme"


def test_mint_rejects_unknown_tenant(tenants_root):
    with pytest.raises(FileNotFoundError):
        mint_token("does-not-exist", "x", root=tenants_root)


def test_mint_generates_distinct_tokens(tenants_root):
    p1, _ = mint_token("acme", "one", root=tenants_root)
    p2, _ = mint_token("acme", "two", root=tenants_root)
    assert p1 != p2


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


def test_resolve_returns_tenant_id_for_valid_token(tenants_root):
    plaintext, _ = mint_token("acme", "prod", root=tenants_root)
    assert resolve_token(plaintext, root=tenants_root) == "acme"


def test_resolve_returns_none_for_unknown_token(tenants_root):
    assert resolve_token("otrcy_live_unknown", root=tenants_root) is None


def test_resolve_returns_none_for_garbage(tenants_root):
    assert resolve_token("", root=tenants_root) is None
    assert resolve_token("not-a-token", root=tenants_root) is None
    assert resolve_token("Bearer abc", root=tenants_root) is None


def test_resolve_isolates_tenants(tenants_root):
    """Tokens issued to acme must never resolve to beta."""
    acme_token, _ = mint_token("acme", "prod", root=tenants_root)
    beta_token, _ = mint_token("beta", "prod", root=tenants_root)
    assert resolve_token(acme_token, root=tenants_root) == "acme"
    assert resolve_token(beta_token, root=tenants_root) == "beta"


def test_resolve_touches_last_used(tenants_root):
    plaintext, record = mint_token("acme", "prod", root=tenants_root)
    assert record.last_used_at is None
    resolve_token(plaintext, root=tenants_root)
    refreshed = list_tokens("acme", root=tenants_root)
    assert refreshed[0].last_used_at is not None


def test_resolve_can_skip_touch(tenants_root):
    plaintext, _ = mint_token("acme", "prod", root=tenants_root)
    resolve_token(plaintext, root=tenants_root, touch_last_used=False)
    refreshed = list_tokens("acme", root=tenants_root)
    assert refreshed[0].last_used_at is None


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------


def test_revoke_removes_from_per_tenant_file(tenants_root):
    plaintext, record = mint_token("acme", "prod", root=tenants_root)
    assert revoke_token("acme", record.hash_prefix, root=tenants_root) is True
    assert list_tokens("acme", root=tenants_root) == []


def test_revoke_removes_from_global_index(tenants_root):
    plaintext, record = mint_token("acme", "prod", root=tenants_root)
    revoke_token("acme", record.hash_prefix, root=tenants_root)
    assert resolve_token(plaintext, root=tenants_root) is None


def test_revoke_unknown_prefix_returns_false(tenants_root):
    mint_token("acme", "prod", root=tenants_root)
    assert revoke_token("acme", "zzzzzzzzzzzz", root=tenants_root) is False


def test_revoke_does_not_touch_other_tenants_tokens(tenants_root):
    p_acme, r_acme = mint_token("acme", "prod", root=tenants_root)
    p_beta, _ = mint_token("beta", "prod", root=tenants_root)
    revoke_token("acme", r_acme.hash_prefix, root=tenants_root)
    assert resolve_token(p_acme, root=tenants_root) is None
    assert resolve_token(p_beta, root=tenants_root) == "beta"


# ---------------------------------------------------------------------------
# Rebuild index
# ---------------------------------------------------------------------------


def test_rebuild_index_recovers_from_corruption(tenants_root):
    mint_token("acme", "prod", root=tenants_root)
    mint_token("beta", "prod", root=tenants_root)
    # Trash the index
    (tenants_root / "_tokens_index.json").write_text("{}", encoding="utf-8")
    n = rebuild_index(root=tenants_root)
    assert n == 2
    index = json.loads(
        (tenants_root / "_tokens_index.json").read_text(encoding="utf-8")
    )
    assert set(index.values()) == {"acme", "beta"}


def test_rebuild_index_ignores_underscore_dirs(tenants_root):
    """Reserved underscore-prefixed dirs (e.g. _deleted) shouldn't be
    scanned for tokens."""
    (tenants_root / "_deleted").mkdir()
    (tenants_root / "_deleted" / "tokens.json").write_text(
        '{"tokens": [{"hash": "deadbeef", "label": "ghost"}]}',
        encoding="utf-8",
    )
    mint_token("acme", "prod", root=tenants_root)
    n = rebuild_index(root=tenants_root)
    assert n == 1
