"""Tests for the secrets→KMS migration script (P16.3.S5)."""

from __future__ import annotations

import pytest


@pytest.fixture
def patched_kms(monkeypatch):
    """Force select_crypto() to return FakeKmsCrypto for the migration
    helper. No network."""
    from runtime.crypto import FakeKmsCrypto

    fake = FakeKmsCrypto(kek_name="projects/test/cryptoKeys/test")
    import runtime.crypto as _rc
    import tools.migrate_secrets_to_kms as _mig

    monkeypatch.setattr(_rc, "select_crypto", lambda: fake)
    monkeypatch.setattr(_mig, "select_crypto", lambda: fake, raising=False)
    yield fake


def _seed_agent_with_plain_secrets(root, tid, aid, body="ANTHROPIC_API_KEY=sk-ant-x\n"):
    """Plant a plaintext secrets.env under tenants/<tid>/agents/<aid>/."""
    agent_dir = root / "tenants" / tid / "agents" / aid
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "secrets.env").write_text(body, encoding="utf-8")
    return agent_dir


def test_migrate_writes_enc_for_every_plaintext(tmp_path, patched_kms):
    _seed_agent_with_plain_secrets(tmp_path, "_default", "support")
    _seed_agent_with_plain_secrets(tmp_path, "acme", "ops")

    from tools.migrate_secrets_to_kms import migrate

    encrypted, skipped, errors = migrate(tmp_path)
    assert encrypted == 2
    assert skipped == 0
    assert errors == 0
    assert (tmp_path / "tenants" / "_default" / "agents" / "support" / "secrets.enc.json").is_file()
    assert (tmp_path / "tenants" / "acme" / "agents" / "ops" / "secrets.enc.json").is_file()


def test_migrate_is_idempotent(tmp_path, patched_kms):
    _seed_agent_with_plain_secrets(tmp_path, "_default", "support")

    from tools.migrate_secrets_to_kms import migrate

    enc1, sk1, er1 = migrate(tmp_path)
    enc2, sk2, er2 = migrate(tmp_path)
    assert enc1 == 1 and sk1 == 0
    assert enc2 == 0 and sk2 == 1  # second run sees the enc file → skip


def test_migrate_dry_run_does_not_write(tmp_path, patched_kms):
    _seed_agent_with_plain_secrets(tmp_path, "_default", "support")

    from tools.migrate_secrets_to_kms import migrate

    enc, sk, err = migrate(tmp_path, dry_run=True)
    assert enc == 1
    assert not (
        tmp_path / "tenants" / "_default" / "agents" / "support" / "secrets.enc.json"
    ).exists()


def test_migrate_with_delete_plaintext_removes_old_file(tmp_path, patched_kms):
    _seed_agent_with_plain_secrets(tmp_path, "_default", "support")

    from tools.migrate_secrets_to_kms import migrate

    migrate(tmp_path, delete_plaintext=True)
    agent_dir = tmp_path / "tenants" / "_default" / "agents" / "support"
    assert (agent_dir / "secrets.enc.json").is_file()
    assert not (agent_dir / "secrets.env").exists()


def test_migrate_round_trips_via_load_secrets(tmp_path, patched_kms):
    """After migration the secrets must still be readable through
    the normal load_secrets path."""
    _seed_agent_with_plain_secrets(
        tmp_path,
        "_default",
        "support",
        body="ANTHROPIC_API_KEY=sk-ant-secret\nOPENAI_API_KEY=sk-openai-secret\n",
    )

    from tools.migrate_secrets_to_kms import migrate
    from runtime.agents.secrets import load_secrets

    migrate(tmp_path, delete_plaintext=True)
    loaded = load_secrets(
        "support", root=tmp_path / "tenants" / "_default" / "agents"
    )
    assert loaded == {
        "ANTHROPIC_API_KEY": "sk-ant-secret",
        "OPENAI_API_KEY": "sk-openai-secret",
    }


def test_migrate_refuses_when_noop_crypto(tmp_path, monkeypatch):
    """If the operator forgot to set OPENTRACY_KMS_KEY_NAME, the
    factory returns Noop and migration would be pointless — refuse."""
    from runtime.crypto import NoopCrypto
    import tools.migrate_secrets_to_kms as _mig

    monkeypatch.setattr(_mig, "select_crypto", lambda: NoopCrypto())
    _seed_agent_with_plain_secrets(tmp_path, "_default", "support")
    enc, sk, er = _mig.migrate(tmp_path)
    assert er == 1
    # No file was written
    assert not (
        tmp_path / "tenants" / "_default" / "agents" / "support" / "secrets.enc.json"
    ).exists()


def test_migrate_handles_legacy_flat_agents_dir(tmp_path, patched_kms):
    """Pre-P16.1 layout (no tenants/ dir, just agents/) also gets migrated."""
    legacy_agent = tmp_path / "agents" / "support"
    legacy_agent.mkdir(parents=True)
    (legacy_agent / "secrets.env").write_text(
        "ANTHROPIC_API_KEY=sk-ant\n", encoding="utf-8"
    )

    from tools.migrate_secrets_to_kms import migrate

    enc, sk, er = migrate(tmp_path)
    assert enc == 1
    assert (legacy_agent / "secrets.enc.json").is_file()
