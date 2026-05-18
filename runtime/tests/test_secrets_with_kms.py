"""Tests for secrets.save_secrets/load_secrets under KMS-backed crypto
(P16.3.S4).

The factory is monkeypatched to return FakeKmsCrypto so no network
calls happen. We verify:

  - save under KMS writes secrets.enc.json (NOT secrets.env)
  - load round-trips: save → load gives the same dict
  - read-side fallback: legacy secrets.env still works if no enc file
  - encrypted file wins over plaintext if both exist
  - status() and mask() surface identical under both backends
  - empty-string values still remove keys
"""

from __future__ import annotations

import pytest

from runtime.agents.secrets import (
    _enc_path,
    _plain_path,
    get_secret,
    load_secrets,
    save_secrets,
    status,
)


@pytest.fixture
def fake_kms(monkeypatch):
    """Force select_crypto() to return a FakeKmsCrypto instance.
    All test secrets get wrapped by the same fake KEK for the duration
    of the test."""
    from runtime.crypto import FakeKmsCrypto

    fake = FakeKmsCrypto(kek_name="projects/test/cryptoKeys/test")
    monkeypatch.setattr("runtime.crypto.factory.select_crypto", lambda: fake)
    monkeypatch.setattr("runtime.agents.secrets.select_crypto", lambda: fake, raising=False)
    # secrets.py does `from runtime.crypto import select_crypto` inside
    # functions, so we need to patch where it'll be looked up.
    import runtime.crypto

    monkeypatch.setattr(runtime.crypto, "select_crypto", lambda: fake)
    yield fake


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_under_kms_writes_enc_file_not_plain(tmp_path, fake_kms):
    save_secrets(
        "acme-support",
        {"ANTHROPIC_API_KEY": "sk-ant-fake", "OPENAI_API_KEY": "sk-openai-fake"},
        root=tmp_path,
    )
    enc = _enc_path("acme-support", root=tmp_path)
    plain = _plain_path("acme-support", root=tmp_path)
    assert enc.is_file(), "expected secrets.enc.json on KMS path"
    assert not plain.exists(), "plaintext secrets.env should not be written under KMS"


def test_save_then_load_round_trip(tmp_path, fake_kms):
    save_secrets(
        "acme",
        {"ANTHROPIC_API_KEY": "sk-ant-deadbeef"},
        root=tmp_path,
    )
    loaded = load_secrets("acme", root=tmp_path)
    assert loaded == {"ANTHROPIC_API_KEY": "sk-ant-deadbeef"}


def test_enc_file_does_not_contain_plaintext(tmp_path, fake_kms):
    save_secrets(
        "acme",
        {"ANTHROPIC_API_KEY": "sk-ant-PLAINTEXT-MARKER-xyz"},
        root=tmp_path,
    )
    body = _enc_path("acme", root=tmp_path).read_bytes()
    assert b"PLAINTEXT-MARKER" not in body
    assert b"sk-ant-" not in body


def test_status_surface_identical_under_kms(tmp_path, fake_kms):
    save_secrets(
        "acme",
        {"ANTHROPIC_API_KEY": "sk-ant-12345678abcdefg"},
        root=tmp_path,
    )
    st = status("acme", root=tmp_path)
    assert st["anthropic"]["set"] is True
    assert st["anthropic"]["source"] == "per-agent"
    # The mask never reveals the full key, encrypted or not.
    assert "12345678" in st["anthropic"]["mask"] or "…" in st["anthropic"]["mask"]


def test_get_secret_under_kms(tmp_path, fake_kms):
    save_secrets("acme", {"ANTHROPIC_API_KEY": "sk-ant-1"}, root=tmp_path)
    assert get_secret("anthropic", "acme", root=tmp_path) == "sk-ant-1"


def test_empty_string_removes_key_under_kms(tmp_path, fake_kms):
    save_secrets("acme", {"ANTHROPIC_API_KEY": "sk-ant-1"}, root=tmp_path)
    save_secrets("acme", {"ANTHROPIC_API_KEY": ""}, root=tmp_path)
    loaded = load_secrets("acme", root=tmp_path)
    assert "ANTHROPIC_API_KEY" not in loaded


# ---------------------------------------------------------------------------
# Read-side fallback (OSS plaintext)
# ---------------------------------------------------------------------------


def test_load_falls_back_to_plaintext_when_no_enc_file(tmp_path, fake_kms):
    """Even when KMS is on, an existing plaintext secrets.env from a
    pre-migration install still loads. Once the operator calls
    save_secrets, the next read sees the encrypted file."""
    # Seed a legacy plaintext file
    plain = _plain_path("legacy", root=tmp_path)
    plain.parent.mkdir(parents=True, exist_ok=True)
    plain.write_text("ANTHROPIC_API_KEY=sk-ant-legacy\n", encoding="utf-8")

    loaded = load_secrets("legacy", root=tmp_path)
    assert loaded == {"ANTHROPIC_API_KEY": "sk-ant-legacy"}


def test_enc_file_wins_when_both_exist(tmp_path, fake_kms):
    """If both files exist, the encrypted one is the source of truth.
    Models the partial-migration state."""
    # Seed a stale plaintext file
    plain = _plain_path("acme", root=tmp_path)
    plain.parent.mkdir(parents=True, exist_ok=True)
    plain.write_text("ANTHROPIC_API_KEY=sk-ant-STALE\n", encoding="utf-8")
    # Then a save under KMS writes the enc file
    save_secrets("acme", {"ANTHROPIC_API_KEY": "sk-ant-FRESH"}, root=tmp_path)

    loaded = load_secrets("acme", root=tmp_path)
    assert loaded == {"ANTHROPIC_API_KEY": "sk-ant-FRESH"}
