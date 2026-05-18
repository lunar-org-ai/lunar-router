"""Tests for the crypto scaffolding (P16.3.S1)."""

from __future__ import annotations

import json

import pytest

from runtime.crypto import (
    Envelope,
    NoopCrypto,
    aesgcm_decrypt,
    aesgcm_encrypt,
    decode_envelope,
    encode_envelope,
    select_crypto,
)
from runtime.crypto.envelope import fresh_dek, fresh_nonce


# ---------------------------------------------------------------------------
# AES-GCM helpers
# ---------------------------------------------------------------------------


def test_aesgcm_round_trip():
    dek = fresh_dek()
    nonce = fresh_nonce()
    plaintext = b"OPENAI_API_KEY=sk-...\nANTHROPIC_API_KEY=sk-ant-...\n"
    ct = aesgcm_encrypt(plaintext, dek, nonce)
    assert ct != plaintext  # actually encrypted
    pt = aesgcm_decrypt(ct, dek, nonce)
    assert pt == plaintext


def test_aesgcm_tamper_detection():
    dek = fresh_dek()
    nonce = fresh_nonce()
    ct = aesgcm_encrypt(b"hello", dek, nonce)
    # Flip a single byte in the ciphertext — GCM tag fails.
    bad = bytearray(ct)
    bad[0] ^= 0x42
    with pytest.raises(Exception):  # cryptography raises InvalidTag
        aesgcm_decrypt(bytes(bad), dek, nonce)


def test_aesgcm_rejects_wrong_dek_size():
    with pytest.raises(ValueError, match="DEK must be"):
        aesgcm_encrypt(b"x", b"too-short", fresh_nonce())


def test_aesgcm_rejects_wrong_nonce_size():
    with pytest.raises(ValueError, match="nonce must be"):
        aesgcm_encrypt(b"x", fresh_dek(), b"short")


def test_dek_and_nonce_freshness():
    """Repeated calls must give different bytes — sanity check on CSPRNG."""
    d1, d2 = fresh_dek(), fresh_dek()
    n1, n2 = fresh_nonce(), fresh_nonce()
    assert d1 != d2
    assert n1 != n2


# ---------------------------------------------------------------------------
# Envelope ser/de
# ---------------------------------------------------------------------------


def test_envelope_round_trip():
    env = Envelope(
        kek="projects/p/locations/l/keyRings/r/cryptoKeys/k",
        kek_version=2,
        nonce=b"\x00" * 12,
        wrapped_dek=b"wrapped-bytes",
        ciphertext=b"cipher-bytes",
    )
    blob = encode_envelope(env)
    parsed = decode_envelope(blob)
    assert parsed.kek == env.kek
    assert parsed.kek_version == 2
    assert parsed.nonce == env.nonce
    assert parsed.wrapped_dek == env.wrapped_dek
    assert parsed.ciphertext == env.ciphertext


def test_envelope_is_pretty_printed_json():
    """We persist to disk; multi-line indented JSON makes diffs readable."""
    env = Envelope(
        kek="projects/p/locations/l/keyRings/r/cryptoKeys/k",
        kek_version=1,
        nonce=b"\x00" * 12,
        wrapped_dek=b"w",
        ciphertext=b"c",
    )
    body = encode_envelope(env).decode("utf-8")
    assert body.count("\n") > 1
    parsed = json.loads(body)
    assert parsed["v"] == 1
    assert parsed["kek_version"] == 1


def test_envelope_rejects_unknown_version():
    bad = json.dumps({"v": 99}).encode("utf-8")
    with pytest.raises(ValueError, match="unsupported envelope version"):
        decode_envelope(bad)


# ---------------------------------------------------------------------------
# NoopCrypto
# ---------------------------------------------------------------------------


def test_noop_is_pass_through():
    c = NoopCrypto()
    payload = b"ANTHROPIC_API_KEY=sk-..."
    assert c.encrypt(payload) is payload or c.encrypt(payload) == payload
    assert c.decrypt(payload) is payload or c.decrypt(payload) == payload


def test_noop_round_trip_through_encrypt_decrypt():
    c = NoopCrypto()
    payload = b"x" * 1000
    assert c.decrypt(c.encrypt(payload)) == payload


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_returns_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv("OPENTRACY_KMS_KEY_NAME", raising=False)
    c = select_crypto()
    assert isinstance(c, NoopCrypto)


def test_factory_returns_noop_when_env_blank(monkeypatch):
    monkeypatch.setenv("OPENTRACY_KMS_KEY_NAME", "   ")
    c = select_crypto()
    assert isinstance(c, NoopCrypto)


def test_factory_rejects_garbage_value(monkeypatch):
    """Fail loud rather than fall back silently when env is set to
    something that isn't a KMS resource name."""
    monkeypatch.setenv("OPENTRACY_KMS_KEY_NAME", "not-a-kms-name")
    with pytest.raises(ValueError, match="doesn't look like"):
        select_crypto()
