"""Tests for FakeKmsCrypto (P16.3.S2).

Pins the envelope encryption contract that production GoogleKmsCrypto
will follow. Two instances with distinct fake KEKs MUST NOT be able
to decrypt each other's output.
"""

from __future__ import annotations

import json

import pytest

from runtime.crypto import FakeKmsCrypto, decode_envelope


def test_fake_kms_round_trip():
    c = FakeKmsCrypto()
    plaintext = b"ANTHROPIC_API_KEY=sk-ant-real-key\nOPENAI_API_KEY=sk-openai\n"
    ct = c.encrypt(plaintext)
    assert plaintext not in ct  # actually encrypted
    pt = c.decrypt(ct)
    assert pt == plaintext


def test_fake_kms_blob_is_envelope_json():
    """Ciphertext format is the on-disk envelope shape we promised."""
    c = FakeKmsCrypto(kek_name="projects/fake/locations/x/keyRings/r/cryptoKeys/k")
    ct = c.encrypt(b"hello")
    body = json.loads(ct.decode("utf-8"))
    assert body["v"] == 1
    assert body["kek"] == "projects/fake/locations/x/keyRings/r/cryptoKeys/k"
    assert body["kek_version"] == 1
    assert all(
        body[k] for k in ("nonce_b64", "wrapped_dek_b64", "ciphertext_b64")
    )


def test_fake_kms_each_encrypt_uses_a_fresh_dek_and_nonce():
    c = FakeKmsCrypto()
    ct1 = c.encrypt(b"identical plaintext")
    ct2 = c.encrypt(b"identical plaintext")
    # Same plaintext → different ciphertext (DEK + nonce change each time).
    assert ct1 != ct2
    # And both decrypt back to the same plaintext.
    assert c.decrypt(ct1) == c.decrypt(ct2) == b"identical plaintext"


def test_fake_kms_cross_instance_isolation():
    """Two fake KMS instances with different fake-KEKs cannot decrypt
    each other's ciphertext — proves the wrapping actually depends on
    the KEK."""
    a = FakeKmsCrypto()
    b = FakeKmsCrypto()
    ct = a.encrypt(b"secret stuff")
    with pytest.raises(Exception):  # InvalidTag from AES-GCM
        b.decrypt(ct)


def test_fake_kms_same_kek_can_decrypt():
    """If you share the fake-KEK bytes, the two instances are
    effectively the same key — used to model rotation tests later."""
    import secrets as _secrets

    kek = _secrets.token_bytes(32)
    a = FakeKmsCrypto(kek=kek, kek_name="projects/fake/cryptoKeys/k")
    b = FakeKmsCrypto(kek=kek, kek_name="projects/fake/cryptoKeys/k")
    ct = a.encrypt(b"x")
    assert b.decrypt(ct) == b"x"


def test_fake_kms_tamper_detection():
    """Tampering with the AES-GCM ciphertext bytes (inside the envelope)
    must fail decryption with an AEAD tag-mismatch error."""
    c = FakeKmsCrypto()
    ct = c.encrypt(b"important")
    # Decode the envelope, flip a byte in the actual ciphertext field,
    # re-encode. This guarantees we're hitting the crypto-protected
    # bytes, not metadata fields like kek_name that AES-GCM doesn't
    # authenticate.
    env = json.loads(ct.decode("utf-8"))
    import base64

    raw_ct = bytearray(base64.b64decode(env["ciphertext_b64"]))
    raw_ct[0] ^= 0x42
    env["ciphertext_b64"] = base64.b64encode(bytes(raw_ct)).decode("ascii")
    tampered = json.dumps(env).encode("utf-8")
    with pytest.raises(Exception):
        c.decrypt(tampered)


def test_fake_kms_envelope_decodes_to_known_kek_name():
    c = FakeKmsCrypto(kek_name="projects/a/locations/b/keyRings/c/cryptoKeys/d")
    env = decode_envelope(c.encrypt(b"x"))
    assert env.kek == "projects/a/locations/b/keyRings/c/cryptoKeys/d"


def test_fake_kms_rejects_wrong_kek_size():
    with pytest.raises(ValueError, match="fake KEK must be"):
        FakeKmsCrypto(kek=b"too-short")
