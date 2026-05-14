"""Tests for GoogleKmsCrypto (P16.3.S3).

No live KMS is contacted — we pass a hand-rolled stub client into
the constructor. The stub mimics the two API methods we actually use
(``encrypt``, ``decrypt``) and lets us verify:

  - the wrapping logic calls the KMS client with the right resource
  - the envelope's kek + kek_version stamp the right values
  - encrypt → decrypt round-trips
  - mismatched kek field on decrypt is rejected
  - tampered ciphertext fails AES-GCM tag verification
"""

from __future__ import annotations

import json
import secrets
from types import SimpleNamespace

import pytest

from runtime.crypto.envelope import (
    aesgcm_decrypt,
    aesgcm_encrypt,
    fresh_nonce,
)
from runtime.crypto.google_kms import (
    GoogleKmsCrypto,
    _extract_kek_version,
)


# ---------------------------------------------------------------------------
# Stub KMS client
# ---------------------------------------------------------------------------


class _StubKmsClient:
    """Pretends to be a real KeyManagementServiceClient.

    Wraps the DEK with AES-GCM under a constant in-memory KEK. The
    ``name`` returned by ``encrypt`` mimics the real
    ``cryptoKeyVersions/<N>`` resource shape so version extraction can
    be tested too."""

    def __init__(self, *, kek: bytes | None = None, version_int: int = 7):
        self._kek = kek or secrets.token_bytes(32)
        self._version = version_int
        self.encrypt_calls: list[dict] = []
        self.decrypt_calls: list[dict] = []

    def encrypt(self, *, request: dict):
        self.encrypt_calls.append(request)
        nonce = fresh_nonce()
        body = aesgcm_encrypt(request["plaintext"], self._kek, nonce)
        return SimpleNamespace(
            ciphertext=nonce + body,
            name=f"{request['name']}/cryptoKeyVersions/{self._version}",
        )

    def decrypt(self, *, request: dict):
        self.decrypt_calls.append(request)
        blob = request["ciphertext"]
        nonce, body = blob[:12], blob[12:]
        plaintext = aesgcm_decrypt(body, self._kek, nonce)
        return SimpleNamespace(plaintext=plaintext)


_KEY_NAME = "projects/p/locations/l/keyRings/r/cryptoKeys/byok-master"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_google_kms_round_trip():
    client = _StubKmsClient()
    c = GoogleKmsCrypto(key_name=_KEY_NAME, client=client)
    ct = c.encrypt(b"ANTHROPIC_API_KEY=sk-...")
    assert client.encrypt_calls and client.encrypt_calls[0]["name"] == _KEY_NAME
    pt = c.decrypt(ct)
    assert pt == b"ANTHROPIC_API_KEY=sk-..."


def test_google_kms_envelope_stamps_kek_and_version():
    client = _StubKmsClient(version_int=42)
    c = GoogleKmsCrypto(key_name=_KEY_NAME, client=client)
    ct = c.encrypt(b"x")
    env = json.loads(ct.decode("utf-8"))
    assert env["kek"] == _KEY_NAME
    assert env["kek_version"] == 42


def test_google_kms_decrypt_rejects_mismatched_kek():
    """If the operator points at key B but the envelope was wrapped
    with key A, fail with a clear message rather than feeding a
    wrong-key blob to the KMS decrypt call."""
    client = _StubKmsClient()
    a = GoogleKmsCrypto(key_name=_KEY_NAME, client=client)
    b = GoogleKmsCrypto(
        key_name="projects/p/locations/l/keyRings/r/cryptoKeys/OTHER",
        client=client,
    )
    ct = a.encrypt(b"important")
    with pytest.raises(ValueError, match="wrapped with"):
        b.decrypt(ct)


def test_google_kms_tamper_detection():
    client = _StubKmsClient()
    c = GoogleKmsCrypto(key_name=_KEY_NAME, client=client)
    ct = c.encrypt(b"important")
    # Flip a byte inside the protected ciphertext_b64 field.
    env = json.loads(ct.decode("utf-8"))
    import base64

    raw = bytearray(base64.b64decode(env["ciphertext_b64"]))
    raw[0] ^= 0x42
    env["ciphertext_b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    tampered = json.dumps(env).encode("utf-8")
    with pytest.raises(Exception):
        c.decrypt(tampered)


def test_google_kms_each_encrypt_uses_a_fresh_dek_and_nonce():
    client = _StubKmsClient()
    c = GoogleKmsCrypto(key_name=_KEY_NAME, client=client)
    ct1 = c.encrypt(b"same plaintext")
    ct2 = c.encrypt(b"same plaintext")
    assert ct1 != ct2
    assert c.decrypt(ct1) == c.decrypt(ct2) == b"same plaintext"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("projects/p/.../cryptoKeyVersions/1", 1),
        ("projects/p/.../cryptoKeyVersions/42", 42),
        ("", 0),
        ("no-slashes-at-all", 0),
        ("ends/in/nothing/", 0),
        ("ends/in/not-a-number/x", 0),
    ],
)
def test_extract_kek_version(name, expected):
    assert _extract_kek_version(name) == expected


# ---------------------------------------------------------------------------
# Factory wiring (smoke)
# ---------------------------------------------------------------------------


def test_factory_returns_google_kms_when_env_set(monkeypatch):
    """Ensure ``select_crypto()`` picks up a real KMS env var and
    instantiates the right class — without ever calling KMS."""
    from runtime.crypto.factory import select_crypto

    monkeypatch.setenv("OPENTRACY_KMS_KEY_NAME", _KEY_NAME)

    # Stub out the client factory so no network call happens.
    monkeypatch.setattr(
        "runtime.crypto.google_kms._client_factory",
        lambda: _StubKmsClient(),
    )
    c = select_crypto()
    assert isinstance(c, GoogleKmsCrypto)
    # The round-trip still works through the factory-created instance.
    assert c.decrypt(c.encrypt(b"hello")) == b"hello"
