"""On-disk envelope format + AES-GCM helpers (P16.3).

Envelope layout (the bytes a KMS-backed ``Crypto.encrypt`` returns):

    {
      "v": 1,                     // format version
      "kek": "projects/...",      // KMS key name used to wrap the DEK
      "kek_version": 1,           // mirrored from kms.encrypt response
      "nonce_b64": "<12 bytes>",  // AES-GCM nonce
      "wrapped_dek_b64": "<…>",   // DEK encrypted by KEK
      "ciphertext_b64": "<…>"     // AES-GCM(plaintext, DEK, nonce)
    }

We base64-encode binary fields so the whole envelope lives in a single
``json.dumps``'d string. Nonces are generated per-encrypt with
``secrets.token_bytes(12)`` — never reused.
"""

from __future__ import annotations

import base64
import json
import secrets
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


_ENVELOPE_VERSION = 1
_NONCE_BYTES = 12
_DEK_BYTES = 32  # AES-256


@dataclass
class Envelope:
    """The JSON-shaped wire format. Fields match :data:`_ENVELOPE_VERSION = 1`."""

    kek: str
    kek_version: int
    nonce: bytes
    wrapped_dek: bytes
    ciphertext: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": _ENVELOPE_VERSION,
            "kek": self.kek,
            "kek_version": self.kek_version,
            "nonce_b64": _b64e(self.nonce),
            "wrapped_dek_b64": _b64e(self.wrapped_dek),
            "ciphertext_b64": _b64e(self.ciphertext),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Envelope":
        v = data.get("v")
        if v != _ENVELOPE_VERSION:
            raise ValueError(f"unsupported envelope version: {v!r}")
        return cls(
            kek=str(data["kek"]),
            kek_version=int(data.get("kek_version", 1)),
            nonce=_b64d(data["nonce_b64"]),
            wrapped_dek=_b64d(data["wrapped_dek_b64"]),
            ciphertext=_b64d(data["ciphertext_b64"]),
        )


def encode_envelope(env: Envelope) -> bytes:
    """JSON-serialize ``env`` into the byte blob a backend's ``encrypt``
    method returns (and ``decrypt`` consumes)."""
    return (json.dumps(env.to_dict(), indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def decode_envelope(blob: bytes) -> Envelope:
    return Envelope.from_dict(json.loads(blob.decode("utf-8")))


# ---------------------------------------------------------------------------
# AES-GCM helpers
# ---------------------------------------------------------------------------


def fresh_dek() -> bytes:
    """Generate a 32-byte DEK from CSPRNG."""
    return secrets.token_bytes(_DEK_BYTES)


def fresh_nonce() -> bytes:
    """Generate a 12-byte AES-GCM nonce. Never reuse with the same DEK."""
    return secrets.token_bytes(_NONCE_BYTES)


def aesgcm_encrypt(plaintext: bytes, dek: bytes, nonce: bytes) -> bytes:
    """AES-256-GCM ciphertext (including 16-byte tag appended).

    Uses the audited cryptography.AESGCM implementation — no custom
    crypto in this module."""
    if len(dek) != _DEK_BYTES:
        raise ValueError(f"DEK must be {_DEK_BYTES} bytes, got {len(dek)}")
    if len(nonce) != _NONCE_BYTES:
        raise ValueError(f"nonce must be {_NONCE_BYTES} bytes, got {len(nonce)}")
    return AESGCM(dek).encrypt(nonce, plaintext, associated_data=None)


def aesgcm_decrypt(ciphertext: bytes, dek: bytes, nonce: bytes) -> bytes:
    """Inverse of :func:`aesgcm_encrypt`. Raises on tampering."""
    if len(dek) != _DEK_BYTES:
        raise ValueError(f"DEK must be {_DEK_BYTES} bytes, got {len(dek)}")
    if len(nonce) != _NONCE_BYTES:
        raise ValueError(f"nonce must be {_NONCE_BYTES} bytes, got {len(nonce)}")
    return AESGCM(dek).decrypt(nonce, ciphertext, associated_data=None)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))
