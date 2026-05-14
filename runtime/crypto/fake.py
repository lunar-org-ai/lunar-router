"""In-memory KMS mock for tests (P16.3.S2).

Implements the envelope encryption flow with a **static fake KEK
held in memory** — no network, no SDK call, no IAM. Useful only in
tests; the factory never selects it.

Each :class:`FakeKmsCrypto` instance carries:

  - a 32-byte fake KEK (random per instance unless one is passed in)
  - a stable "kek name" string for the envelope's ``kek`` field

Two instances with different fake KEKs cannot decrypt each other's
ciphertext — verified by the test suite. That gives tests a way to
exercise the tamper-resistance contract without standing up real KMS.
"""

from __future__ import annotations

import secrets

from runtime.crypto.envelope import (
    Envelope,
    aesgcm_decrypt,
    aesgcm_encrypt,
    decode_envelope,
    encode_envelope,
    fresh_dek,
    fresh_nonce,
)


_KEK_BYTES = 32  # AES-256 — matches the DEK size for symmetric wrapping


class FakeKmsCrypto:
    """Envelope encryption with an in-memory fake KEK.

    The ``encrypt`` and ``decrypt`` methods follow the same
    envelope shape :class:`GoogleKmsCrypto` will produce, so callers
    can swap backends transparently.
    """

    name = "fake-kms"

    def __init__(
        self,
        *,
        kek: bytes | None = None,
        kek_name: str = "projects/fake/locations/test/keyRings/test/cryptoKeys/test",
        kek_version: int = 1,
    ) -> None:
        if kek is None:
            kek = secrets.token_bytes(_KEK_BYTES)
        if len(kek) != _KEK_BYTES:
            raise ValueError(f"fake KEK must be {_KEK_BYTES} bytes")
        self._kek = kek
        self._kek_name = kek_name
        self._kek_version = kek_version

    # ------------------------------------------------------------------
    # Crypto Protocol
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes) -> bytes:
        dek = fresh_dek()
        nonce = fresh_nonce()
        ciphertext = aesgcm_encrypt(plaintext, dek, nonce)
        # Wrap the DEK using the KEK. We use the same AES-GCM primitive
        # with a separate nonce; a real KMS would do the equivalent in
        # the cloud HSM.
        wrap_nonce = fresh_nonce()
        wrapped = wrap_nonce + aesgcm_encrypt(dek, self._kek, wrap_nonce)
        env = Envelope(
            kek=self._kek_name,
            kek_version=self._kek_version,
            nonce=nonce,
            wrapped_dek=wrapped,
            ciphertext=ciphertext,
        )
        return encode_envelope(env)

    def decrypt(self, ciphertext: bytes) -> bytes:
        env = decode_envelope(ciphertext)
        # The wrapped DEK carries its own 12-byte nonce prefix.
        wrap_nonce = env.wrapped_dek[:12]
        wrapped_body = env.wrapped_dek[12:]
        dek = aesgcm_decrypt(wrapped_body, self._kek, wrap_nonce)
        return aesgcm_decrypt(env.ciphertext, dek, env.nonce)
