"""Google Cloud KMS envelope encryption (P16.3.S3).

Implements the same :class:`Crypto` Protocol as :class:`NoopCrypto`
and :class:`FakeKmsCrypto`, but uses ``google-cloud-kms`` to wrap
the DEK in a real cloud-hosted KEK.

This module is **lazy-imported** by :mod:`runtime.crypto.factory` —
OSS clones that never set ``OPENTRACY_KMS_KEY_NAME`` don't need
``google-cloud-kms`` installed. The dependency lives in the ``kms``
optional extra in ``pyproject.toml``.

Threat model: see ``PLAN_P16.3.md``. At-rest leak protected; in-memory
leak out of scope.
"""

from __future__ import annotations

import logging
from typing import Any

from runtime.crypto.envelope import (
    Envelope,
    aesgcm_decrypt,
    aesgcm_encrypt,
    decode_envelope,
    encode_envelope,
    fresh_dek,
    fresh_nonce,
)


logger = logging.getLogger("runtime.crypto.google_kms")


# Allow tests to inject a fake KMS client without touching network.
# ``_client_factory()`` returns the real client by default; tests can
# monkeypatch this attribute on the module to return their own stub.
def _client_factory() -> Any:
    """Lazy-create a ``KeyManagementServiceClient``. Importing
    ``google.cloud.kms`` here means OSS clones never load it unless
    KMS is actually selected by the factory."""
    from google.cloud import kms  # type: ignore[import-not-found]

    return kms.KeyManagementServiceClient()


class GoogleKmsCrypto:
    """Production envelope-encryption backend.

    ``key_name`` is the full KMS resource name, e.g.
    ``projects/opentracy-train/locations/us-east4/keyRings/opentracy-byok/cryptoKeys/byok-master``.
    Caller (the factory) is responsible for validating that shape
    before instantiating us.
    """

    name = "google-kms"

    def __init__(self, *, key_name: str, client: Any | None = None) -> None:
        self._key_name = key_name
        # Late-bind the client so import order doesn't pull in
        # google-cloud-kms when callers go through the factory's noop
        # path. Tests can pass ``client=stub`` directly to skip the
        # default factory.
        self._client = client

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _client_factory()
        return self._client

    def _wrap_dek(self, dek: bytes) -> tuple[bytes, int]:
        """Call KMS to encrypt the DEK with the configured KEK.

        Returns ``(wrapped_dek_bytes, kek_version)``. The version
        mirrors the ``name`` field from ``kms.encrypt`` response so we
        can stamp it in the envelope for future rotation handling.
        """
        resp = self.client.encrypt(
            request={"name": self._key_name, "plaintext": dek}
        )
        # The ``name`` on the response is a key VERSION resource:
        #   projects/.../cryptoKeys/byok-master/cryptoKeyVersions/3
        # We want the integer suffix only.
        version = _extract_kek_version(getattr(resp, "name", "") or "")
        return resp.ciphertext, version

    def _unwrap_dek(self, wrapped_dek: bytes) -> bytes:
        resp = self.client.decrypt(
            request={"name": self._key_name, "ciphertext": wrapped_dek}
        )
        return resp.plaintext

    # ------------------------------------------------------------------
    # Crypto Protocol
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes) -> bytes:
        dek = fresh_dek()
        nonce = fresh_nonce()
        try:
            wrapped, version = self._wrap_dek(dek)
        except Exception as e:
            logger.error("kms wrap failed: %s", e)
            raise
        ciphertext = aesgcm_encrypt(plaintext, dek, nonce)
        env = Envelope(
            kek=self._key_name,
            kek_version=version or 1,
            nonce=nonce,
            wrapped_dek=wrapped,
            ciphertext=ciphertext,
        )
        return encode_envelope(env)

    def decrypt(self, ciphertext: bytes) -> bytes:
        env = decode_envelope(ciphertext)
        if env.kek != self._key_name:
            # Stamp mismatch — operator pointed at a different key than
            # what wrote the envelope. Make the failure crystal-clear.
            raise ValueError(
                f"envelope was wrapped with {env.kek!r}, "
                f"but this client is configured for {self._key_name!r}"
            )
        try:
            dek = self._unwrap_dek(env.wrapped_dek)
        except Exception as e:
            logger.error("kms unwrap failed: %s", e)
            raise
        return aesgcm_decrypt(env.ciphertext, dek, env.nonce)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_kek_version(name: str) -> int:
    """Pull the trailing integer from a key-version resource name.

    ``.../cryptoKeyVersions/3`` → 3. Returns 0 if the shape doesn't
    match — caller treats that as "unknown" and stamps 1.
    """
    if not name:
        return 0
    parts = name.rsplit("/", 1)
    if len(parts) != 2:
        return 0
    tail = parts[1].strip()
    if not tail.isdigit():
        return 0
    return int(tail)
