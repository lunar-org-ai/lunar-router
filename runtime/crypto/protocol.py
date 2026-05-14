"""Crypto Protocol (P16.3).

Every backend implements the same two-method surface. Callers pass
``bytes`` in both directions; serialization to disk is the backend's
job. For envelope encryption, the on-disk shape is the
``Envelope`` JSON in :mod:`runtime.crypto.envelope`.
"""

from __future__ import annotations

from typing import Protocol


class Crypto(Protocol):
    """Pluggable encrypt/decrypt boundary."""

    def encrypt(self, plaintext: bytes) -> bytes:
        """Return ciphertext bytes. Caller persists them as-is."""
        ...

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Return plaintext bytes. Raises on tamper / wrong key."""
        ...
