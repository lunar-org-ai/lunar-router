"""NoopCrypto — pass-through encrypt/decrypt for OSS local (P16.3).

When no KMS env var is set, ``select_crypto()`` returns this. The on-disk
``secrets.env`` stays plaintext, exactly as P3.1 shipped it. The only
visible effect of having this Class at all is that callers can program
against the :class:`Crypto` Protocol regardless of whether KMS is on.
"""

from __future__ import annotations


class NoopCrypto:
    """``encrypt(x)`` and ``decrypt(x)`` both return ``x``.

    Used by every test that doesn't explicitly want envelope behavior,
    and by the OSS default at runtime. The contract is that the bytes
    returned by ``encrypt`` are safe to persist verbatim and feed back
    into ``decrypt`` to get the original plaintext.
    """

    name = "noop"

    def encrypt(self, plaintext: bytes) -> bytes:
        return plaintext

    def decrypt(self, ciphertext: bytes) -> bytes:
        return ciphertext
