"""Crypto backend selection (P16.3).

Reads env on every call so tests can flip via ``monkeypatch.setenv``
without restarting the process.

Decision tree:

  OPENTRACY_KMS_KEY_NAME unset → :class:`NoopCrypto`
  OPENTRACY_KMS_KEY_NAME set + looks like ``projects/…`` → :class:`GoogleKmsCrypto`
  OPENTRACY_KMS_KEY_NAME set to anything else → raise (fail loud)

Note that there's no ``FakeKmsCrypto`` branch — tests instantiate
that one directly to avoid any chance of the operator's real KMS
config leaking into the suite.
"""

from __future__ import annotations

import os

from runtime.crypto.noop import NoopCrypto
from runtime.crypto.protocol import Crypto


_ENV_KEY_NAME = "OPENTRACY_KMS_KEY_NAME"
_KMS_NAME_PREFIX = "projects/"


def select_crypto() -> Crypto:
    """Pick a Crypto backend from the environment."""
    raw = (os.environ.get(_ENV_KEY_NAME) or "").strip()
    if not raw:
        return NoopCrypto()
    if not raw.startswith(_KMS_NAME_PREFIX):
        raise ValueError(
            f"{_ENV_KEY_NAME}={raw!r} doesn't look like a KMS key resource name "
            f"(expected '{_KMS_NAME_PREFIX}…/cryptoKeys/<name>'). "
            "If you intended OSS plaintext mode, unset the variable; "
            "if you intended hosted KMS, fix the value."
        )
    # Lazy import so OSS clones without google-cloud-kms installed don't
    # blow up at import time. The kms extra in pyproject pulls this in.
    from runtime.crypto.google_kms import GoogleKmsCrypto

    return GoogleKmsCrypto(key_name=raw)
