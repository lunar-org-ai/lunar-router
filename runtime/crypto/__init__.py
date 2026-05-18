"""Envelope encryption for at-rest secrets (P16.3).

Three backends, selected by :func:`select_crypto` from env:

  - :class:`NoopCrypto` — pass-through. OSS default; preserves plaintext
    ``secrets.env`` behavior.
  - :class:`FakeKmsCrypto` — in-memory mock with a static fake KEK.
    Tests only — never invoked by production code paths.
  - :class:`GoogleKmsCrypto` — real envelope encryption via
    ``google-cloud-kms``. Activated by ``OPENTRACY_KMS_KEY_NAME``.

All three implement the same :class:`Crypto` Protocol so callers
(currently just ``runtime/agents/secrets.py``) treat them
interchangeably.
"""

from runtime.crypto.envelope import (  # noqa: F401
    Envelope,
    aesgcm_decrypt,
    aesgcm_encrypt,
    decode_envelope,
    encode_envelope,
)
from runtime.crypto.factory import select_crypto  # noqa: F401
from runtime.crypto.fake import FakeKmsCrypto  # noqa: F401
from runtime.crypto.noop import NoopCrypto  # noqa: F401
from runtime.crypto.protocol import Crypto  # noqa: F401
