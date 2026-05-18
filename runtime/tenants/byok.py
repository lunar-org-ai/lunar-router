"""Per-tenant BYOK provider keys (P16.4).

Each tenant brings their own Anthropic/OpenAI/etc keys. Plaintext never
touches disk — we encrypt with the platform-level KMS (envelope) and
store one file per provider under ``tenants/<t>/byok/<provider>.enc``.

Reads decrypt on demand; the LLM call path in
``runtime.store.onboarding_chat`` and the live agent's generate stage
fall back to ``get_provider_key()`` when the matching env var is unset.

Fail-closed: in multi-tenant mode there's no platform fallback. If a
tenant hasn't pasted their key, calls into Anthropic short-circuit to
the offline "[offline]" response so we never accidentally bill the
operator account for tenant traffic (see :mod:`memory/byok_strict`).

Storage shape on disk (one file per provider):

    {
      "provider": "anthropic",
      "ciphertext_b64": "<base64 envelope>",
      "mask": "sk-ant-…1A2B",
      "created_at": "2026-05-15T13:55:00Z",
      "rotated_at": null,
      "last_used_at": null
    }

The ``mask`` is computed from the plaintext at write time and is the
only thing the UI ever reads back; the rest of the file requires KMS
decrypt access.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from runtime.crypto import select_crypto
from runtime.tenants.registry import get_tenant_dir


logger = logging.getLogger("runtime.tenants.byok")


SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai"})


def _byok_dir(tenant_id: str) -> Path:
    return get_tenant_dir(tenant_id) / "byok"


def _key_path(tenant_id: str, provider: str) -> Path:
    return _byok_dir(tenant_id) / f"{provider}.json"


def _mask(plaintext: str) -> str:
    """Show enough of the key to be recognizable without exposing it.

    Anthropic keys are ``sk-ant-…``, OpenAI are ``sk-…`` — both prefixes
    are public, the random tail is the secret. Mask the middle.
    """
    if len(plaintext) <= 12:
        return "•" * len(plaintext)
    head = plaintext[:7]
    tail = plaintext[-4:]
    return f"{head}…{tail}"


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def set_provider_key(
    tenant_id: str,
    provider: str,
    plaintext: str,
) -> dict[str, str]:
    """Encrypt + persist a provider key for one tenant.

    Returns the public metadata (mask, created_at) — never the
    plaintext or ciphertext. Overwrites any existing key for the same
    provider; callers wanting rotation semantics should read the
    existing record first to capture ``rotated_at``.
    """
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported provider {provider!r}; expected one of "
            f"{sorted(SUPPORTED_PROVIDERS)}"
        )
    plaintext = (plaintext or "").strip()
    if not plaintext:
        raise ValueError("plaintext key is empty")

    crypto = select_crypto()
    ciphertext = crypto.encrypt(plaintext.encode("utf-8"))
    ciphertext_b64 = base64.b64encode(ciphertext).decode("ascii")

    path = _key_path(tenant_id, provider)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve created_at if a record exists (rotation case).
    existing = _load_record(path)
    created_at = existing.get("created_at") if existing else _now_iso()
    rotated_at = _now_iso() if existing else None

    record = {
        "provider": provider,
        "ciphertext_b64": ciphertext_b64,
        "mask": _mask(plaintext),
        "created_at": created_at,
        "rotated_at": rotated_at,
        "last_used_at": None,
    }
    _write_atomic(path, record)
    return {"mask": record["mask"], "created_at": created_at}


def get_provider_key(tenant_id: str, provider: str) -> Optional[str]:
    """Decrypt + return the plaintext key for the LLM caller, or None.

    None signals "not configured" — callers should fail-closed in
    multi-tenant mode rather than fall back to a platform key.
    """
    record = _load_record(_key_path(tenant_id, provider))
    if record is None:
        return None
    blob = record.get("ciphertext_b64")
    if not isinstance(blob, str):
        return None
    try:
        ciphertext = base64.b64decode(blob)
        plaintext = select_crypto().decrypt(ciphertext)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "byok decrypt failed for tenant=%s provider=%s: %s",
            tenant_id, provider, e,
        )
        return None
    # best-effort last_used_at — don't fail the call if the bookkeeping
    # write fails (read-only mount, race, etc).
    try:
        record["last_used_at"] = _now_iso()
        _write_atomic(_key_path(tenant_id, provider), record)
    except OSError:
        pass
    return plaintext.decode("utf-8")


def delete_provider_key(tenant_id: str, provider: str) -> bool:
    """Remove the stored key. Returns True if a record existed."""
    path = _key_path(tenant_id, provider)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def list_provider_keys(tenant_id: str) -> list[dict[str, Optional[str]]]:
    """Public listing — masks and timestamps only, no ciphertext."""
    base = _byok_dir(tenant_id)
    if not base.is_dir():
        return []
    out: list[dict[str, Optional[str]]] = []
    for path in sorted(base.glob("*.json")):
        record = _load_record(path)
        if record is None:
            continue
        out.append({
            "provider": record.get("provider"),
            "mask": record.get("mask"),
            "created_at": record.get("created_at"),
            "rotated_at": record.get("rotated_at"),
            "last_used_at": record.get("last_used_at"),
        })
    return out


# ─── Internal helpers ─────────────────────────────────────────────


def _load_record(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("byok: failed to read %s: %s", path, e)
        return None


def _write_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
