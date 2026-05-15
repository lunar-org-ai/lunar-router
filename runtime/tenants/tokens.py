"""Per-tenant Bearer tokens (P16.1).

Format: ``otrcy_live_<base32_no_pad>`` where the random component
carries 160 bits of entropy (32 base32 chars). Plaintext is shown
ONCE at mint time and never persisted — only ``sha256(token)`` lives
on disk, both per-tenant in ``tenants/<tid>/tokens.json`` and in the
global lookup index ``tenants/_tokens_index.json``.

The global index is a flat ``{hash: tenant_id}`` map. It exists so the
ASGI tenant middleware can resolve an incoming token in O(1) without
scanning every tenant's file. The per-tenant file is the canonical
store; the index can be rebuilt from the per-tenant files via
:func:`rebuild_index`.

Token authentication is intentionally simple in P16.1:
  - constant-time hash comparison via :func:`hmac.compare_digest`
  - no expiry, no scopes — granted tokens grant full access to the
    tenant's data until explicitly revoked
  - operator admin token (``BACKEND_API_KEYS`` env) stays a separate
    layer for ``/v1/admin/*`` routes

Future phases (P16.3) layer KMS envelope encryption + scopes; this
module sticks to the minimum that works for staging deploy.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from runtime.tenants.types import TokenRecord

# Per-tenant locks serialize the read-modify-write cycle inside a
# single uvicorn worker. They don't help across Cloud Run instances —
# for that, `_load_tokens` below recovers from concat-corruption that
# concurrent writes from two instances can leave on gcsfuse. Together,
# the two layers eliminate the "Extra data" JSONDecodeError loop we
# saw at every Firebase login.
_TENANT_LOCKS: dict[str, threading.Lock] = {}
_TENANT_LOCKS_GUARD = threading.Lock()


def _lock_for(tenant_dir: Path) -> threading.Lock:
    key = str(tenant_dir)
    with _TENANT_LOCKS_GUARD:
        lock = _TENANT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _TENANT_LOCKS[key] = lock
        return lock


logger = logging.getLogger("runtime.tenants.tokens")


_DEFAULT_ROOT = Path("tenants")
_TOKENS_FILE = "tokens.json"
_INDEX_FILE = "_tokens_index.json"

_TOKEN_PREFIX = "otrcy_live_"
_TOKEN_ENTROPY_BYTES = 20         # 20 bytes → 32 base32 chars
_TOKEN_VISIBLE_RE_DESC = (
    f"{_TOKEN_PREFIX}<32 char a-z2-7>"
)


# ---------------------------------------------------------------------------
# Mint
# ---------------------------------------------------------------------------


def mint_token(
    tenant_id: str,
    label: str,
    *,
    root: Optional[Path] = None,
    now_iso: Optional[str] = None,
) -> tuple[str, TokenRecord]:
    """Create a new token for ``tenant_id`` and append it to that
    tenant's tokens.json (and the global index).

    Returns ``(plaintext, record)``. The plaintext is the ONLY place the
    raw token is ever returned — callers must show it to the operator
    once and forget it. Hash is what lives on disk.

    Raises ``FileNotFoundError`` if the tenant dir doesn't exist.
    """
    rroot = _resolve_root(root)
    tenant_dir = rroot / tenant_id
    if not tenant_dir.is_dir():
        raise FileNotFoundError(f"tenants/{tenant_id}/ missing on disk")

    plaintext = _generate_plaintext()
    record = TokenRecord(
        hash=_hash_token(plaintext),
        label=label,
        created_at=_now_iso(now_iso),
        last_used_at=None,
    )

    _append_token(tenant_dir, record)
    _index_set(rroot, record.hash, tenant_id)
    return plaintext, record


# ---------------------------------------------------------------------------
# Resolve (hot path — every authenticated request hits this)
# ---------------------------------------------------------------------------


def resolve_token(
    token: str,
    *,
    root: Optional[Path] = None,
    touch_last_used: bool = True,
    now_iso: Optional[str] = None,
) -> Optional[str]:
    """Return ``tenant_id`` for a plaintext token, or ``None`` if unknown.

    Comparison uses :func:`hmac.compare_digest` against the stored hash.
    On hit, touches ``last_used_at`` in the per-tenant file unless
    ``touch_last_used`` is False (e.g. health probes shouldn't write).
    """
    if not token or not token.startswith(_TOKEN_PREFIX):
        return None

    rroot = _resolve_root(root)
    index = _load_index(rroot)
    if not index:
        return None

    candidate_hash = _hash_token(token)

    # Index gives us tenant_id by exact hash match. The compare_digest
    # below is belt-and-suspenders for timing-leak resistance — dict
    # lookup is already constant-time on hash, but a future change might
    # add a fallback scan, so we keep the discipline.
    tenant_id: Optional[str] = None
    for stored_hash, tid in index.items():
        if hmac.compare_digest(stored_hash, candidate_hash):
            tenant_id = tid
            break
    if tenant_id is None:
        return None

    if touch_last_used:
        try:
            _touch_last_used(rroot / tenant_id, candidate_hash, _now_iso(now_iso))
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("failed to touch last_used_at: %s", e)

    return tenant_id


# ---------------------------------------------------------------------------
# Revoke + list
# ---------------------------------------------------------------------------


def revoke_token(
    tenant_id: str,
    hash_prefix: str,
    *,
    root: Optional[Path] = None,
) -> bool:
    """Remove the token whose hash starts with ``hash_prefix`` from this
    tenant's file and the global index. Returns True if a token was
    removed; False if no match.

    The prefix is the first 12 hex chars of the sha256 — exposed by the
    UI to operators so they don't need to handle the full hash."""
    rroot = _resolve_root(root)
    tenant_dir = rroot / tenant_id
    with _lock_for(tenant_dir):
        records = _load_tokens(tenant_dir)
        kept: list[TokenRecord] = []
        removed_hash: Optional[str] = None
        for rec in records:
            if rec.hash.startswith(hash_prefix):
                removed_hash = rec.hash
                continue
            kept.append(rec)
        if removed_hash is None:
            return False
        _save_tokens(tenant_dir, kept)
    _index_remove(rroot, removed_hash)
    return True


def list_tokens(
    tenant_id: str,
    *,
    root: Optional[Path] = None,
) -> list[TokenRecord]:
    """List every token for a tenant. Plaintext is never available
    after mint — operators see ``hash_prefix`` + label + timestamps."""
    return _load_tokens(_resolve_root(root) / tenant_id)


# ---------------------------------------------------------------------------
# Index maintenance
# ---------------------------------------------------------------------------


def rebuild_index(*, root: Optional[Path] = None) -> int:
    """Rebuild ``tenants/_tokens_index.json`` from every per-tenant
    ``tokens.json``. Returns the number of entries written.

    Use cases: recover from corrupted index, or after a manual file
    edit. Idempotent.
    """
    rroot = _resolve_root(root)
    rebuilt: dict[str, str] = {}
    if not rroot.is_dir():
        return 0
    for tenant_dir in sorted(rroot.iterdir()):
        if not tenant_dir.is_dir() or tenant_dir.name.startswith("_"):
            continue
        for rec in _load_tokens(tenant_dir):
            rebuilt[rec.hash] = tenant_dir.name
    _save_index(rroot, rebuilt)
    return len(rebuilt)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _generate_plaintext() -> str:
    raw = secrets.token_bytes(_TOKEN_ENTROPY_BYTES)
    b32 = base64.b32encode(raw).rstrip(b"=").lower().decode("ascii")
    return f"{_TOKEN_PREFIX}{b32}"


def _hash_token(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _load_tokens(tenant_dir: Path) -> list[TokenRecord]:
    path = tenant_dir / _TOKENS_FILE
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
        return [TokenRecord.from_dict(t) for t in (data.get("tokens") or [])]
    except json.JSONDecodeError:
        # Two writers on different Cloud Run instances raced and the
        # file ended up with multiple concatenated JSON objects (e.g.
        # `{"tokens":[…]}{"tokens":[…]}`). Recover by parsing each
        # object in turn and unioning their `tokens` arrays, deduped
        # by hash. Then rewrite the file so subsequent reads are fast.
        logger.warning(
            "tokens.json at %s had concatenated/corrupt JSON; recovering",
            path,
        )
        merged: dict[str, TokenRecord] = {}
        decoder = json.JSONDecoder()
        cursor = 0
        n = len(text)
        while cursor < n:
            # Skip whitespace between objects.
            while cursor < n and text[cursor] in " \t\r\n":
                cursor += 1
            if cursor >= n:
                break
            try:
                obj, end = decoder.raw_decode(text, cursor)
            except json.JSONDecodeError:
                # Trailing garbage we can't parse — stop here, accept
                # whatever we recovered so far.
                break
            cursor = end
            if isinstance(obj, dict):
                for t in obj.get("tokens") or []:
                    try:
                        rec = TokenRecord.from_dict(t)
                    except Exception:  # noqa: BLE001
                        continue
                    merged[rec.hash] = rec
        recovered = list(merged.values())
        # Best-effort rewrite — if the rewrite itself loses to another
        # racer, the next call will just recover again.
        try:
            _save_tokens(tenant_dir, recovered)
        except OSError as e:
            logger.warning("could not rewrite recovered tokens.json: %s", e)
        return recovered


def _save_tokens(tenant_dir: Path, records: list[TokenRecord]) -> None:
    tenant_dir.mkdir(parents=True, exist_ok=True)
    path = tenant_dir / _TOKENS_FILE
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            {"tokens": [r.to_dict() for r in records]},
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    tmp.replace(path)


def _append_token(tenant_dir: Path, record: TokenRecord) -> None:
    with _lock_for(tenant_dir):
        records = _load_tokens(tenant_dir)
        records.append(record)
        _save_tokens(tenant_dir, records)


def _touch_last_used(tenant_dir: Path, token_hash: str, when: str) -> None:
    with _lock_for(tenant_dir):
        records = _load_tokens(tenant_dir)
        changed = False
        for rec in records:
            if rec.hash == token_hash:
                rec.last_used_at = when
                changed = True
                break
        if changed:
            _save_tokens(tenant_dir, records)


def _load_index(root: Path) -> dict[str, str]:
    path = root / _INDEX_FILE
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("token index unreadable, returning empty: %s", e)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _save_index(root: Path, index: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / _INDEX_FILE
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def _index_set(root: Path, token_hash: str, tenant_id: str) -> None:
    index = _load_index(root)
    index[token_hash] = tenant_id
    _save_index(root, index)


def _index_remove(root: Path, token_hash: str) -> None:
    index = _load_index(root)
    if index.pop(token_hash, None) is not None:
        _save_index(root, index)


def _resolve_root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else _DEFAULT_ROOT


def _now_iso(override: Optional[str] = None) -> str:
    if override:
        return override
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
