"""Per-agent BYOK secrets (P3.1 + P16.3).

Each agent stores provider keys per the encryption backend selected
by ``runtime.crypto.select_crypto()``:

  - **Noop** (OSS local default): ``agents/<id>/secrets.env`` —
    plain KEY=VALUE text, mode 0600. Identical to P3.1.
  - **GoogleKmsCrypto** (hosted, ``OPENTRACY_KMS_KEY_NAME`` set):
    ``agents/<id>/secrets.enc.json`` — KMS-wrapped DEK + AES-GCM
    ciphertext of the same dotenv body. Plaintext file is gone.

The format chosen at *write* time is whatever the factory returns.
At *read* time we prefer the encrypted file when both are present,
falling back to plaintext for OSS / pre-migration installs.

Two paths to a key:

  1) Per-agent (preferred): file as above, either plaintext or
     encrypted. Each agent can use its operator's own key without
     polluting the global env.

  2) Global fallback (legacy): values in ``os.environ`` (loaded
     from the repo's ``.env`` during server startup). Lets the dev
     install keep working with a single key for everything.

Resolution helper ``get_secret(provider, agent_id)`` walks (1) → (2)
and returns ``None`` if neither is set. The generate stage uses this
to pick which SDK to dispatch on + which key to authenticate with.

Masking: ``mask_key("sk-ant-api03-abc...xyz")`` → ``"sk-ant-…xyz"``.
The UI never sees the raw key — server returns the mask on read and
only writes when given a fresh value.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


logger = logging.getLogger("runtime.agents.secrets")


# Provider → env var conventions. The first entry is the canonical
# variable name; aliases are honored on read for compat with tools that
# use different casing.
PROVIDERS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
    "openai": ["OPENAI_API_KEY"],
}

KNOWN_PROVIDERS = tuple(PROVIDERS.keys())


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


_PLAIN_FILENAME = "secrets.env"
_ENC_FILENAME = "secrets.enc.json"


def _agent_dir(agent_id: str, *, root: Optional[Path] = None) -> Path:
    """The per-agent directory both secret files live under."""
    if root is not None:
        return Path(root) / agent_id
    try:
        from runtime.agents import registry as _reg
        return _reg.agents_root() / agent_id
    except Exception:
        return Path("agents") / agent_id


def _plain_path(agent_id: str, *, root: Optional[Path] = None) -> Path:
    return _agent_dir(agent_id, root=root) / _PLAIN_FILENAME


def _enc_path(agent_id: str, *, root: Optional[Path] = None) -> Path:
    return _agent_dir(agent_id, root=root) / _ENC_FILENAME


# Legacy alias — some tests still call ``_secrets_path``.
def _secrets_path(agent_id: str, *, root: Optional[Path] = None) -> Path:
    """Resolve the on-disk path that load/save will use *right now*.

    Returns ``secrets.enc.json`` when one of the encrypted backends
    is selected, ``secrets.env`` otherwise. Read-side fallback (see
    :func:`load_secrets`) still works regardless of which one we
    return."""
    from runtime.crypto import NoopCrypto, select_crypto
    crypto = select_crypto()
    if isinstance(crypto, NoopCrypto):
        return _plain_path(agent_id, root=root)
    return _enc_path(agent_id, root=root)


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def _parse_dotenv(text: str) -> dict[str, str]:
    """Lenient KEY=VALUE parser shared by both plaintext + decrypted paths.

    Skips blank lines + ``#`` comments, strips matching quote pairs,
    ignores malformed lines instead of raising. Identical to
    ``runtime/dotenv.py`` behavior so operator intuition transfers."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        out[key] = value
    return out


def _serialize_dotenv(merged: dict[str, str]) -> str:
    body_lines = [
        "# Agent BYOK secrets — gitignored, never commit.",
        "# Provider keys: ANTHROPIC_API_KEY, OPENAI_API_KEY.",
        "",
    ]
    for k in sorted(merged.keys()):
        body_lines.append(f"{k}={merged[k]}")
    return "\n".join(body_lines) + "\n"


def load_secrets(agent_id: str, *, root: Optional[Path] = None) -> dict[str, str]:
    """Read the agent's secrets dict. Returns ``{}`` when missing.

    Read-side fallback:
      1. ``secrets.enc.json`` if present → decrypt via the current
         crypto backend → parse dotenv.
      2. ``secrets.env`` plaintext → parse dotenv.
      3. neither → ``{}``.

    The fallback handles the case where an OSS install migrates to
    KMS (encrypted file exists alongside legacy plaintext) — and
    vice versa — without breaking. The encrypted file always wins.
    """
    enc = _enc_path(agent_id, root=root)
    if enc.is_file():
        try:
            from runtime.crypto import select_crypto
            blob = enc.read_bytes()
            plaintext = select_crypto().decrypt(blob)
            return _parse_dotenv(plaintext.decode("utf-8"))
        except Exception as e:
            logger.warning("failed to decrypt %s: %s", enc, e)
            # Fall through to plaintext attempt — but DON'T silently
            # return {} on a decrypt error, that'd hide config rot.
            raise

    plain = _plain_path(agent_id, root=root)
    if not plain.is_file():
        return {}
    try:
        return _parse_dotenv(plain.read_text(encoding="utf-8"))
    except OSError as e:
        logger.warning("failed to read %s: %s", plain, e)
        return {}


def save_secrets(
    agent_id: str,
    secrets: dict[str, str],
    *,
    root: Optional[Path] = None,
) -> Path:
    """Persist secrets via the current crypto backend.

    ``secrets`` is merged into whatever's on disk; passing an empty
    string as a value removes the key. Returns the path actually
    written — ``secrets.env`` under Noop, ``secrets.enc.json`` under
    KMS.

    When migrating from Noop to KMS, the resulting encrypted file
    takes precedence; we deliberately do NOT delete the legacy
    plaintext here. ``tools/migrate_secrets_to_kms.py`` is the
    explicit migration path that removes the old file once the
    operator confirms.
    """
    from runtime.crypto import NoopCrypto, select_crypto

    existing = load_secrets(agent_id, root=root)
    merged: dict[str, str] = {**existing}
    for k, v in secrets.items():
        if v == "":
            merged.pop(k, None)
        else:
            merged[k] = v

    body = _serialize_dotenv(merged)
    crypto = select_crypto()
    agent_dir = _agent_dir(agent_id, root=root)
    agent_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(crypto, NoopCrypto):
        path = _plain_path(agent_id, root=root)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(body, encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass  # Windows / weird filesystems — best-effort
        tmp.replace(path)
        return path

    # KMS-backed path. Encrypt the same dotenv body and write the
    # envelope JSON file.
    path = _enc_path(agent_id, root=root)
    tmp = path.with_suffix(".tmp")
    ciphertext = crypto.encrypt(body.encode("utf-8"))
    tmp.write_bytes(ciphertext)
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def get_secret(
    provider: str,
    agent_id: Optional[str] = None,
    *,
    root: Optional[Path] = None,
) -> Optional[str]:
    """Look up the key for ``provider``. Resolution order:
       1. ``agents/<agent_id>/secrets.env`` (when agent_id given)
       2. Tenant BYOK (multi-tenant only) — ``tenants/<active>/byok/<provider>.json``
       3. ``os.environ`` (OSS local only — BYOK strict mode skips this)
       4. ``None``

    Always returns the FIRST canonical env var listed for the provider
    when writing back; reads tolerate aliases.

    BYOK strict (multi-tenant): step 3 is intentionally skipped so a
    platform-level env var can never accidentally bill the operator
    account for tenant traffic. Generate stage fails closed → "[offline]".
    """
    candidates = PROVIDERS.get(provider.lower())
    if not candidates:
        return None

    if agent_id:
        per_agent = load_secrets(agent_id, root=root)
        for name in candidates:
            if name in per_agent and per_agent[name]:
                return per_agent[name]

    # Tenant-level BYOK — only active when multi-tenant mode is on.
    try:
        from runtime.tenants.feature import is_multi_tenant_enabled
    except Exception:
        is_multi_tenant_enabled = lambda: False  # noqa: E731
    if is_multi_tenant_enabled():
        try:
            from runtime.tenant_context import get_active
            from runtime.tenants.byok import get_provider_key
            tid = get_active()
            if tid:
                v = get_provider_key(tid, provider.lower())
                if v:
                    return v
        except Exception as e:  # noqa: BLE001
            logger.warning("byok lookup failed for %s: %s", provider, e)
        # BYOK strict: no platform-env fallback in multi-tenant mode.
        return None

    for name in candidates:
        v = os.environ.get(name)
        if v:
            return v
    return None


def status(agent_id: Optional[str] = None, *, root: Optional[Path] = None) -> dict[str, dict]:
    """Per-provider status: source (per-agent | global | unset) + mask.

    Used by ``GET /agents/<id>/secrets`` so the UI can show which keys
    are set, where they came from, and a teaser of the value without
    leaking it.
    """
    out: dict[str, dict] = {}
    per_agent = load_secrets(agent_id, root=root) if agent_id else {}
    for provider, names in PROVIDERS.items():
        agent_val: Optional[str] = None
        for n in names:
            if n in per_agent and per_agent[n]:
                agent_val = per_agent[n]
                break
        global_val: Optional[str] = None
        for n in names:
            v = os.environ.get(n)
            if v:
                global_val = v
                break

        if agent_val:
            out[provider] = {
                "set": True,
                "source": "per-agent",
                "mask": mask_key(agent_val),
                "var": names[0],
            }
        elif global_val:
            out[provider] = {
                "set": True,
                "source": "global",
                "mask": mask_key(global_val),
                "var": names[0],
            }
        else:
            out[provider] = {
                "set": False,
                "source": "unset",
                "mask": None,
                "var": names[0],
            }
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mask_key(key: str) -> str:
    """Reveal first ~7 chars + last 4 chars; everything else as ellipsis."""
    if not key:
        return ""
    if len(key) <= 14:
        return key[:2] + "…" + key[-2:]
    return f"{key[:7]}…{key[-4:]}"


def provider_for_model(model_id: str) -> Optional[str]:
    """Detect provider by model id prefix. Conservative — only recognizes
    the model families we ship. Unknown → returns None so the caller can
    fall back to a default or raise."""
    m = (model_id or "").strip().lower()
    if not m:
        return None
    if m.startswith("claude") or m.startswith("anthropic"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("openai"):
        return "openai"
    return None
