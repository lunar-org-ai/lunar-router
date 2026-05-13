"""Per-agent BYOK secrets (P3.1).

Each agent stores provider keys in ``agents/<id>/secrets.env`` — a
plain KEY=VALUE file we treat as the source of truth at runtime. The
file is gitignored and mode-0600 on create so leaks via repo or
permission slop are less likely.

Two paths to a key:

  1) Per-agent (preferred): ``agents/<id>/secrets.env`` has
     ``ANTHROPIC_API_KEY=...`` or ``OPENAI_API_KEY=...``. Each agent
     can use its operator's own key without polluting the global env.

  2) Global fallback (legacy): values in ``os.environ`` (loaded from
     the repo's ``.env`` during server startup). Lets the dev install
     keep working with a single key for everything.

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


def _secrets_path(agent_id: str, *, root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else Path("agents")
    return base / agent_id / "secrets.env"


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def load_secrets(agent_id: str, *, root: Optional[Path] = None) -> dict[str, str]:
    """Parse ``agents/<id>/secrets.env``. Returns ``{}`` when missing.

    Lenient KEY=VALUE parser: skips blank lines + ``#`` comments,
    strips matching quote pairs, ignores malformed lines instead of
    raising. Keep behavior identical to ``runtime/dotenv.py`` so the
    operator's intuition transfers.
    """
    path = _secrets_path(agent_id, root=root)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("failed to read %s: %s", path, e)
        return {}
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


def save_secrets(
    agent_id: str,
    secrets: dict[str, str],
    *,
    root: Optional[Path] = None,
) -> Path:
    """Persist secrets, creating the file with mode 0600 if missing.

    ``secrets`` is merged into whatever's on disk: pass an empty string
    as the value to remove a key, otherwise the existing value is
    preserved if the new dict omits it.
    """
    path = _secrets_path(agent_id, root=root)
    existing = load_secrets(agent_id, root=root) if path.is_file() else {}
    merged: dict[str, str] = {**existing}
    for k, v in secrets.items():
        if v == "":
            merged.pop(k, None)
        else:
            merged[k] = v

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    body_lines = [
        "# Agent BYOK secrets — gitignored, never commit.",
        "# Provider keys: ANTHROPIC_API_KEY, OPENAI_API_KEY.",
        "",
    ]
    for k in sorted(merged.keys()):
        body_lines.append(f"{k}={merged[k]}")
    tmp.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass  # Windows / weird filesystems — best-effort
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
       2. ``os.environ`` (global .env loaded at startup)
       3. ``None``

    Always returns the FIRST canonical env var listed for the provider
    when writing back; reads tolerate aliases.
    """
    candidates = PROVIDERS.get(provider.lower())
    if not candidates:
        return None

    if agent_id:
        per_agent = load_secrets(agent_id, root=root)
        for name in candidates:
            if name in per_agent and per_agent[name]:
                return per_agent[name]

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
