"""Shared LLM completion transport for harness sub-systems.

Mirrors the transport-selection logic in
``harness/introspection/agent.py`` but exposes a **tool-free**
completion API instead of the introspection-aware tool-use loop. Used
by P15.3.5's LLMJudge and any future harness component that needs a
plain "prompt in → text out" call against the same brain Claude Code
provides locally.

Two transports, picked automatically:

- ``anthropic_api`` — when ``ANTHROPIC_API_KEY`` is set. Direct Anthropic
  SDK call, ``messages.create`` with no ``tools=``.
- ``claude_code_cli`` — when the ``claude`` binary is on PATH. Spawns
  ``claude --print`` headless. Subprocess inherits CWD so it picks up
  the project's ``.mcp.json``.

Override with the ``BRAIN_TRANSPORT`` env var if both are available.
Raises ``BrainNotAvailableError`` when neither is reachable.

Note that we do **not** refactor ``harness/introspection/agent.py`` to
use this helper — the introspection path keeps its private functions
intact so we don't risk regressing a working surface. Only new code
(P15.3.5+) uses this helper. If they meaningfully drift over time,
refactor introspection in a follow-up.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional


logger = logging.getLogger("harness.brain.transport")


DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_TOKENS = 1024
DEFAULT_API_MODEL = "claude-sonnet-4-5"


class BrainNotAvailableError(RuntimeError):
    """Raised when no transport is configured.

    Set ``ANTHROPIC_API_KEY`` or install the ``claude`` CLI to fix.
    """


def select_transport() -> str:
    """Pick a transport. Returns 'anthropic_api' | 'claude_code_cli' | 'none'.

    Resolution order:
      1. ``BRAIN_TRANSPORT`` env var if set to a known value.
      2. ``anthropic_api`` if ``ANTHROPIC_API_KEY`` is set.
      3. ``claude_code_cli`` if ``claude`` is on PATH.
      4. ``none``.
    """
    forced = os.getenv("BRAIN_TRANSPORT", "").strip()
    if forced in ("anthropic_api", "claude_code_cli"):
        return forced
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic_api"
    if shutil.which("claude"):
        return "claude_code_cli"
    return "none"


def complete(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    transport: Optional[str] = None,
) -> str:
    """Run a single tool-free completion against the harness brain.

    Args:
        prompt: User prompt.
        system_prompt: Optional system message; passed as Anthropic SDK
            ``system=`` or via the CLI ``--append-system-prompt`` flag.
        model: Optional model id (Anthropic API only). CLI ignores this
            and uses whatever ``claude`` is configured with.
        max_tokens: API ``max_tokens``. CLI doesn't expose this directly.
        temperature: API temperature. CLI doesn't expose this directly.
        timeout_s: CLI subprocess timeout. API has no equivalent here.
        transport: Optional explicit transport. ``None`` → auto-select.

    Returns:
        The assistant's text reply.

    Raises:
        BrainNotAvailableError: When no transport is reachable.
        RuntimeError: When the chosen transport returns an error.
    """
    chosen = transport or select_transport()
    if chosen == "none":
        raise BrainNotAvailableError(
            "no brain transport — set ANTHROPIC_API_KEY or install `claude` CLI"
        )
    if chosen == "anthropic_api":
        return _complete_via_api(
            prompt,
            system_prompt=system_prompt,
            model=model or DEFAULT_API_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return _complete_via_cli(
        prompt,
        system_prompt=system_prompt,
        timeout_s=timeout_s,
    )


def _complete_via_api(
    prompt: str,
    *,
    system_prompt: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise BrainNotAvailableError(
            "anthropic_api transport selected but ANTHROPIC_API_KEY is missing"
        )

    # Lazy import — only pay the cost when this transport is used.
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    try:
        resp = client.messages.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"anthropic API call failed: {type(e).__name__}: {e}") from e

    text_blocks = [
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
    ]
    return "\n\n".join(t for t in text_blocks if t).strip()


def _complete_via_cli(
    prompt: str,
    *,
    system_prompt: Optional[str],
    timeout_s: int,
) -> str:
    args: list[str] = [
        "claude",
        "--print",
        # Auto-accept any MCP tool calls. The brain shouldn't be calling tools
        # for a plain completion, but this matches the introspection setup so
        # behavior is consistent in case the model ever does.
        "--permission-mode",
        "bypassPermissions",
    ]
    if system_prompt:
        args += ["--append-system-prompt", system_prompt]
    args.append(prompt)

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"claude --print subprocess timed out after {timeout_s}s"
        ) from e
    except FileNotFoundError as e:
        raise BrainNotAvailableError(
            "`claude` CLI not found on PATH"
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"claude --print exited {proc.returncode}: {proc.stderr[:300]}"
        )

    return (proc.stdout or "").strip()
