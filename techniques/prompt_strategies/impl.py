"""prompt_strategies — real LLM call routed to Anthropic OR OpenAI (P3.1).

The ``direct`` variant loads a prompt template, renders it with the
retrieved documents + the user request, and calls whichever provider
matches the model id picked by the route stage:

  - ``claude-*`` / ``anthropic-*``  → Anthropic SDK
  - ``gpt-*`` / ``o1-*`` / ``o3-*`` → OpenAI SDK

Token usage is stashed in ``ctx.state["llm_usage"]`` so the executor
can compute exact cost instead of the char-based estimate it falls
back to.

BYOK (P3.1): the key is resolved through ``runtime.agents.secrets``:
  1) ``agents/<active_id>/secrets.env`` (operator's per-agent key)
  2) ``os.environ`` (global ``.env`` from server startup)
  3) None → offline marker, response still non-empty, tests still pass

Offline path is intact for the cold-clone case: missing key / missing
SDK package → clearly-marked response, no crash. Same shape the rest
of the pipeline expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from runtime.protocols import BaseTechnique, Context, Document, Stage


logger = logging.getLogger("techniques.prompt_strategies")

_DEFAULT_MODEL = "claude-haiku-4-5"
_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer based on the provided context. "
    "If the context does not contain the answer, say so explicitly."
)


class _Direct:
    def __init__(self, knobs: dict[str, Any]) -> None:
        self.prompt_path: str = str(knobs.get("prompt", "../prompts/system.md"))
        self.max_tokens: int = int(knobs.get("max_tokens", 2048))
        self.temperature: float = float(knobs.get("temperature", 0.3))

    def execute(self, context: Context) -> Context:
        from runtime.agent_context import get_active
        from runtime.agents.secrets import get_secret, provider_for_model

        model = (
            context.routing.model
            if context.routing and context.routing.model
            else _DEFAULT_MODEL
        )
        system = _load_system_prompt(self.prompt_path)
        user_msg = _render_user_message(context.documents, context.request)

        provider = provider_for_model(model) or "anthropic"
        agent_id = get_active()
        api_key = get_secret(provider, agent_id=agent_id)

        if not api_key:
            context.response = (
                f"[offline] No {provider} API key set for agent {agent_id!r} — "
                f"would have called {model} with {len(context.documents)} doc(s). "
                f"Request: {context.request!r}"
            )
            return context

        try:
            if provider == "openai":
                text, usage = _call_openai(
                    model=model,
                    system=system,
                    user=user_msg,
                    api_key=api_key,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            else:
                text, usage = _call_anthropic(
                    model=model,
                    system=system,
                    user=user_msg,
                    api_key=api_key,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
        except _SdkUnavailable as e:
            logger.warning("%s SDK unavailable: %s", provider, e)
            context.response = (
                f"[offline] {provider} SDK not installed — would have called "
                f"{model}. Request: {context.request!r}"
            )
            return context
        except Exception as e:
            logger.warning("%s call failed (%s); returning error marker", provider, e)
            context.response = (
                f"[error] LLM call failed: {type(e).__name__}: {e}. "
                f"Request: {context.request!r}"
            )
            return context

        context.response = text or "(empty response)"
        if usage:
            context.state["llm_usage"] = {**usage, "model": model, "provider": provider}
        return context


class _SdkUnavailable(RuntimeError):
    """Raised when the provider's SDK isn't importable."""


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _call_anthropic(
    *,
    model: str,
    system: str,
    user: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, Optional[dict]]:
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise _SdkUnavailable("anthropic") from e

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = ""
    if resp.content:
        parts = [getattr(block, "text", "") for block in resp.content]
        text = "".join(p for p in parts if p)
    usage = getattr(resp, "usage", None)
    usage_dict: Optional[dict] = None
    if usage is not None:
        usage_dict = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }
    return text, usage_dict


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _call_openai(
    *,
    model: str,
    system: str,
    user: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, Optional[dict]]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise _SdkUnavailable("openai") from e

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = ""
    if resp.choices:
        text = (resp.choices[0].message.content or "")
    usage_obj = getattr(resp, "usage", None)
    usage_dict: Optional[dict] = None
    if usage_obj is not None:
        # OpenAI uses prompt_tokens / completion_tokens.
        usage_dict = {
            "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        }
    return text, usage_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_system_prompt(path: str) -> str:
    """Resolve the prompt path against a few likely roots; fall back to default."""
    candidates = [
        Path(path),
        Path("agent/pipeline") / path,
        Path("agent") / path,
        Path.cwd() / path,
    ]
    for p in candidates:
        try:
            resolved = p.resolve()
            if resolved.is_file():
                return resolved.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return _DEFAULT_SYSTEM


def _render_user_message(documents: list[Document], request: str) -> str:
    if not documents:
        return request
    blocks = []
    for i, doc in enumerate(documents, 1):
        blocks.append(f"[{i}] {doc.content}")
    context_str = "\n\n".join(blocks)
    return f"Context:\n{context_str}\n\nQuestion: {request}"


class PromptStrategiesTechnique(BaseTechnique):
    name = "prompt_strategies"
    variants = ("direct",)

    def compile(self, variant: str, knobs: dict[str, Any]) -> Stage:
        if variant not in self.variants:
            raise ValueError(
                f"prompt_strategies: unknown variant {variant!r}; expected one of {self.variants}"
            )
        return _Direct(knobs)
