"""Lunar Router SDK — OpenAI-compatible client across 13 providers.

This module is the plumbing behind ``lunar_router.completion`` and
``lunar_router.Router``. It parses ``provider/model`` strings, resolves a
target (direct-to-provider HTTP, Lunar engine, or explicit ``api_base``),
builds an OpenAI-schema request body, and wraps the response with Lunar
metadata (``_cost``, ``_latency_ms``, ``_provider``, ``_routing``).

The SDK is the entry point for the broader Lunar loop: every request can be
captured as a trace, traces become datasets, datasets become distilled
student models, and those models get swapped in under your app via routing
aliases — without your code changing. See ``lunar_router`` package docs.

Quick start:
    >>> import lunar_router as lr
    >>> resp = lr.completion(
    ...     model="openai/gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "Hello"}],
    ... )
    >>> print(resp.choices[0].message.content, f"${resp._cost:.6f}")

Streaming:
    >>> for chunk in lr.completion(model="openai/gpt-4o-mini",
    ...                            messages=[...], stream=True):
    ...     print(chunk.choices[0].delta.content or "", end="")

Router with fallbacks:
    >>> router = lr.Router(
    ...     model_list=[
    ...         {"model_name": "smart", "model": "openai/gpt-4o"},
    ...         {"model_name": "smart", "model": "anthropic/claude-sonnet-4-6"},
    ...     ],
    ...     fallbacks=[{"smart": ["deepseek/deepseek-chat"]}],
    ... )
    >>> resp = router.completion(model="smart", messages=[...])

Engine mode (drop-in OpenAI SDK, zero code changes):
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
    # All providers routed through the Lunar engine; each request is a trace.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator, Optional, Union

from .model_prices import MODEL_INFO, model_cost

# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com", "api_key_env": "ANTHROPIC_API_KEY", "format": "anthropic"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "api_key_env": "MISTRAL_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"},
    "perplexity": {"base_url": "https://api.perplexity.ai", "api_key_env": "PERPLEXITY_API_KEY"},
    "cerebras": {"base_url": "https://api.cerebras.ai/v1", "api_key_env": "CEREBRAS_API_KEY"},
    "sambanova": {"base_url": "https://api.sambanova.ai/v1", "api_key_env": "SAMBANOVA_API_KEY"},
    "together": {"base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "fireworks": {"base_url": "https://api.fireworks.ai/inference/v1", "api_key_env": "FIREWORKS_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key_env": "GEMINI_API_KEY"},
    "cohere": {"base_url": "https://api.cohere.com/compatibility/v1", "api_key_env": "COHERE_API_KEY"},
    "bedrock": {"base_url": "", "api_key_env": "AWS_ACCESS_KEY_ID", "format": "bedrock"},
}

# Prefix → provider (for auto-detection when no "provider/" prefix given)
_MODEL_PREFIX_MAP: dict[str, str] = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude": "anthropic",
    "llama": "groq",
    "mixtral": "groq",
    "gemma": "groq",
    "mistral": "mistral",
    "ministral": "mistral",
    "codestral": "mistral",
    "pixtral": "mistral",
    "deepseek": "deepseek",
    "gemini": "gemini",
    "command": "cohere",
    "c4ai": "cohere",
    "sonar": "perplexity",
}

# Engine URL (Go engine). Engine routing is opt-in: traffic goes direct to the
# provider unless LUNAR_ENGINE_URL is explicitly set or force_engine=True.
ENGINE_URL: str = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
_ENGINE_EXPLICITLY_SET: bool = "LUNAR_ENGINE_URL" in os.environ

# ---------------------------------------------------------------------------
# Response types (OpenAI-compatible with attribute access)
# ---------------------------------------------------------------------------


class _AttrDict(dict):
    """Dict with recursive attribute access for OpenAI-compatible responses."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, _AttrDict):
                return _AttrDict(val)
            if isinstance(val, list):
                return [_AttrDict(v) if isinstance(v, dict) else v for v in val]
            return val
        except KeyError:
            return None


class ModelResponse(_AttrDict):
    """OpenAI-compatible chat completion response with Lunar metadata.

    Standard fields: id, object, model, choices, usage
    Lunar extras:  _provider, _cost, _latency_ms, _routing
    """
    pass


class StreamChunk(_AttrDict):
    """Single SSE chunk from a streaming response."""
    pass


# ---------------------------------------------------------------------------
# Model parsing
# ---------------------------------------------------------------------------


def parse_model(model: str) -> tuple[str, str]:
    """Parse 'provider/model_name' into (provider, model_name).

    If no provider prefix, auto-detects from model name.
    Returns ("", model) if provider cannot be determined.
    """
    if "/" in model:
        provider, _, model_name = model.partition("/")
        return provider, model_name

    for prefix, prov in _MODEL_PREFIX_MAP.items():
        if model.startswith(prefix):
            return prov, model
    return "", model


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------

_engine_available: Optional[bool] = None


def _check_engine() -> bool:
    """Probe the engine (cached). On first successful connect, triggers key reload.

    Only called when engine routing has been explicitly opted into (via
    LUNAR_ENGINE_URL or force_engine=True). We no longer silently probe
    localhost — that made behavior depend on whether an unrelated server
    happened to be running on :8080.
    """
    global _engine_available
    if _engine_available is not None:
        return _engine_available
    try:
        req = urllib.request.Request(f"{ENGINE_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=1) as resp:
            _engine_available = resp.status == 200
        if _engine_available:
            # Tell engine to reload keys from ~/.lunar/secrets.json
            try:
                reload_req = urllib.request.Request(
                    f"{ENGINE_URL}/v1/config/reload", method="POST",
                    headers={"Content-Type": "application/json"},
                    data=b"{}",
                )
                urllib.request.urlopen(reload_req, timeout=2)
            except Exception:
                pass
    except Exception:
        _engine_available = False
    return _engine_available


def reset_engine_cache() -> None:
    """Reset the cached engine availability check."""
    global _engine_available
    _engine_available = None


# ---------------------------------------------------------------------------
# Core: completion()
# ---------------------------------------------------------------------------


def completion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stream: bool = False,
    stop: Optional[Union[str, list[str]]] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    timeout: float = 120.0,
    num_retries: int = 0,
    fallbacks: Optional[list[str]] = None,
    force_engine: bool = False,
    force_direct: bool = False,
    **kwargs: Any,
) -> Union[ModelResponse, Iterator[StreamChunk]]:
    """Call any LLM with a unified interface.

    Args:
        model: Model identifier. Supports "provider/model" format
               (e.g., "openai/gpt-4o-mini") or bare model names
               (e.g., "gpt-4o-mini") with auto-detection.
               Use "auto" for semantic routing via the Go engine.
        messages: OpenAI-format messages list.
        api_key: Override API key (default: from env).
        api_base: Override base URL.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        top_p: Nucleus sampling parameter.
        stream: If True, returns an iterator of StreamChunks.
        stop: Stop sequence(s).
        tools: Function/tool definitions.
        tool_choice: Tool choice strategy.
        timeout: Request timeout in seconds.
        num_retries: Number of retries on transient errors.
        fallbacks: List of fallback models (tried in order on failure).
        force_engine: Always route through Go engine.
        force_direct: Always call provider APIs directly.
        **kwargs: Extra provider-specific parameters.

    Returns:
        ModelResponse (non-streaming) or Iterator[StreamChunk] (streaming).
    """
    models_to_try = [model] + (fallbacks or [])

    last_error: Optional[Exception] = None
    for m in models_to_try:
        for attempt in range(1 + num_retries):
            try:
                if stream:
                    return _stream_completion(
                        m, messages,
                        api_key=api_key, api_base=api_base,
                        temperature=temperature, max_tokens=max_tokens,
                        top_p=top_p, stop=stop, tools=tools,
                        tool_choice=tool_choice, timeout=timeout,
                        force_engine=force_engine, force_direct=force_direct,
                        **kwargs,
                    )
                else:
                    return _send_completion(
                        m, messages,
                        api_key=api_key, api_base=api_base,
                        temperature=temperature, max_tokens=max_tokens,
                        top_p=top_p, stop=stop, tools=tools,
                        tool_choice=tool_choice, timeout=timeout,
                        force_engine=force_engine, force_direct=force_direct,
                        **kwargs,
                    )
            except Exception as e:
                last_error = e
                if attempt < num_retries:
                    time.sleep(0.5 * (attempt + 1))

    raise last_error or RuntimeError("completion failed with no models to try")


async def acompletion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stream: bool = False,
    stop: Optional[Union[str, list[str]]] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    timeout: float = 120.0,
    num_retries: int = 0,
    fallbacks: Optional[list[str]] = None,
    force_engine: bool = False,
    force_direct: bool = False,
    **kwargs: Any,
) -> Union[ModelResponse, Any]:
    """Async version of completion(). Requires openai SDK installed.

    Same parameters as completion(). Uses openai.AsyncOpenAI internally and
    shares request preparation (target resolution, body build) with the sync
    path via _prepare_request — so force_engine / force_direct / engine
    pass-through behave identically.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai package required for async. Install: pip install openai")

    models_to_try = [model] + (fallbacks or [])
    last_error: Optional[Exception] = None

    for m in models_to_try:
        for attempt in range(1 + num_retries):
            try:
                base_url, key, body, provider_name, effective_model = _prepare_request(
                    m, messages,
                    api_key=api_key, api_base=api_base,
                    temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                    stop=stop, tools=tools, tool_choice=tool_choice,
                    stream=stream, force_engine=force_engine, force_direct=force_direct,
                    extra=kwargs,
                )

                client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout)
                start = time.time()
                resp = await client.chat.completions.create(**body)
                latency_ms = (time.time() - start) * 1000

                if stream:
                    return resp  # AsyncStream

                return _wrap_response(resp, provider_name, effective_model, latency_ms)

            except Exception as e:
                last_error = e
                if attempt < num_retries:
                    import asyncio
                    await asyncio.sleep(0.5 * (attempt + 1))

    raise last_error or RuntimeError("acompletion failed")


# ---------------------------------------------------------------------------
# Internal: send / stream
# ---------------------------------------------------------------------------


def _resolve_target(
    provider_name: str,
    model_name: str,
    api_key: Optional[str],
    api_base: Optional[str],
) -> tuple[str, str]:
    """Determine base_url and api_key for the request.

    Resolution order:
      1. Explicit ``api_base`` override — always wins.
      2. Engine routing — only if ``LUNAR_ENGINE_URL`` is explicitly set.
         (Previously we silently probed localhost:8080 and used it if up,
         which made behavior non-deterministic across machines.)
      3. Direct provider call — requires a known provider and API key.
    """
    if api_base:
        return api_base, api_key or "none"

    if _ENGINE_EXPLICITLY_SET:
        _check_engine()  # side effect: reload keys on first use
        return f"{ENGINE_URL}/v1", api_key or os.environ.get("LUNAR_API_KEY", "none")

    if not provider_name:
        raise ValueError(
            f"Cannot resolve provider for model '{model_name}'. "
            "Use 'provider/model' format (e.g. 'openai/gpt-4o'), "
            "set LUNAR_ENGINE_URL to route via the Lunar engine, "
            "or pass force_engine=True."
        )
    cfg = PROVIDERS.get(provider_name, {})
    base = cfg.get("base_url", "")
    if not base:
        raise ValueError(f"Unknown provider: {provider_name}")
    key = api_key or os.environ.get(cfg.get("api_key_env", ""), "")
    if not key:
        raise ValueError(
            f"No API key for {provider_name}. "
            f"Set {cfg.get('api_key_env', '?')} or pass api_key="
        )
    return base, key


def _build_body(
    model_name: str,
    messages: list[dict],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    stop: Optional[Union[str, list[str]]],
    tools: Optional[list[dict]],
    tool_choice: Optional[Union[str, dict]],
    stream: bool,
    extra: dict,
) -> dict[str, Any]:
    """Build the OpenAI-compatible request body."""
    body: dict[str, Any] = {"model": model_name, "messages": messages, "stream": stream}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if top_p is not None:
        body["top_p"] = top_p
    if stop is not None:
        body["stop"] = stop
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    body.update(extra)
    return body


def _prepare_request(
    model: str,
    messages: list[dict],
    *,
    api_key: Optional[str],
    api_base: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    stop: Optional[Union[str, list[str]]],
    tools: Optional[list[dict]],
    tool_choice: Optional[Union[str, dict]],
    stream: bool,
    force_engine: bool,
    force_direct: bool,
    extra: dict,
) -> tuple[str, str, dict[str, Any], str, str]:
    """Resolve target, pick effective model name, and build request body.

    Single source of truth shared by sync, stream, and async paths so the
    force_engine / force_direct / engine-pass-through semantics can't drift.

    Returns:
        (base_url, api_key, body, provider_name, effective_model_name)
    """
    provider_name, model_name = parse_model(model)

    if force_engine:
        base_url = f"{ENGINE_URL}/v1"
        key = api_key or os.environ.get("LUNAR_API_KEY", "none")
        effective_model = model  # pass full "provider/model" to engine
    elif force_direct:
        cfg = PROVIDERS.get(provider_name, {})
        base_url = api_base or cfg.get("base_url", "")
        if not base_url:
            raise ValueError(f"Unknown provider: {provider_name}")
        key = api_key or os.environ.get(cfg.get("api_key_env", ""), "")
        effective_model = model_name
    else:
        base_url, key = _resolve_target(provider_name, model_name, api_key, api_base)
        # When routing through the engine, it expects the full "provider/model" string
        effective_model = model if base_url.startswith(ENGINE_URL) else model_name

    body = _build_body(
        effective_model, messages, temperature, max_tokens,
        top_p, stop, tools, tool_choice, stream, extra,
    )
    return base_url, key, body, provider_name, effective_model


def _send_completion(
    model: str,
    messages: list[dict],
    *,
    api_key: Optional[str],
    api_base: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    stop: Optional[Union[str, list[str]]],
    tools: Optional[list[dict]],
    tool_choice: Optional[Union[str, dict]],
    timeout: float,
    force_engine: bool,
    force_direct: bool,
    **kwargs: Any,
) -> ModelResponse:
    base_url, key, body, provider_name, effective_model = _prepare_request(
        model, messages,
        api_key=api_key, api_base=api_base,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p,
        stop=stop, tools=tools, tool_choice=tool_choice,
        stream=False, force_engine=force_engine, force_direct=force_direct,
        extra=kwargs,
    )

    # Try openai SDK first (better DX, handles retries, types)
    try:
        return _send_via_openai_sdk(
            base_url, key, body, timeout, provider_name, effective_model,
        )
    except ImportError:
        pass

    # Fallback: raw HTTP
    return _send_via_http(
        base_url, key, body, timeout, provider_name, effective_model,
    )


def _send_via_openai_sdk(
    base_url: str, api_key: str, body: dict[str, Any],
    timeout: float, provider_name: str, model_name: str,
) -> ModelResponse:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    start = time.time()
    resp = client.chat.completions.create(**body)
    latency_ms = (time.time() - start) * 1000

    return _wrap_response(resp, provider_name, model_name, latency_ms)


def _send_via_http(
    base_url: str, api_key: str, body: dict[str, Any],
    timeout: float, provider_name: str, model_name: str,
) -> ModelResponse:
    """Fallback: call provider via raw HTTP (no openai SDK needed)."""
    data = json.dumps(body).encode()
    url = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_data = json.loads(resp.read())
    latency_ms = (time.time() - start) * 1000

    return _build_model_response(resp_data, provider_name, model_name, latency_ms)


def _stream_completion(
    model: str,
    messages: list[dict],
    *,
    api_key: Optional[str],
    api_base: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    stop: Optional[Union[str, list[str]]],
    tools: Optional[list[dict]],
    tool_choice: Optional[Union[str, dict]],
    timeout: float,
    force_engine: bool,
    force_direct: bool,
    **kwargs: Any,
) -> Iterator[StreamChunk]:
    """Stream completion via openai SDK or raw SSE."""
    base_url, key, body, _provider, _model = _prepare_request(
        model, messages,
        api_key=api_key, api_base=api_base,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p,
        stop=stop, tools=tools, tool_choice=tool_choice,
        stream=True, force_engine=force_engine, force_direct=force_direct,
        extra=kwargs,
    )

    # Prefer openai SDK for streaming (handles SSE parsing)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        stream = client.chat.completions.create(**body)
        for chunk in stream:
            yield StreamChunk(chunk.model_dump())
        return
    except ImportError:
        pass

    # Fallback: raw SSE
    yield from _stream_via_http(base_url, key, body, timeout)


def _stream_via_http(
    base_url: str, api_key: str, body: dict[str, Any], timeout: float,
) -> Generator[StreamChunk, None, None]:
    """Raw SSE streaming fallback."""
    data = json.dumps(body).encode()
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=timeout)

    for line in resp:
        line = line.decode("utf-8").strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                yield StreamChunk(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass


# ---------------------------------------------------------------------------
# Response wrapping
# ---------------------------------------------------------------------------


def _wrap_response(resp: Any, provider_name: str, model_name: str, latency_ms: float) -> ModelResponse:
    """Wrap an openai SDK response into ModelResponse."""
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    return _build_model_response(data, provider_name, model_name, latency_ms)


def _build_model_response(
    data: dict, provider_name: str, model_name: str, latency_ms: float,
) -> ModelResponse:
    """Build a ModelResponse from a raw dict, adding Lunar metadata."""
    # Calculate cost
    cost = 0.0
    usage = data.get("usage") or {}
    if usage:
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost = model_cost(model_name, input_tokens, output_tokens)

    # Check if engine returned cost
    engine_cost = data.pop("cost", None)
    if isinstance(engine_cost, dict):
        cost = engine_cost.get("total_cost_usd", cost)

    resp = ModelResponse(data)
    resp["_provider"] = provider_name
    resp["_cost"] = cost
    resp["_latency_ms"] = latency_ms
    resp["_routing"] = {
        k.replace("X-Lunar-", "").lower().replace("-", "_"): v
        for k, v in data.items()
        if isinstance(k, str) and k.startswith("X-Lunar-")
    }
    return resp


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------


@dataclass
class _Deployment:
    model_name: str  # alias
    model: str       # full "provider/model" string
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    weight: float = 1.0
    # Runtime stats
    requests: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0


class Router:
    """Multi-model router with load balancing and fallbacks.

    Similar to LiteLLM Router - maps logical model names to multiple
    provider deployments and handles failover.

    Example:
        router = Router(
            model_list=[
                {"model_name": "gpt-4", "model": "openai/gpt-4o"},
                {"model_name": "gpt-4", "model": "anthropic/claude-3-5-sonnet-20241022"},
                {"model_name": "fast",  "model": "groq/llama-3.3-70b-versatile"},
            ],
            fallbacks=[{"gpt-4": ["deepseek/deepseek-chat"]}],
            strategy="round-robin",
        )
        response = router.completion(model="gpt-4", messages=[...])
    """

    def __init__(
        self,
        model_list: list[dict[str, Any]],
        fallbacks: Optional[list[dict[str, list[str]]]] = None,
        strategy: str = "round-robin",
        num_retries: int = 2,
        timeout: float = 120.0,
    ):
        """
        Args:
            model_list: List of deployment configs. Each dict has:
                - model_name: Logical name/alias (e.g., "gpt-4")
                - model: Provider/model string (e.g., "openai/gpt-4o")
                - api_key: Optional override
                - api_base: Optional override
                - weight: Load balancing weight (default 1.0)
            fallbacks: List of {model_name: [fallback_models]} mappings.
            strategy: "round-robin", "least-cost", "lowest-latency", "weighted-random"
            num_retries: Retries per deployment before moving to next.
            timeout: Request timeout in seconds.
        """
        self._deployments: dict[str, list[_Deployment]] = {}
        self._fallbacks: dict[str, list[str]] = {}
        self._strategy = strategy
        self._num_retries = num_retries
        self._timeout = timeout
        self._rr_counters: dict[str, int] = {}

        for cfg in model_list:
            name = cfg["model_name"]
            dep = _Deployment(
                model_name=name,
                model=cfg["model"],
                api_key=cfg.get("api_key"),
                api_base=cfg.get("api_base"),
                weight=cfg.get("weight", 1.0),
            )
            self._deployments.setdefault(name, []).append(dep)

        if fallbacks:
            for fb_map in fallbacks:
                for name, fb_models in fb_map.items():
                    self._fallbacks[name] = fb_models

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Route a completion through the configured deployments.

        Args:
            model: Logical model name (must match a model_name in model_list).
            messages: Chat messages.
            **kwargs: Passed through to completion().
        """
        deployments = self._deployments.get(model, [])
        if not deployments:
            # Try as a direct model call
            return completion(model=model, messages=messages, timeout=self._timeout, **kwargs)

        ordered = self._order_deployments(model, deployments)
        last_error: Optional[Exception] = None

        for dep in ordered:
            for attempt in range(1 + self._num_retries):
                try:
                    start = time.time()
                    resp = completion(
                        model=dep.model,
                        messages=messages,
                        api_key=dep.api_key,
                        api_base=dep.api_base,
                        timeout=self._timeout,
                        **kwargs,
                    )
                    dep.requests += 1
                    dep.total_latency_ms += (time.time() - start) * 1000
                    return resp
                except Exception as e:
                    dep.requests += 1
                    dep.errors += 1
                    last_error = e
                    if attempt < self._num_retries:
                        time.sleep(0.3 * (attempt + 1))

        # Try fallbacks
        fb_models = self._fallbacks.get(model, [])
        for fb in fb_models:
            try:
                return completion(model=fb, messages=messages, timeout=self._timeout, **kwargs)
            except Exception as e:
                last_error = e

        raise last_error or RuntimeError(f"All deployments failed for model '{model}'")

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async version of completion()."""
        deployments = self._deployments.get(model, [])
        if not deployments:
            return await acompletion(model=model, messages=messages, timeout=self._timeout, **kwargs)

        import asyncio

        ordered = self._order_deployments(model, deployments)
        last_error: Optional[Exception] = None

        for dep in ordered:
            for attempt in range(1 + self._num_retries):
                try:
                    start = time.time()
                    resp = await acompletion(
                        model=dep.model,
                        messages=messages,
                        api_key=dep.api_key,
                        api_base=dep.api_base,
                        timeout=self._timeout,
                        **kwargs,
                    )
                    dep.requests += 1
                    dep.total_latency_ms += (time.time() - start) * 1000
                    return resp
                except Exception as e:
                    dep.requests += 1
                    dep.errors += 1
                    last_error = e
                    if attempt < self._num_retries:
                        await asyncio.sleep(0.3 * (attempt + 1))

        fb_models = self._fallbacks.get(model, [])
        for fb in fb_models:
            try:
                return await acompletion(model=fb, messages=messages, timeout=self._timeout, **kwargs)
            except Exception as e:
                last_error = e

        raise last_error or RuntimeError(f"All deployments failed for model '{model}'")

    def get_stats(self) -> dict[str, Any]:
        """Return per-deployment statistics."""
        stats = {}
        for name, deps in self._deployments.items():
            stats[name] = [
                {
                    "model": d.model,
                    "requests": d.requests,
                    "errors": d.errors,
                    "error_rate": d.errors / d.requests if d.requests > 0 else 0,
                    "avg_latency_ms": d.total_latency_ms / d.requests if d.requests > 0 else 0,
                }
                for d in deps
            ]
        return stats

    def _order_deployments(self, model: str, deployments: list[_Deployment]) -> list[_Deployment]:
        """Order deployments according to strategy."""
        if len(deployments) <= 1:
            return deployments

        if self._strategy == "round-robin":
            idx = self._rr_counters.get(model, 0)
            self._rr_counters[model] = (idx + 1) % len(deployments)
            return deployments[idx:] + deployments[:idx]

        if self._strategy == "least-cost":
            return sorted(deployments, key=lambda d: _deployment_cost(d.model))

        if self._strategy == "lowest-latency":
            def avg_lat(d: _Deployment) -> float:
                return d.total_latency_ms / d.requests if d.requests > 0 else float("inf")
            return sorted(deployments, key=avg_lat)

        if self._strategy == "weighted-random":
            import random
            weights = [d.weight for d in deployments]
            result = []
            remaining = list(deployments)
            remaining_weights = list(weights)
            while remaining:
                [choice] = random.choices(remaining, weights=remaining_weights, k=1)
                idx = remaining.index(choice)
                result.append(remaining.pop(idx))
                remaining_weights.pop(idx)
            return result

        return deployments


def _deployment_cost(model: str) -> float:
    """Get approximate cost for sorting."""
    _, model_name = parse_model(model)
    info = MODEL_INFO.get(model_name)
    if info:
        return info[0] + info[1]  # sum of input + output per-token costs
    return float("inf")


# ---------------------------------------------------------------------------
# Trace ingestion
# ---------------------------------------------------------------------------


def add_trace(
    messages: Optional[list[dict[str, Any]]] = None,
    *,
    input: Optional[str] = None,
    output: Optional[str] = None,
    model: str = "",
    provider: str = "",
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    cost_usd: Optional[float] = None,
    latency_ms: Optional[float] = None,
    is_error: bool = False,
    source: str = "manual",
    tags: Optional[list[str]] = None,
    engine_url: Optional[str] = None,
) -> dict:
    """Add a single trace to ClickHouse.

    Accepts either messages (OpenAI format) or simple input/output strings.
    Missing metadata (tokens, cost) is auto-estimated.

    Args:
        messages: OpenAI-format messages list.
        input: Simple input text (alternative to messages).
        output: Simple output text (alternative to messages).
        model: Model name (e.g., "gpt-4o-mini").
        provider: Provider name (e.g., "openai").
        tokens_in: Input token count (auto-estimated if None).
        tokens_out: Output token count (auto-estimated if None).
        cost_usd: Total cost in USD (auto-calculated from model pricing if None).
        latency_ms: Latency in milliseconds.
        is_error: Whether this trace represents an error.
        source: Source tag (default "manual").
        tags: Optional list of tags.
        engine_url: Override engine URL.

    Returns:
        dict with ingestion result.

    Example:
        # From messages
        lr.add_trace(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            model="gpt-4o-mini",
        )

        # From simple input/output
        lr.add_trace(input="Hello!", output="Hi there!", model="gpt-4o-mini")
    """
    return add_traces([{
        "messages": messages,
        "input": input or "",
        "output": output or "",
        "model": model,
        "provider": provider,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "is_error": is_error,
        "source": source,
        "tags": tags,
    }], engine_url=engine_url)


def add_traces(
    traces: list[dict[str, Any]],
    *,
    engine_url: Optional[str] = None,
) -> dict:
    """Add multiple traces to ClickHouse in one call.

    Each trace dict can have: messages, input, output, model, provider,
    tokens_in, tokens_out, cost_usd, latency_ms, is_error, source, tags.

    Args:
        traces: List of trace dicts.
        engine_url: Override engine URL.

    Returns:
        dict with ingestion result {"message": "...", "ingested": N}.

    Example:
        lr.add_traces([
            {"input": "Hello!", "output": "Hi!", "model": "gpt-4o-mini"},
            {"input": "Bye!", "output": "Goodbye!", "model": "gpt-4o-mini"},
        ])
    """
    url = (engine_url or ENGINE_URL) + "/v1/traces"

    # Clean None values from traces
    clean = []
    for t in traces:
        clean.append({k: v for k, v in t.items() if v is not None})

    payload = json.dumps({"traces": clean}).encode()
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"Trace ingestion failed ({e.code}): {body}") from e


def import_traces(
    path: str,
    *,
    source: str = "file-import",
    model: str = "",
    provider: str = "",
    engine_url: Optional[str] = None,
) -> dict:
    """Import traces from a JSONL or JSON file.

    Supported formats:
        - JSONL: one JSON object per line (messages or input/output)
        - JSON array: [{"messages": [...]}, ...]
        - OpenAI fine-tuning format: {"messages": [...]} per line

    Args:
        path: Path to the file.
        source: Source tag for all imported traces.
        model: Default model name (used if trace doesn't specify one).
        provider: Default provider name.
        engine_url: Override engine URL.

    Returns:
        dict with ingestion result.

    Example:
        lr.import_traces("training_data.jsonl", model="gpt-4o-mini")
    """
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = p.read_text(encoding="utf-8")
    traces: list[dict] = []

    # Try JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            traces = data
        elif isinstance(data, dict) and "traces" in data:
            traces = data["traces"]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL
    if not traces:
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not traces:
        raise ValueError(f"No traces found in {path}")

    # Apply defaults
    for t in traces:
        if source and "source" not in t:
            t["source"] = source
        if model and "model" not in t:
            t["model"] = model
        if provider and "provider" not in t:
            t["provider"] = provider

    return add_traces(traces, engine_url=engine_url)
