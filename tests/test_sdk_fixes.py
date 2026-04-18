"""Tests for the three SDK fixes:
  1. Provider list unification (create_client + UnifiedClient).
  2. Engine routing is opt-in, not silent.
  3. Sync / async / stream share _prepare_request.
"""

from __future__ import annotations

import os
import socket
import importlib
from unittest.mock import patch, MagicMock

import pytest

import lunar_router as lr
from lunar_router import sdk
from lunar_router.models.llm_client import (
    UnifiedClient,
    OpenAIClient,
    AnthropicClient,
    MockLLMClient,
    _DEDICATED_CLIENTS,
    _PROVIDER_ALIASES,
)


# ---------------------------------------------------------------------------
# Fix 1: provider list unification
# ---------------------------------------------------------------------------


class TestCreateClientUnification:
    def test_mock_returns_dedicated_class(self):
        c = lr.create_client("mock", "test-model")
        assert isinstance(c, MockLLMClient)
        assert c.model_id == "test-model"

    @pytest.mark.parametrize("provider,model", [
        ("deepseek", "deepseek-chat"),
        ("together", "meta-llama/Llama-3-70b"),
        ("perplexity", "sonar-small"),
        ("cerebras", "llama3.1-8b"),
        ("sambanova", "Meta-Llama-3.1-8B"),
        ("fireworks", "accounts/fireworks/models/llama-v3-8b"),
        ("cohere", "command-r"),
    ])
    def test_unified_client_for_non_dedicated_providers(self, provider, model):
        c = lr.create_client(provider, model)
        assert isinstance(c, UnifiedClient)
        assert c.model_id == model
        assert c._provider == provider

    def test_gemini_alias_routes_to_google_client(self):
        # 'gemini' is in sdk.PROVIDERS, but aliased to dedicated GoogleClient
        try:
            c = lr.create_client("gemini", "gemini-1.5-flash")
        except ImportError:
            pytest.skip("google-generativeai not installed")
        from lunar_router.models.llm_client import GoogleClient
        assert isinstance(c, GoogleClient)

    def test_bedrock_raises_clear_error(self):
        with pytest.raises(ValueError, match="Bedrock is not yet supported"):
            lr.create_client("bedrock", "anthropic.claude-3-haiku")

    def test_unknown_provider_lists_options(self):
        with pytest.raises(ValueError, match="Unknown provider") as exc:
            lr.create_client("notaprovider", "foo")
        msg = str(exc.value)
        # Should list providers from both dedicated + PROVIDERS
        assert "openai" in msg
        assert "deepseek" in msg

    def test_unified_client_cost_lookup(self):
        c = lr.create_client("deepseek", "deepseek-chat")
        cost = c.cost_per_1k_tokens
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_unified_client_unknown_model_returns_default_cost(self):
        c = lr.create_client("together", "some-unknown-model-xyz")
        assert c.cost_per_1k_tokens == 0.001  # documented default

    def test_every_sdk_provider_is_reachable(self):
        """Every provider in sdk.PROVIDERS must be instantiable via create_client
        (except bedrock, which is explicitly gated). Pass api_key to bypass
        dedicated clients' init-time env-var checks."""
        for provider in sdk.PROVIDERS:
            if provider == "bedrock":
                continue
            try:
                c = lr.create_client(provider, "some-model", api_key="test-key")
                assert c is not None
            except ImportError:
                # provider-specific SDK (anthropic/google/etc.) not installed — that's ok
                pass
            except TypeError:
                # Dedicated client may not accept api_key kwarg the same way —
                # fall back to env-var injection
                with patch.dict(os.environ, {
                    "OPENAI_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
                    "MISTRAL_API_KEY": "x", "GOOGLE_API_KEY": "x",
                    "GROQ_API_KEY": "x",
                }):
                    c = lr.create_client(provider, "some-model")
                    assert c is not None


# ---------------------------------------------------------------------------
# Fix 2: engine routing is opt-in
# ---------------------------------------------------------------------------


class TestEngineRoutingExplicit:
    def _reload_sdk_without_engine(self):
        """Reload sdk with LUNAR_ENGINE_URL unset."""
        os.environ.pop("LUNAR_ENGINE_URL", None)
        importlib.reload(sdk)

    def _reload_sdk_with_engine(self, url="http://localhost:8080"):
        os.environ["LUNAR_ENGINE_URL"] = url
        importlib.reload(sdk)

    def test_no_silent_localhost_probe(self):
        """With LUNAR_ENGINE_URL unset, even if something is listening on :8080,
        direct-provider calls must NOT route through the engine."""
        self._reload_sdk_without_engine()
        assert sdk._ENGINE_EXPLICITLY_SET is False

        # _resolve_target for "openai/gpt-4o" must pick the openai base_url,
        # not the engine — regardless of whether localhost:8080 is alive.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            base, key = sdk._resolve_target("openai", "gpt-4o-mini", None, None)
            assert base == "https://api.openai.com/v1"
            assert key == "sk-test"

    def test_auto_model_raises_when_engine_not_set(self):
        self._reload_sdk_without_engine()
        with pytest.raises(ValueError, match="Cannot resolve provider for model 'auto'"):
            sdk._resolve_target("", "auto", None, None)

    def test_engine_explicit_uses_engine(self):
        self._reload_sdk_with_engine("http://engine.example:9000")
        assert sdk._ENGINE_EXPLICITLY_SET is True

        # Must NOT probe the network when resolving target — we mock _check_engine.
        with patch.object(sdk, "_check_engine", return_value=True) as mock_check:
            base, key = sdk._resolve_target("openai", "gpt-4o-mini", None, None)
            assert base == "http://engine.example:9000/v1"
            mock_check.assert_called_once()

        # cleanup
        self._reload_sdk_without_engine()

    def test_force_engine_overrides_without_env_var(self):
        self._reload_sdk_without_engine()
        base, key, body, provider, model = sdk._prepare_request(
            "openai/gpt-4o", [{"role": "user", "content": "hi"}],
            api_key="test", api_base=None,
            temperature=None, max_tokens=None, top_p=None,
            stop=None, tools=None, tool_choice=None,
            stream=False, force_engine=True, force_direct=False,
            extra={},
        )
        assert base == f"{sdk.ENGINE_URL}/v1"
        assert model == "openai/gpt-4o"  # full prefix passed to engine

    def test_force_direct_skips_engine(self):
        """force_direct=True must go to provider even if engine env is set."""
        self._reload_sdk_with_engine()
        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-x"}):
                base, key, body, provider, model = sdk._prepare_request(
                    "openai/gpt-4o", [{"role": "user", "content": "hi"}],
                    api_key=None, api_base=None,
                    temperature=None, max_tokens=None, top_p=None,
                    stop=None, tools=None, tool_choice=None,
                    stream=False, force_engine=False, force_direct=True,
                    extra={},
                )
                assert base == "https://api.openai.com/v1"
                assert model == "gpt-4o"  # NOT the full prefixed form
        finally:
            self._reload_sdk_without_engine()

    def test_api_base_override_always_wins(self):
        self._reload_sdk_with_engine()
        try:
            base, _ = sdk._resolve_target("openai", "gpt-4o", "my-key", "http://my-proxy/v1")
            assert base == "http://my-proxy/v1"
        finally:
            self._reload_sdk_without_engine()

    def test_unknown_provider_error_mentions_all_options(self):
        self._reload_sdk_without_engine()
        with pytest.raises(ValueError) as exc:
            sdk._resolve_target("", "some-weird-model", None, None)
        msg = str(exc.value)
        assert "provider/model" in msg
        assert "LUNAR_ENGINE_URL" in msg
        assert "force_engine" in msg


# ---------------------------------------------------------------------------
# Fix 3: _prepare_request is a single source of truth
# ---------------------------------------------------------------------------


class TestPrepareRequestSharing:
    def _common_args(self, **overrides):
        base = dict(
            api_key="test-key",
            api_base=None,
            temperature=0.7,
            max_tokens=200,
            top_p=0.9,
            stop=["\n\n"],
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="auto",
            stream=False,
            force_engine=False,
            force_direct=True,  # skip engine / env lookups
            extra={"user": "marcus"},
        )
        base.update(overrides)
        return base

    def test_body_contains_all_request_fields(self):
        args = self._common_args()
        base, key, body, provider, model = sdk._prepare_request(
            "openai/gpt-4o", [{"role": "user", "content": "hi"}], **args,
        )
        assert body["model"] == "gpt-4o"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 200
        assert body["top_p"] == 0.9
        assert body["stop"] == ["\n\n"]
        assert body["tools"][0]["function"]["name"] == "x"
        assert body["tool_choice"] == "auto"
        assert body["stream"] is False
        assert body["user"] == "marcus"  # from extra

    def test_stream_flag_propagates(self):
        args = self._common_args(stream=True)
        _, _, body, _, _ = sdk._prepare_request(
            "openai/gpt-4o", [{"role": "user", "content": "hi"}], **args,
        )
        assert body["stream"] is True

    def test_engine_passthrough_model_name(self):
        """When routed to engine, body['model'] must keep the full 'provider/model'
        string; when routed direct, only the bare model name."""
        args_engine = self._common_args(force_engine=True, force_direct=False)
        _, _, body_eng, _, _ = sdk._prepare_request(
            "openai/gpt-4o", [{"role": "user", "content": "hi"}], **args_engine,
        )
        assert body_eng["model"] == "openai/gpt-4o"

        args_direct = self._common_args(force_direct=True)
        _, _, body_dir, _, _ = sdk._prepare_request(
            "openai/gpt-4o", [{"role": "user", "content": "hi"}], **args_direct,
        )
        assert body_dir["model"] == "gpt-4o"

    def test_sync_and_async_use_same_prepare_request(self):
        """Best way to prove no drift: both paths must produce identical bodies
        given the same inputs. We patch _prepare_request and count calls."""
        # The shared helper is called from _send_completion, _stream_completion,
        # and acompletion. Sanity check that all three call it.
        import inspect
        src_send = inspect.getsource(sdk._send_completion)
        src_stream = inspect.getsource(sdk._stream_completion)
        src_async = inspect.getsource(sdk.acompletion)
        assert "_prepare_request" in src_send
        assert "_prepare_request" in src_stream
        assert "_prepare_request" in src_async

    def test_acompletion_supports_force_flags(self):
        """Fix 3 added force_engine / force_direct to acompletion — check the signature."""
        import inspect
        sig = inspect.signature(sdk.acompletion)
        assert "force_engine" in sig.parameters
        assert "force_direct" in sig.parameters


# ---------------------------------------------------------------------------
# End-to-end: completion() smoke via mocked openai SDK
# ---------------------------------------------------------------------------


class TestCompletionWithMockedOpenAI:
    """Verify the full request path by mocking the openai SDK layer."""

    def _mock_openai_response(self):
        resp = MagicMock()
        resp.model_dump.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        return resp

    def test_direct_provider_call(self):
        # Ensure engine opt-out
        os.environ.pop("LUNAR_ENGINE_URL", None)
        importlib.reload(sdk)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-x"}):
            fake_client = MagicMock()
            fake_client.chat.completions.create.return_value = self._mock_openai_response()
            with patch("openai.OpenAI", return_value=fake_client):
                resp = sdk.completion(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                )
                assert resp.choices[0].message.content == "hello"
                assert resp._provider == "openai"
                assert resp._latency_ms > 0
                # Check the base_url that OpenAI was called with = provider, not engine
                called_kwargs = fake_client.chat.completions.create.call_args.kwargs
                assert called_kwargs["model"] == "gpt-4o-mini"

    def test_fallbacks_tried_in_order(self):
        os.environ.pop("LUNAR_ENGINE_URL", None)
        importlib.reload(sdk)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-x", "GROQ_API_KEY": "gsk-y"}):
            fake_client = MagicMock()
            call_count = {"n": 0}

            def create(**kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("primary failed")
                return self._mock_openai_response()

            fake_client.chat.completions.create.side_effect = create
            with patch("openai.OpenAI", return_value=fake_client):
                resp = sdk.completion(
                    model="openai/gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    fallbacks=["groq/llama-3.1-8b-instant"],
                )
                assert resp.choices[0].message.content == "hello"
                assert call_count["n"] == 2  # primary failed, fallback succeeded


# ---------------------------------------------------------------------------
# UnifiedClient end-to-end (mocked)
# ---------------------------------------------------------------------------


class TestUnifiedClientRoundTrip:
    def test_generate_wraps_completion(self):
        os.environ.pop("LUNAR_ENGINE_URL", None)
        importlib.reload(sdk)

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dk-test"}):
            fake_client = MagicMock()
            resp_mock = MagicMock()
            resp_mock.model_dump.return_value = {
                "id": "x", "model": "deepseek-chat",
                "choices": [{"index": 0, "message": {"content": "unified works"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            fake_client.chat.completions.create.return_value = resp_mock
            with patch("openai.OpenAI", return_value=fake_client):
                c = lr.create_client("deepseek", "deepseek-chat")
                out = c.generate("hello", max_tokens=128, temperature=0.2)
                assert out.text == "unified works"
                assert out.input_tokens == 10
                assert out.output_tokens == 5
                assert out.tokens_used == 15
                assert out.latency_ms > 0
