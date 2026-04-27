"""Tests for the Azure OpenAI provider in opentracy.sdk.

Azure differs from the other OpenAI-compatible providers in three ways:
  1. The endpoint is per-resource (lives in AZURE_OPENAI_ENDPOINT), not a
     fixed URL in PROVIDERS.
  2. Auth + URL go through ``openai.AzureOpenAI``, not ``openai.OpenAI``.
  3. ``model=`` in the request body is the *deployment name*, not the
     underlying model.

These tests cover all three.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

from opentracy import sdk


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------


class TestAzureRegistered:
    def test_azure_in_providers(self):
        assert "azure" in sdk.PROVIDERS

    def test_azure_uses_azure_api_key_env(self):
        cfg = sdk.PROVIDERS["azure"]
        assert cfg["api_key_env"] == "AZURE_OPENAI_API_KEY"

    def test_parse_model_azure_prefix(self):
        provider, model = sdk.parse_model("azure/my-gpt5-deploy")
        assert provider == "azure"
        assert model == "my-gpt5-deploy"


# ---------------------------------------------------------------------------
# _resolve_target — endpoint comes from env, not PROVIDERS
# ---------------------------------------------------------------------------


class TestAzureResolveTarget:
    def test_reads_endpoint_from_env(self):
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_ENDPOINT": "https://my-res.openai.azure.com",
        }, clear=False):
            base, key = sdk._resolve_target("azure", "my-deploy", None, None)
            assert base == "https://my-res.openai.azure.com"
            assert key == "az-key"

    def test_explicit_api_base_overrides_env(self):
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_ENDPOINT": "https://wrong.openai.azure.com",
        }, clear=False):
            base, key = sdk._resolve_target(
                "azure", "my-deploy",
                api_key="override-key",
                api_base="https://override.openai.azure.com",
            )
            assert base == "https://override.openai.azure.com"
            assert key == "override-key"

    def test_missing_endpoint_raises(self):
        # Clear AZURE_OPENAI_ENDPOINT specifically; ENGINE_URL must not be set
        # or _resolve_target short-circuits to engine routing.
        env = {k: v for k, v in os.environ.items()
               if k not in ("AZURE_OPENAI_ENDPOINT", "LUNAR_ENGINE_URL")}
        env["AZURE_OPENAI_API_KEY"] = "az-key"
        with patch.dict(os.environ, env, clear=True):
            importlib.reload(sdk)  # re-read _ENGINE_EXPLICITLY_SET
            with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
                sdk._resolve_target("azure", "my-deploy", None, None)

    def test_missing_api_key_raises(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("AZURE_OPENAI_API_KEY", "LUNAR_ENGINE_URL")}
        env["AZURE_OPENAI_ENDPOINT"] = "https://my-res.openai.azure.com"
        with patch.dict(os.environ, env, clear=True):
            importlib.reload(sdk)
            with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
                sdk._resolve_target("azure", "my-deploy", None, None)


# ---------------------------------------------------------------------------
# completion() — uses AzureOpenAI client, passes deployment as model
# ---------------------------------------------------------------------------


def _mock_openai_response():
    resp = MagicMock()
    resp.model_dump.return_value = {
        "id": "chatcmpl-az",
        "object": "chat.completion",
        "model": "gpt-5",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    return resp


class TestAzureCompletion:
    def test_uses_azure_openai_client(self):
        os.environ.pop("LUNAR_ENGINE_URL", None)
        importlib.reload(sdk)

        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_ENDPOINT": "https://my-res.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-10-21",
        }, clear=False):
            fake_client = MagicMock()
            fake_client.chat.completions.create.return_value = _mock_openai_response()

            with patch("openai.AzureOpenAI", return_value=fake_client) as mock_az:
                resp = sdk.completion(
                    model="azure/my-gpt5-deploy",
                    messages=[{"role": "user", "content": "ping"}],
                )

                # AzureOpenAI was constructed with the right per-resource args
                kwargs = mock_az.call_args.kwargs
                assert kwargs["api_key"] == "az-key"
                assert kwargs["azure_endpoint"] == "https://my-res.openai.azure.com"
                assert kwargs["api_version"] == "2024-10-21"

                # body.model must be the *deployment name*, not "gpt-5"
                body = fake_client.chat.completions.create.call_args.kwargs
                assert body["model"] == "my-gpt5-deploy"

                assert resp.choices[0].message.content == "hi"
                assert resp._provider == "azure"

    def test_default_api_version_when_env_unset(self):
        os.environ.pop("LUNAR_ENGINE_URL", None)
        importlib.reload(sdk)

        env = {k: v for k, v in os.environ.items()
               if k not in ("AZURE_OPENAI_API_VERSION", "LUNAR_ENGINE_URL")}
        env["AZURE_OPENAI_API_KEY"] = "az-key"
        env["AZURE_OPENAI_ENDPOINT"] = "https://my-res.openai.azure.com"

        with patch.dict(os.environ, env, clear=True):
            fake_client = MagicMock()
            fake_client.chat.completions.create.return_value = _mock_openai_response()

            with patch("openai.AzureOpenAI", return_value=fake_client) as mock_az:
                sdk.completion(
                    model="azure/my-deploy",
                    messages=[{"role": "user", "content": "ping"}],
                )
                # Default in _send_via_azure_sdk: 2024-10-21
                assert mock_az.call_args.kwargs["api_version"] == "2024-10-21"

    def test_custom_api_version_from_env(self):
        os.environ.pop("LUNAR_ENGINE_URL", None)
        importlib.reload(sdk)

        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_ENDPOINT": "https://my-res.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
        }, clear=False):
            fake_client = MagicMock()
            fake_client.chat.completions.create.return_value = _mock_openai_response()

            with patch("openai.AzureOpenAI", return_value=fake_client) as mock_az:
                sdk.completion(
                    model="azure/my-deploy",
                    messages=[{"role": "user", "content": "ping"}],
                )
                assert mock_az.call_args.kwargs["api_version"] == "2025-01-01-preview"

    def test_passes_through_reasoning_kwargs(self):
        """GPT-5 reasoning params (max_completion_tokens, reasoning_effort)
        must reach the Azure client untouched — they're how the user controls
        reasoning model behavior."""
        os.environ.pop("LUNAR_ENGINE_URL", None)
        importlib.reload(sdk)

        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_ENDPOINT": "https://my-res.openai.azure.com",
        }, clear=False):
            fake_client = MagicMock()
            fake_client.chat.completions.create.return_value = _mock_openai_response()

            with patch("openai.AzureOpenAI", return_value=fake_client):
                sdk.completion(
                    model="azure/my-gpt5",
                    messages=[{"role": "user", "content": "ping"}],
                    max_completion_tokens=256,
                    reasoning_effort="minimal",
                )
                body = fake_client.chat.completions.create.call_args.kwargs
                assert body["max_completion_tokens"] == 256
                assert body["reasoning_effort"] == "minimal"
