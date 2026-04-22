"""Tests for the weights module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from opentracy.weights import (
    _resolve_package_id,
    download_weights,
    get_weights_path,
    list_available_weights,
    WeightsConfig,
    WEIGHTS_REGISTRY,
)


class TestResolvePackageId:
    def test_alias_default(self):
        assert _resolve_package_id("default") == "weights-default"

    def test_alias_mmlu(self):
        assert _resolve_package_id("mmlu") == "weights-mmlu-v1"

    def test_alias_mmlu_v1(self):
        assert _resolve_package_id("mmlu-v1") == "weights-mmlu-v1"

    def test_already_prefixed(self):
        assert _resolve_package_id("weights-custom") == "weights-custom"

    def test_auto_prefix(self):
        assert _resolve_package_id("my-weights") == "weights-my-weights"


class TestWeightsConfig:
    def test_basic_creation(self):
        config = WeightsConfig(name="test")
        assert config.name == "test"
        assert config.num_clusters == 100
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_defaults(self):
        config = WeightsConfig(name="test")
        assert config.version == "1.0.0"
        assert config.models == []
        assert config.source_type == "huggingface"


class TestWeightsRegistry:
    def test_contains_mmlu_v1(self):
        assert "weights-mmlu-v1" in WEIGHTS_REGISTRY

    def test_contains_default(self):
        assert "weights-default" in WEIGHTS_REGISTRY

    def test_mmlu_v1_config(self):
        config = WEIGHTS_REGISTRY["weights-mmlu-v1"]
        assert config.num_clusters == 100
        assert "gpt-4o" in config.models
        assert config.source_type == "huggingface"


class TestListAvailableWeights:
    def test_returns_list(self):
        result = list_available_weights()
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_all_are_weights_config(self):
        for config in list_available_weights():
            assert isinstance(config, WeightsConfig)


class TestGetWeightsPath:
    def test_returns_path(self):
        result = get_weights_path("default")
        assert isinstance(result, Path)

    def test_resolves_alias(self):
        path = get_weights_path("mmlu-v1")
        assert "weights-mmlu-v1" in str(path)


class TestDownloadWeights:
    @patch("opentracy.weights._bundled_path", return_value=None)
    @patch("opentracy.weights.hub_download")
    def test_calls_hub_download(self, mock_download, _mock_bundled):
        mock_download.return_value = Path("/fake/path")
        result = download_weights("default", verbose=True)
        mock_download.assert_called_once_with("weights-default", force=False, quiet=False)
        assert result == Path("/fake/path")

    @patch("opentracy.weights._bundled_path", return_value=None)
    @patch("opentracy.weights.hub_download")
    def test_force_redownload(self, mock_download, _mock_bundled):
        mock_download.return_value = Path("/fake/path")
        download_weights("mmlu-v1", force=True)
        mock_download.assert_called_once_with("weights-mmlu-v1", force=True, quiet=False)

    def test_returns_bundled_path_when_available(self):
        path = download_weights("default", verbose=False)
        assert path.exists()
        assert (path / "clusters").exists()
