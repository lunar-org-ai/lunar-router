"""Tests for the Hub download manager."""

import json
import pytest
from pathlib import Path

from lunar_router.hub import LUNAR_DATA_HOME, Hub, Package, PackageIndex
from lunar_router.hub.manager import _get_hub


class TestPackage:
    def test_from_dict(self):
        data = {
            "id": "test-pkg",
            "name": "Test Package",
            "version": "1.0.0",
            "description": "A test package",
            "category": "weights",
        }
        pkg = Package.from_dict(data)
        assert pkg.id == "test-pkg"
        assert pkg.name == "Test Package"
        assert pkg.version == "1.0.0"
        assert pkg.category == "weights"

    def test_from_dict_defaults(self):
        data = {"id": "minimal"}
        pkg = Package.from_dict(data)
        assert pkg.name == "minimal"
        assert pkg.version == "1.0.0"
        assert pkg.source_type == "huggingface"
        assert pkg.archive_type == "zip"

    def test_to_dict_roundtrip(self):
        data = {
            "id": "roundtrip",
            "name": "Roundtrip Test",
            "version": "2.0.0",
            "description": "Test roundtrip",
            "category": "weights",
        }
        pkg = Package.from_dict(data)
        result = pkg.to_dict()
        assert result["id"] == "roundtrip"
        assert result["version"] == "2.0.0"


class TestPackageIndex:
    def test_from_dict(self):
        data = {
            "version": "1.0.0",
            "updated": "2025-01-01",
            "packages": [
                {"id": "pkg-1", "name": "Package 1", "description": "First", "category": "weights"},
                {"id": "pkg-2", "name": "Package 2", "description": "Second", "category": "data"},
            ],
        }
        index = PackageIndex.from_dict(data)
        assert len(index.packages) == 2
        assert index.get("pkg-1") is not None
        assert index.get("pkg-1").name == "Package 1"

    def test_get_nonexistent(self):
        index = PackageIndex.from_dict({"packages": []})
        assert index.get("nope") is None

    def test_list_by_category(self):
        data = {
            "packages": [
                {"id": "w1", "category": "weights"},
                {"id": "d1", "category": "data"},
                {"id": "w2", "category": "weights"},
            ],
        }
        index = PackageIndex.from_dict(data)
        weights = index.list_by_category("weights")
        assert len(weights) == 2

    def test_load_bundled(self):
        """The bundled index.json should load successfully."""
        index = PackageIndex.load_bundled()
        assert len(index.packages) >= 2
        assert "weights-mmlu-v1" in index.packages


class TestHub:
    def test_path(self, tmp_path):
        hub = Hub(data_home=tmp_path)
        assert hub.path("test-pkg") == tmp_path / "test-pkg"

    def test_is_installed_false(self, tmp_path):
        hub = Hub(data_home=tmp_path)
        assert not hub.is_installed("nonexistent")

    def test_is_installed_true(self, tmp_path):
        hub = Hub(data_home=tmp_path)
        pkg_dir = tmp_path / "test-pkg"
        pkg_dir.mkdir()
        (pkg_dir / "manifest.json").write_text("{}")
        assert hub.is_installed("test-pkg")

    def test_remove_nonexistent(self, tmp_path):
        hub = Hub(data_home=tmp_path)
        assert hub.remove("nonexistent", quiet=True) is False

    def test_remove_installed(self, tmp_path):
        hub = Hub(data_home=tmp_path)
        pkg_dir = tmp_path / "test-pkg"
        pkg_dir.mkdir()
        (pkg_dir / "manifest.json").write_text("{}")
        assert hub.remove("test-pkg", quiet=True) is True
        assert not pkg_dir.exists()

    def test_verify_not_installed(self, tmp_path):
        hub = Hub(data_home=tmp_path)
        assert hub.verify("nonexistent") is False

    def test_verify_missing_manifest(self, tmp_path):
        hub = Hub(data_home=tmp_path)
        (tmp_path / "test-pkg").mkdir()
        assert hub.verify("test-pkg") is False

    def test_list_packages(self):
        hub = Hub()
        packages = hub.list_packages()
        assert len(packages) >= 2

    def test_info_nonexistent(self):
        hub = Hub()
        assert hub.info("totally-fake-package-xyz") is None

    def test_info_existing(self):
        hub = Hub()
        info = hub.info("weights-mmlu-v1")
        assert info is not None
        assert info["id"] == "weights-mmlu-v1"
        assert "installed" in info


class TestModuleLevelAPI:
    def test_global_hub_singleton(self):
        hub1 = _get_hub()
        hub2 = _get_hub()
        assert hub1 is hub2

    def test_lunar_data_home_is_path(self):
        assert isinstance(LUNAR_DATA_HOME, Path)
