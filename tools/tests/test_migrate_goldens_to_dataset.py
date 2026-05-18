"""Tests for tools/migrate_goldens_to_dataset.

We stub the embedder pool with a deterministic mock to keep these tests
fast (the real MiniLM download is gated behind OPENTRACY_RUN_SLOW elsewhere).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


class _MockEmbedder:
    """Deterministic 4-dim embedder for tests. Stable per prompt."""

    def embed(self, prompt: str) -> list[float]:
        # Toy: ASCII-sum modulo, normalized — gives different vectors for
        # different prompts but is reproducible.
        s = sum(ord(c) for c in prompt)
        return [
            (s % 7) / 10.0,
            (s % 11) / 10.0,
            (s % 13) / 10.0,
            (s % 17) / 10.0,
        ]


@pytest.fixture
def patched_embedder(monkeypatch):
    """Replace tools.migrate_goldens_to_dataset._get_embedder with a mock."""
    import tools.migrate_goldens_to_dataset as mig
    monkeypatch.setattr(mig, "_get_embedder", lambda: _MockEmbedder())


@pytest.fixture
def goldens_dir(tmp_path: Path) -> Path:
    """Build a tiny goldens dir with 3 yaml files."""
    d = tmp_path / "evals_golden"
    d.mkdir()
    (d / "golden_001.yaml").write_text(
        "id: golden_001\n"
        "input:\n"
        "  request: \"What is your refund policy?\"\n"
        "expected:\n"
        "  contains: [\"refund\"]\n"
        "  category: policy\n"
    )
    (d / "golden_002.yaml").write_text(
        "id: golden_002\n"
        "input:\n"
        "  request: \"Where is my order #12345?\"\n"
        "expected:\n"
        "  exact: \"track via /orders\"\n"
        "  category: order_status\n"
    )
    (d / "golden_003.yaml").write_text(
        "id: golden_003\n"
        "input:\n"
        "  request: \"How do I reset my password?\"\n"
        "expected:\n"
        "  contains: [\"reset\"]\n"
    )
    return d


@pytest.fixture
def datasets_dir(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "datasets"
    d.mkdir()
    monkeypatch.setattr("router.data.dataset_io.DEFAULT_DATASETS_DIR", d)
    return d


def _run_migration(args: list[str]) -> int:
    from tools.migrate_goldens_to_dataset import main
    return main(args)


def test_migration_produces_v1_with_samples(
    patched_embedder, goldens_dir: Path, datasets_dir: Path
):
    code = _run_migration([
        "--name", "goldens",
        "--goldens-dir", str(goldens_dir),
        "--datasets-dir", str(datasets_dir),
    ])
    assert code == 0
    v1 = datasets_dir / "goldens" / "v1.json"
    assert v1.exists()
    payload = json.loads(v1.read_text())
    assert payload["version"] == 1
    assert payload["name"] == "goldens"
    assert len(payload["samples"]) == 3
    # ground_truth resolved: golden_002 has 'exact' → uses it; 001/003 have 'contains' first token
    by_prompt = {s["prompt"]: s for s in payload["samples"]}
    assert by_prompt["What is your refund policy?"]["ground_truth"] == "refund"
    assert by_prompt["Where is my order #12345?"]["ground_truth"] == "track via /orders"
    assert by_prompt["How do I reset my password?"]["ground_truth"] == "reset"
    # embedding shape matches our mock
    assert all(len(s["embedding"]) == 4 for s in payload["samples"])
    # tag from category (when present)
    assert by_prompt["What is your refund policy?"]["tag"] == "policy"
    assert by_prompt["How do I reset my password?"]["tag"] is None


def test_migration_writes_v0_placeholder(
    patched_embedder, goldens_dir: Path, datasets_dir: Path
):
    _run_migration([
        "--name", "goldens",
        "--goldens-dir", str(goldens_dir),
        "--datasets-dir", str(datasets_dir),
    ])
    v0 = datasets_dir / "goldens" / "v0.json"
    assert v0.exists()
    placeholder = json.loads(v0.read_text())
    assert placeholder["version"] == 0
    assert placeholder["samples"] == []


def test_migration_is_byte_identical_on_rerun(
    patched_embedder, goldens_dir: Path, datasets_dir: Path
):
    _run_migration([
        "--name", "goldens",
        "--goldens-dir", str(goldens_dir),
        "--datasets-dir", str(datasets_dir),
    ])
    v1 = datasets_dir / "goldens" / "v1.json"
    first = v1.read_bytes()
    # Re-run with --force to clobber and verify identical bytes.
    _run_migration([
        "--name", "goldens",
        "--goldens-dir", str(goldens_dir),
        "--datasets-dir", str(datasets_dir),
        "--force",
    ])
    second = v1.read_bytes()
    assert first == second, "second migration should produce byte-identical output"


def test_migration_dry_run_writes_nothing(
    patched_embedder, goldens_dir: Path, datasets_dir: Path
):
    code = _run_migration([
        "--name", "goldens",
        "--goldens-dir", str(goldens_dir),
        "--datasets-dir", str(datasets_dir),
        "--dry-run",
    ])
    assert code == 0
    assert not (datasets_dir / "goldens").exists()


def test_migration_refuses_to_clobber_without_force(
    patched_embedder, goldens_dir: Path, datasets_dir: Path
):
    code1 = _run_migration([
        "--name", "goldens",
        "--goldens-dir", str(goldens_dir),
        "--datasets-dir", str(datasets_dir),
    ])
    assert code1 == 0
    code2 = _run_migration([
        "--name", "goldens",
        "--goldens-dir", str(goldens_dir),
        "--datasets-dir", str(datasets_dir),
    ])
    assert code2 == 2  # refused without --force


def test_migration_updates_registry(
    patched_embedder, goldens_dir: Path, datasets_dir: Path
):
    _run_migration([
        "--name", "goldens",
        "--goldens-dir", str(goldens_dir),
        "--datasets-dir", str(datasets_dir),
    ])
    reg = json.loads((datasets_dir / "_registry.json").read_text())
    entry = reg["datasets"]["goldens"]
    assert entry["current_version"] == 1
    assert entry["size"] == 3
    assert entry["owner"] == "human"
    assert entry["sourceType"] == "manual"
