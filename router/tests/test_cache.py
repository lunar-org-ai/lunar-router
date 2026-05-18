"""Tests for router/evaluation/cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from router.evaluation.cache import CachedResponse, ResponseCache


def test_cache_add_and_get_round_trip():
    cache = ResponseCache()
    cache.add(prompt="Q?", model_id="haiku", response_text="ans", loss=0.0)
    e = cache.get("Q?", "haiku")
    assert e is not None
    assert isinstance(e, CachedResponse)
    assert e.response_text == "ans"
    assert e.loss == 0.0


def test_cache_hash_prompt_deterministic():
    h1 = ResponseCache.hash_prompt("hello world")
    h2 = ResponseCache.hash_prompt("hello world")
    assert h1 == h2
    assert ResponseCache.hash_prompt("other") != h1


def test_cache_get_all_models():
    cache = ResponseCache()
    cache.add("Q?", "haiku", "ans-h", 0.0)
    cache.add("Q?", "sonnet", "ans-s", 0.5)
    cache.add("Q2", "haiku", "ans2", 1.0)
    models = cache.get_all_models("Q?")
    assert set(models.keys()) == {"haiku", "sonnet"}
    assert models["haiku"].loss == 0.0


def test_cache_coverage_per_model():
    cache = ResponseCache()
    cache.add("p1", "a", "x", 0.0)
    cache.add("p2", "a", "x", 0.0)
    cache.add("p1", "b", "x", 0.0)
    assert cache.coverage("a") == 2
    assert cache.coverage("b") == 1
    assert cache.coverage("c") == 0


def test_cache_save_load_round_trip(tmp_path: Path):
    cache = ResponseCache()
    cache.add("p1", "a", "ans-a", 0.0, latency_ms=12.5, tokens_used=80)
    cache.add("p1", "b", "ans-b", 1.0, metadata={"note": "x"})
    path = tmp_path / "cache.jsonl"
    cache.save(path)

    reloaded = ResponseCache(path=path)
    assert len(reloaded) == 2
    assert reloaded.coverage("a") == 1
    e = reloaded.get("p1", "a")
    assert e is not None
    assert e.latency_ms == 12.5
    assert e.tokens_used == 80


def test_cache_purge_model_drops_only_that_model():
    cache = ResponseCache()
    cache.add("p1", "a", "x", 0.0)
    cache.add("p1", "b", "x", 0.0)
    cache.add("p2", "a", "x", 0.0)
    dropped = cache.purge_model("a")
    assert dropped == 2
    assert cache.coverage("a") == 0
    assert cache.coverage("b") == 1


def test_cache_skips_malformed_lines_on_load(tmp_path: Path):
    path = tmp_path / "cache.jsonl"
    path.write_text(
        "{not valid json\n"
        + '{"prompt_hash":"h","model_id":"m","response_text":"r","loss":0.0}\n'
    )
    cache = ResponseCache(path=path)
    # Only the valid line was loaded.
    assert len(cache) == 1
    assert cache.coverage("m") == 1


def test_cache_save_without_path_raises():
    cache = ResponseCache()
    with pytest.raises(ValueError):
        cache.save()
