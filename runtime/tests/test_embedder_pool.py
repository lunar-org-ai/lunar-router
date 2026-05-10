"""Tests for runtime/embedder_pool.py."""

from __future__ import annotations

import pytest

from runtime.embedder_pool import EmbedderPool, get_pool, reset_pool


def test_pool_singleton_across_calls():
    """get_pool() returns the same instance every time."""
    reset_pool()
    p1 = get_pool()
    p2 = get_pool()
    assert p1 is p2
    reset_pool()


def test_pool_caches_embedder_after_first_get(monkeypatch):
    """A second .get() call returns the same embedder (no second load)."""
    reset_pool()
    pool = EmbedderPool()

    # Use the lightweight MockEmbeddingProvider so we don't load MiniLM in CI.
    from router.core.embeddings import MockEmbeddingProvider, PromptEmbedder

    fake = PromptEmbedder(MockEmbeddingProvider(dimension=8), cache_enabled=False)
    pool._embedder = fake  # short-circuit the lazy init for tests
    a = pool.get()
    b = pool.get()
    assert a is fake
    assert a is b


def test_warm_invokes_get_and_a_no_op_embed():
    """warm() should populate the pool and run one embed without raising."""
    reset_pool()
    pool = EmbedderPool()

    from router.core.embeddings import MockEmbeddingProvider, PromptEmbedder

    fake = PromptEmbedder(MockEmbeddingProvider(dimension=8), cache_enabled=False)
    pool._embedder = fake
    pool.warm()  # should not raise
    # And the embedder is still cached.
    assert pool.get() is fake


def test_reset_pool_clears_singleton():
    pool = get_pool()
    assert pool is get_pool()
    reset_pool()
    new_pool = get_pool()
    assert new_pool is not pool
    reset_pool()
