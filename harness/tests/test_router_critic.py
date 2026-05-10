"""Tests for harness.critics.router_critic.RouterCritic."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.types import Mutation
from harness.critics.router_critic import RouterCritic
from harness.types import CriticContext, Proposal
from router.core.clustering import ClusterAssigner, ClusterResult
from router.core.embeddings import PromptEmbedder
from router.data.dataset import PromptDataset, PromptSample
from router.evaluation.cache import ResponseCache


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeAssigner(ClusterAssigner):
    """Assigns based on prompt prefix encoded in the embedding's dim 0."""

    @property
    def num_clusters(self) -> int:
        return 2

    def assign(self, embedding: np.ndarray) -> ClusterResult:
        cid = 1 if embedding[0] > 0.5 else 0
        probs = np.zeros(2)
        probs[cid] = 1.0
        return ClusterResult(cluster_id=cid, probabilities=probs)

    def save(self, path):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def load(cls, path):  # pragma: no cover
        raise NotImplementedError


class _ClusterEncodingProvider:
    model_name = "cluster-enc"
    _dim = 8

    @property
    def dimension(self):
        return self._dim

    def embed(self, text):
        v = np.zeros(self._dim)
        v[0] = 1.0 if text.startswith("hard:") else 0.0
        return v

    def embed_batch(self, texts):
        return np.asarray([self.embed(t) for t in texts])


def _embedder() -> PromptEmbedder:
    return PromptEmbedder(_ClusterEncodingProvider(), cache_enabled=False)


def _payload(*, psi_haiku, psi_sonnet, k: int = 2) -> dict:
    centroids = np.array([[0.0] * 8, [1.0] + [0.0] * 7])  # cluster 0 vs 1
    return {
        "version": 1,
        "k": k,
        "centroids": centroids.tolist(),
        "model_psi": {
            "haiku": {
                "psi_vector": list(psi_haiku),
                "cost_per_1k_tokens": 0.001,
                "cluster_sample_counts": [10, 10],
                "metadata": {},
            },
            "sonnet": {
                "psi_vector": list(psi_sonnet),
                "cost_per_1k_tokens": 0.003,
                "cluster_sample_counts": [10, 10],
                "metadata": {},
            },
        },
        "cost_weight": 0.0,
        "embedder_model": "test",
        "embedding_dim": 8,
        "metadata": {"silhouette": 0.7},
    }


def _full_coverage_cache(prompts):
    """easy: → both correct; hard: → only sonnet correct."""
    cache = ResponseCache()
    for p in prompts:
        cache.add(p, "sonnet", "ans-s", 0.0)
        cache.add(p, "haiku", "ans-h", 0.0 if p.startswith("easy:") else 1.0)
    return cache


def _ctx_for(payload: dict) -> CriticContext:
    proposal = Proposal(
        mutations=[
            Mutation(
                file=f"versions/router_config_v{payload['version']}.json",
                path="<inline_payload>",
                value=payload,
            )
        ],
        description="test",
        source="claude_code",
    )
    return CriticContext(proposal=proposal, candidate_result=None)


def _critic(*, payload: dict, prompts: list[str], **extra) -> RouterCritic:
    cache = _full_coverage_cache(prompts)
    ds = PromptDataset([PromptSample(prompt=p, ground_truth="") for p in prompts])
    params = {
        "embedder": _embedder(),
        "cache": cache,
        "dataset": ds,
        "centroids": payload["centroids"],
        "eval_lambda_steps": 3,
        **extra,
    }
    return RouterCritic(params=params)


# ---------------------------------------------------------------------------


def test_router_critic_passes_when_psi_is_informative():
    prompts = [f"easy:p{i}" for i in range(5)] + [f"hard:p{i}" for i in range(5)]
    payload = _payload(
        psi_haiku=[0.1, 0.9],
        psi_sonnet=[0.1, 0.1],
    )
    critic = _critic(payload=payload, prompts=prompts)
    verdict = critic.verdict(_ctx_for(payload))
    # Cold-start (no current config). Candidate should beat AUROC=0.5 baseline.
    assert verdict.approved is True
    assert "delta_auroc" in verdict.reason
    assert verdict.severity == "info"


def test_router_critic_fails_on_regression_via_strict_thresholds():
    """An empty-Psi candidate cold-start fails because AUROC ~= 0.5 ≤ 0.5
    + min_auroc_improvement=0.0; we tighten the gate to require >0 improvement."""
    prompts = [f"easy:p{i}" for i in range(5)] + [f"hard:p{i}" for i in range(5)]
    payload = _payload(
        psi_haiku=[0.0, 0.0],   # empty Psi
        psi_sonnet=[0.0, 0.0],
    )
    critic = _critic(
        payload=payload,
        prompts=prompts,
        min_auroc_improvement=0.1,  # require at least +0.1 AUROC over baseline 0.5
    )
    verdict = critic.verdict(_ctx_for(payload))
    assert verdict.approved is False
    assert verdict.severity == "block"


def test_router_critic_blocks_on_missing_payload():
    proposal = Proposal(mutations=[], description="bad", source="claude_code")
    ctx = CriticContext(proposal=proposal, candidate_result=None)
    critic = RouterCritic()
    verdict = critic.verdict(ctx)
    assert verdict.approved is False
    assert "no mutations" in verdict.reason


def test_router_critic_blocks_on_wrong_payload_type():
    proposal = Proposal(
        mutations=[Mutation(file="versions/router_config_v1.json", path="x", value="not-a-dict")],
        description="bad",
        source="claude_code",
    )
    ctx = CriticContext(proposal=proposal, candidate_result=None)
    critic = RouterCritic()
    verdict = critic.verdict(ctx)
    assert verdict.approved is False
    assert "dict payload" in verdict.reason or "model_psi" in verdict.reason


def test_router_critic_blocks_when_cache_or_dataset_missing():
    payload = _payload(psi_haiku=[0.1, 0.9], psi_sonnet=[0.1, 0.1])
    # No cache / dataset in params.
    critic = RouterCritic(params={"embedder": _embedder(), "centroids": payload["centroids"]})
    verdict = critic.verdict(_ctx_for(payload))
    assert verdict.approved is False
    assert "dataset" in verdict.reason and "cache" in verdict.reason
