"""LLM-powered cluster labeling, coherence scoring, and outlier detection.

Thin wrapper around the harness AgentRunner. All prompts live in .md files
under lunar_router/harness/agents/ — edit those to change behavior.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..harness.runner import AgentRunner
from .models import ClusterLabel

logger = logging.getLogger(__name__)

DEFAULT_ENGINE_URL = "http://localhost:8080"


class ClusterLabeler:
    """Labels clusters using harness agents. Behavior defined by .md prompt files."""

    def __init__(self, engine_url: str = DEFAULT_ENGINE_URL, **kwargs):
        self.runner = AgentRunner(engine_url=engine_url)

    async def label_cluster(self, sample_prompts: list[str]) -> ClusterLabel:
        """Run cluster_labeler agent to generate a structured domain label."""
        if not sample_prompts:
            return ClusterLabel.unknown()

        numbered = "\n".join(f'{i+1}. "{p[:200]}"' for i, p in enumerate(sample_prompts[:10]))
        user_input = f"Sample prompts:\n{numbered}"

        try:
            result = await self.runner.run("cluster_labeler", user_input)
            if "parse_error" in result.data:
                logger.warning(f"Label parse error: {result.data['parse_error']}")
                return ClusterLabel.unknown()
            return ClusterLabel.from_dict(result.data)
        except Exception as e:
            logger.warning(f"LLM labeling failed: {type(e).__name__}: {e}", exc_info=True)
            return ClusterLabel.unknown()

    async def score_coherence(self, sample_prompts: list[str]) -> float:
        """Run coherence_scorer agent to rate cluster coherence 0-1."""
        if len(sample_prompts) < 2:
            return 0.0

        numbered = "\n".join(f'{i+1}. "{p[:150]}"' for i, p in enumerate(sample_prompts[:10]))
        user_input = f"Prompts:\n{numbered}"

        try:
            result = await self.runner.run("coherence_scorer", user_input)
            return float(result.get("coherence", 0.5))
        except Exception as e:
            logger.warning(f"Coherence scoring failed: {e}")
            return 0.5

    async def detect_outliers(self, sample_prompts: list[str]) -> list[int]:
        """Run outlier_detector agent to find prompts that don't belong."""
        if len(sample_prompts) < 3:
            return []

        numbered = "\n".join(f'{i+1}. "{p[:150]}"' for i, p in enumerate(sample_prompts[:15]))
        user_input = f"Prompts:\n{numbered}"

        try:
            result = await self.runner.run("outlier_detector", user_input)
            indices = result.get("outlier_indices", [])
            # Convert 1-based to 0-based
            return [i - 1 for i in indices if isinstance(i, int) and 1 <= i <= len(sample_prompts)]
        except Exception as e:
            logger.warning(f"Outlier detection failed: {e}")
            return []

    async def check_merge(self, prompts_a: list[str], prompts_b: list[str]) -> tuple[bool, str]:
        """Run merge_checker agent to decide if two clusters should merge."""
        group_a = "\n".join(f'  A{i+1}. "{p[:100]}"' for i, p in enumerate(prompts_a[:5]))
        group_b = "\n".join(f'  B{i+1}. "{p[:100]}"' for i, p in enumerate(prompts_b[:5]))
        user_input = f"Group A:\n{group_a}\n\nGroup B:\n{group_b}"

        try:
            result = await self.runner.run("merge_checker", user_input)
            return bool(result.get("same_domain", False)), result.get("reason", "")
        except Exception as e:
            logger.warning(f"Merge check failed: {e}")
            return False, f"LLM error: {e}"
