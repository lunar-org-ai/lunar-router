"""Compiler — turns AgentConfig into an executable Pipeline.

Discovers techniques by importing `techniques.<name>.impl` and looking for a
class named `<CamelName>Technique` (e.g. techniques.rag.impl.RagTechnique,
techniques.prompt_strategies.impl.PromptStrategiesTechnique). Each stage is
compiled into a Stage object via BaseTechnique.compile(variant, knobs).
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Optional

from runtime.protocols import BaseTechnique, Stage
from runtime.types import AgentConfig, StageConfig

logger = logging.getLogger(__name__)


@dataclass
class Pipeline:
    """An ordered list of compiled stages plus the config they came from."""

    stages: list[Stage]
    config: AgentConfig
    memory: Optional[Stage] = None


def _camel(name: str) -> str:
    """rag → Rag; prompt_strategies → PromptStrategies."""
    return "".join(part.title() for part in name.split("_"))


def _load_technique(name: str) -> BaseTechnique:
    try:
        mod = importlib.import_module(f"techniques.{name}.impl")
    except ImportError as e:
        raise ValueError(f"unknown technique {name!r}: {e}") from e

    cls_name = f"{_camel(name)}Technique"
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ValueError(f"techniques.{name}.impl is missing class {cls_name}")
    instance = cls()
    if not isinstance(instance, BaseTechnique):
        raise TypeError(f"{cls_name} must inherit from BaseTechnique")
    return instance


def _compile_stage(stage_cfg: StageConfig) -> Stage:
    technique = _load_technique(stage_cfg.technique)
    return technique.compile(stage_cfg.variant, stage_cfg.knobs)


def compile_agent(cfg: AgentConfig, strict: bool = False) -> Pipeline:
    """Compile an AgentConfig into an executable Pipeline.

    Main pipeline stages always fail hard on unknown technique. Cross-cutting
    concerns (memory, etc.) compile best-effort: if the technique isn't yet
    implemented, log a warning and skip — unless strict=True.
    """
    stages: list[Stage] = [_compile_stage(s) for s in cfg.pipeline]

    memory: Optional[Stage] = None
    if cfg.cross_cutting.memory is not None:
        try:
            memory = _compile_stage(cfg.cross_cutting.memory)
        except (ValueError, TypeError) as e:
            if strict:
                raise
            logger.warning(
                "cross_cutting.memory not compiled (%s). Pipeline will run without memory.",
                e,
            )
    pipeline = Pipeline(stages=stages, config=cfg, memory=memory)

    # P15.3.10 follow-up: warm the embedder pool when uniroute is used so the
    # first /run doesn't pay the ~1-3s MiniLM load. Best-effort — if the
    # router extra isn't installed (no torch / sentence-transformers), the
    # warmup gracefully no-ops so agent compile still succeeds.
    if _agent_uses_uniroute(cfg):
        try:
            from runtime.embedder_pool import get_pool

            get_pool().warm()
            logger.info("uniroute variant detected; embedder pool warmed")
        except Exception as e:
            logger.warning(
                "embedder warmup skipped (router extras may be missing): %s", e
            )

    return pipeline


def _agent_uses_uniroute(cfg: AgentConfig) -> bool:
    """Return True if any pipeline stage is the uniroute routing variant."""
    for stage in cfg.pipeline:
        if stage.technique == "routing" and stage.variant == "uniroute":
            return True
    return False
