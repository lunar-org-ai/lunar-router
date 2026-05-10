"""One-shot offline replay — seed evals/_response_cache/cache.jsonl.

Without a populated cache, ``RouterEvaluator`` raises ``CacheGapError``
on every (prompt, model) it doesn't know about, and ``propose_router_retrain``
returns ``"blocked"`` with reason ``cache_missing``.

This script generates responses for each (golden, model) pair and stores
them. Re-running is safe — already-cached entries are skipped.

Usage:

    python -m tools.populate_response_cache \\
        --models claude-haiku-4-5,claude-sonnet-4-6 \\
        --metric exact_match

    python -m tools.populate_response_cache --models claude-haiku-4-5 --dry-run

Defaults:
  --goldens evals/golden/
  --out evals/_response_cache/cache.jsonl
  --metric exact_match
  --max-prompts 0   (0 = no cap)

Notes:
  - ``ANTHROPIC_API_KEY`` must be set for AnthropicClient to work.
  - OpenAI/Mistral clients are deferred; this script only hits Anthropic.
    Pass non-Anthropic model IDs at your own risk — the AnthropicClient's
    cost lookup falls back to a default but the API call may fail.
  - Cost estimate: each call ≈ a single short prompt + response. With 8
    goldens × 2 models, expect <$0.10 on Haiku/Sonnet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


logger = logging.getLogger("tools.populate_response_cache")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model IDs (e.g. claude-haiku-4-5,claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--goldens",
        default="evals/golden",
        help="Directory of <id>.yaml golden files (default: evals/golden).",
    )
    parser.add_argument(
        "--out",
        default="evals/_response_cache/cache.jsonl",
        help="Cache JSONL path (default: evals/_response_cache/cache.jsonl).",
    )
    parser.add_argument(
        "--metric",
        default="exact_match",
        choices=["exact_match", "contains", "normalized_exact", "f1", "mmlu"],
        help="Loss function (from router/core/metrics.py).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Cap on number of goldens to process (0 = no cap).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making API calls.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("error: --models requires at least one id", file=sys.stderr)
        return 2

    golden_dir = Path(args.goldens)
    if not golden_dir.exists():
        print(f"error: goldens dir not found: {golden_dir}", file=sys.stderr)
        return 2

    cache_path = Path(args.out)
    return _run(
        models=models,
        golden_dir=golden_dir,
        cache_path=cache_path,
        metric_name=args.metric,
        max_prompts=args.max_prompts,
        dry_run=args.dry_run,
    )


def _run(
    *,
    models: list[str],
    golden_dir: Path,
    cache_path: Path,
    metric_name: str,
    max_prompts: int,
    dry_run: bool,
) -> int:
    from evals.loader import load_golden
    from router.core.metrics import MetricType, get_metric
    from router.evaluation.cache import ResponseCache
    from router.models.llm_client import AnthropicClient, MockLLMClient

    metric_fn = get_metric(MetricType(metric_name))

    golden_paths = sorted(golden_dir.glob("*.yaml"))
    if max_prompts > 0:
        golden_paths = golden_paths[:max_prompts]
    if not golden_paths:
        print(f"warning: no golden YAML files in {golden_dir}", file=sys.stderr)
        return 1

    print(
        f"[populate] {len(golden_paths)} goldens × {len(models)} models = "
        f"{len(golden_paths) * len(models)} (prompt, model) pairs",
        file=sys.stderr,
    )

    if dry_run:
        for path in golden_paths:
            print(f"  golden: {path.stem}")
        for m in models:
            print(f"  model:  {m}")
        return 0

    cache = ResponseCache(path=cache_path)
    clients = {m: _client_for(m) for m in models}

    new_count = 0
    skipped = 0
    failed = 0

    for path in golden_paths:
        gid = path.stem
        try:
            g = load_golden(gid, golden_dir=golden_dir)
        except Exception as e:
            logger.warning("golden %s failed to load: %s", gid, e)
            failed += 1
            continue

        prompt = g.input.request
        ground_truth = g.expected.exact or ""

        for model_id, client in clients.items():
            if cache.has(prompt, model_id):
                skipped += 1
                continue

            try:
                response = client.generate(prompt, max_tokens=256, temperature=0.0)
            except NotImplementedError as e:
                logger.warning("model %s deferred: %s", model_id, e)
                failed += 1
                continue
            except Exception as e:
                logger.warning(
                    "generate failed (%s on %s): %s", model_id, gid, e
                )
                failed += 1
                continue

            loss = float(metric_fn(response.text, ground_truth)) if ground_truth else 0.5
            cache.add(
                prompt=prompt,
                model_id=model_id,
                response_text=response.text,
                loss=loss,
                latency_ms=response.latency_ms,
                tokens_used=response.tokens_used,
            )
            new_count += 1

    cache.save()
    print(
        f"[populate] done — added {new_count}, skipped (already cached) {skipped}, "
        f"failed {failed}. Cache now has {len(cache)} entries at {cache_path}",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 1


def _client_for(model_id: str):
    """Pick the right concrete client for a model_id.

    Today only Anthropic is fully wired (P15.3.1 stubbed OpenAI/Mistral).
    For test-only IDs starting with ``mock-`` we return a MockLLMClient.
    """
    from router.models.llm_client import AnthropicClient, MockLLMClient

    if model_id.startswith("mock-"):
        return MockLLMClient(model=model_id, default_response="mock")

    # Default: Anthropic. If the model_id isn't in AnthropicClient.COSTS,
    # AnthropicClient falls back to a default cost; the SDK call may still
    # succeed if the model name is valid for the API.
    return AnthropicClient(model=model_id)


if __name__ == "__main__":
    sys.exit(main())
