"""Per-trace cost estimation.

Wraps the pricing table from `router/models/llm_client.py` to produce
(tokens_in, tokens_out, cost_usd) from char counts. Stub agents that
don't actually call an LLM still produce realistic-shape numbers
(small but non-zero) so the UI's "Avg cost / conv" tile has data to
render before P1.9 (real LLM) lands.

When P1.9 ships, the executor swaps the char-based estimate for the
real `usage.input_tokens` / `usage.output_tokens` from the Anthropic
SDK — same fields, no schema change.
"""

from __future__ import annotations

from typing import Optional

# Anthropic charges ~$/1M-input + $/1M-output; their public docs use a
# blended-rate that the existing router pricing table already encodes
# as $/1k tokens. We reuse it for cost-per-trace estimation.
from router.models.llm_client import AnthropicClient


# Anthropic tokenizer: rule-of-thumb ~4 chars per token for English.
# Code/IDs push it slightly higher; long stub strings push slightly
# lower. 4 is a safe estimate that under-counts by 5-10% for prose.
_CHARS_PER_TOKEN = 4

_DEFAULT_MODEL = "claude-haiku-4-5"


def estimate_tokens(text: Optional[str]) -> int:
    """Cheap token estimate: max(1, len(text) // CHARS_PER_TOKEN). Empty → 0."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_cost(
    prompt: Optional[str],
    response: Optional[str],
    model: Optional[str] = None,
) -> tuple[int, int, float]:
    """Estimate (tokens_in, tokens_out, cost_usd) for one trace.

    Args:
        prompt: the user's input request (None or '' → 0 tokens).
        response: the agent's output (None or '' → 0 tokens).
        model: routing_model from the trace's route stage. Falls back to
               ``claude-haiku-4-5`` when unknown. Models missing from
               ``AnthropicClient.COSTS`` get the cheapest known rate.

    Returns:
        ``(tokens_in, tokens_out, cost_usd)``. Cost is rounded to 6
        decimals so JSONL trace files stay readable.
    """
    tokens_in = estimate_tokens(prompt)
    tokens_out = estimate_tokens(response)
    total_tokens = tokens_in + tokens_out

    rate_per_1k = _resolve_rate(model)
    cost = round(total_tokens * rate_per_1k / 1000.0, 6)
    return tokens_in, tokens_out, cost


def _resolve_rate(model: Optional[str]) -> float:
    """Pick $/1k for `model`. Fall back to the default model's rate when
    unknown — never raise, because we want stubs to keep working."""
    if model and model in AnthropicClient.COSTS:
        return AnthropicClient.COSTS[model]
    return AnthropicClient.COSTS.get(_DEFAULT_MODEL, 0.001)
