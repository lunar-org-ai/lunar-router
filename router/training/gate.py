"""First-fit gate.

Cheap pure function the harness brain (P15.3.9) and proposer (P15.3.7)
both call before any expensive embedding / fitting work. Gate logic is
isolated here so it's easy to test and easy to tweak independently of
the trainer.
"""

from __future__ import annotations

from typing import Optional


DEFAULT_MIN_CORPUS_SIZE = 200


def check_first_fit_eligibility(
    corpus_size: int,
    min_corpus_size: int = DEFAULT_MIN_CORPUS_SIZE,
    requested_k: Optional[int] = None,
) -> tuple[bool, str]:
    """Decide whether the corpus is large enough to fit clusters.

    Returns ``(eligible, reason)``. When eligible is False, the reason
    explains why so the Claude Code brain (P15.3.9) can surface it in
    its own rationale ("not enough data — drift_score=0.04, n_traces=87").

    The recommendation we offer (in the docstring, not enforced) is
    K ≈ sqrt(N/2), capped at [4, 32]. The brain reads it as a hint and
    picks K via its own reasoning.

    Args:
        corpus_size: Number of usable prompts available.
        min_corpus_size: Floor below which we refuse to fit.
        requested_k: Optional K the caller wants to use. If supplied, we
                     also enforce N >= 2K (otherwise sklearn's silhouette
                     score is undefined).

    Returns:
        (eligible, reason) — reason is "ok" when eligible is True.
    """
    if corpus_size < min_corpus_size:
        return False, (
            f"corpus_size={corpus_size} < min_corpus_size={min_corpus_size}"
        )
    if requested_k is not None:
        if requested_k < 2:
            return False, f"K={requested_k} < 2"
        if corpus_size < 2 * requested_k:
            return False, (
                f"corpus_size={corpus_size} < 2 * K={requested_k} "
                "(silhouette undefined)"
            )
    return True, "ok"
