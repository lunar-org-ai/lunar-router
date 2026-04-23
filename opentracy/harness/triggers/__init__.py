"""Trigger engine — the spine that turns signals into agent runs.

Reads ledger signals + policy YAML, matches them, dispatches with budget
enforcement, and writes the dispatch itself back to the ledger so the
chain stays reconstructible.
"""

from .engine import TriggerEngine
from .policies import Policy, PolicyBudget, PolicyMatch, load_policies
from .runner import TriggerEngineLoop

__all__ = [
    "Policy",
    "PolicyBudget",
    "PolicyMatch",
    "TriggerEngine",
    "TriggerEngineLoop",
    "load_policies",
]
