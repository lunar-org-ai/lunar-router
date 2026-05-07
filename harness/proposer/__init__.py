"""harness.proposer — strategies that generate Proposals.

v0: heuristic only (sweep a knob over a list of values). Claude Code-driven
proposer lands later: spawns a session, reads program.md + recent traces,
returns a structured Proposal.
"""

from harness.proposer.heuristic import sweep_knob

__all__ = ["sweep_knob"]
