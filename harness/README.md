# harness/

The optimizer. Closes the loop between signals and changes.

- `synthesizer/` — generates code/policy artifacts (AutoHarness-style iterative refinement).
- `proposer/` — turns trace/eval signals into candidate mutations.
- `critics/` — gates: budget, eval-lift, safety, scope.
- `approver/` — auto-approve per `policies/` or hand to human via `ui/`.
- `executor/` — applies an approved candidate to `agent/` (the live config).
- `rollback/` — auto-revert on outcome regressions.
- `claude_code/` — Claude Code's operating instructions, scope, allowlist.

Not editable by the loop itself. The motor doesn't rewrite the motor.
