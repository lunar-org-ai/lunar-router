# policies/

Human-set rules for the harness. Not mutated by the loop.

- `auto_approve.yaml` — per change-type: Auto / Review / Off, with thresholds.
- `rollback_triggers.yaml` — auto-revert conditions (CSAT drop, eval regression).
- `budget.yaml` — cost / latency caps per request type.

Edited by humans through `ui/Policies` or directly in code review.
