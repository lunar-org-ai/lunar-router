# evals/

The loss function. Without good evals, "self-improvement" is drift.

- `golden/` — labeled examples; grows with corrections (mutable, append-mostly).
- `suites/*.yaml` — eval suite definitions (which goldens, which rubrics).
- `rubrics/` — scoring functions (exact-match, LLM-as-judge, regex, custom).
- `runners/` — takes any `agent/` config + a suite, returns scores.
- `attribution/` — per-knob marginal contribution analysis.
- `reports/` — generated outputs per run; consumed by `ui/` and `harness/`.

This is the load-bearing pillar. Eval coverage is the project's real metric.
