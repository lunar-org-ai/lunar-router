# PLAN — P15.3.6 · Evaluation harness for router configs

| Field | Value |
|---|---|
| Phase | P15.3.6 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | P15.3.1 (core + models), P15.3.2 (uniroute decision), P15.3.5 (judge for cache labels) |
| Unblocks | P15.3.7 (router_critic uses `RouterEvaluator` to gate promotions) |
| Reference | `/Users/diogovieira/Developer/open_project/OpenTracy/opentracy/evaluation` |

## Goal

Score a `router_config` (current or candidate) against a fixed eval suite,
producing the comparator the AHE quality gate needs: `delta_auroc`,
`delta_avg_error`, `delta_win_rate` between two configs.

This phase ships the **scoring** layer — the critic in AHE terms.
It does **not** decide promotions (that's the approver in P15.3.7) and
it does **not** run the LLM at eval time (responses come from a
`ResponseCache` populated up-front).

The roadmap DoD for this phase:
- Evaluating an **empty config** returns AUROC = 0.5 (uninformative — Ψ is zeros, predictions are uniform).
- Evaluating a **fitted config on its training data** returns AUROC > 0.5.
- Cache hit ratio is logged on every eval.

## Scope

### In scope
- `router/evaluation/__init__.py` — package marker.
- `router/evaluation/metrics.py` — port `compute_auroc`, `compute_apgr`, `compute_cpt`, `compute_pgr_at_savings`, `compute_win_rate`, plus the `RoutingMetrics` dataclass.
- `router/evaluation/baselines.py` — port `RandomBaseline`, `OracleBaseline`, `AlwaysStrongBaseline`, `AlwaysWeakBaseline` + `BaselineResult`.
- `router/evaluation/cache.py` — port `CachedResponse` + `ResponseCache`. Default storage path: `evals/_response_cache/cache.jsonl`.
- `router/evaluation/evaluator.py` — port `RouterEvaluator` + `EvaluationResult` + `ParetoPoint`. Slim: drop the `populate_from_dataset` LLM-call path (we don't replay live during eval); keep the Pareto λ sweep.
- `evals/types.py` (extend) — add optional `kind: Literal["agent", "router"] = "agent"` to `Suite` for back-compat; existing suites default to `"agent"`.
- `evals/runners/router_runner.py` — **new**. Runner for `kind: "router"` suites: loads a `router_config`, runs `UniRouteRouter.route()` over each golden prompt, looks up the (prompt, selected_model) response in `ResponseCache`, computes per-prompt rubric scores, aggregates into `EvaluationResult`. Writes a standard `evals/reports/<id>.json` so the existing UI surface still works.
- `evals/runners/runner.py` (extend) — dispatch on suite `kind`: `"agent"` keeps current behavior, `"router"` calls `router_runner`.
- `router/evaluation/cache_populator.py` — helper that walks existing `evals/reports/*.json` and harvests per-(prompt, model) responses into the cache. Best-effort — flags gaps the operator must fill via T6's offline-replay tool.
- `tools/populate_response_cache.py` (script) — explicit one-shot offline replay: given a list of model IDs + a goldens dataset, calls each model on each prompt and stores in `ResponseCache`. Used to seed the cache before the first router eval.
- `router/tests/test_metrics.py`, `test_cache.py`, `test_evaluator.py`, `test_router_runner.py`.

### Out of scope (deferred)
- LLM replay at eval time → script in `tools/` is offline; eval reads cache only.
- Multi-judge ensembling for cache labels → single judge from P15.3.5.
- Cache invalidation policy (when a model upgrades, do we drop its rows?) → manual `ResponseCache.purge_model(model_id)` available; no automation in this phase.
- Streaming the cache from cloud storage → local file only.
- Exposing `EvaluationResult` via a new HTTP endpoint → critic in P15.3.7 reads it directly; UI consumption deferred to P15.3.10.

## Reference → target file map

| Reference | Target | Port mode |
|---|---|---|
| `evaluation/metrics.py` | `router/evaluation/metrics.py` | verbatim |
| `evaluation/baselines.py` | `router/evaluation/baselines.py` | verbatim |
| `evaluation/response_cache.py` | `router/evaluation/cache.py` | verbatim — adjust default path to `evals/_response_cache/cache.jsonl` |
| `evaluation/evaluator.py` | `router/evaluation/evaluator.py` | partial — port `RouterEvaluator`, `EvaluationResult`, `ParetoPoint`; **drop** any LLM-call paths (cache-only) |
| — | `evals/runners/router_runner.py` | new |
| — | `router/evaluation/cache_populator.py` | new — reads `evals/reports/*.json` |
| — | `tools/populate_response_cache.py` | new — explicit offline replay script |

## Pre-work

- Verify P15.3.5 produced or could produce judge verdicts → those become the `is_correct` labels per `(prompt, model)` cache entry. If P15.3.5's smoke test ran successfully, `evals/preference_pairs/pp_<...>.jsonl` has labels we can fold in. If not, the cache populator runs in "unlabeled" mode and the AUROC test in T8 uses a synthesized labeled cache.
- No new pyproject deps. `numpy` already present; `sklearn.metrics.roc_auc_score` is part of `scikit-learn` already in `[router]`.

Verify before starting:
- P15.3.1, P15.3.2, P15.3.5 green.
- `from router.augmentation.judge import LLMJudge` works (P15.3.5 deliverable).

## Tasks (atomic, ordered)

### T1 — Port `metrics.py`
Copy `<REF>/evaluation/metrics.py` → `router/evaluation/metrics.py`. Verbatim. Adjust imports (none expected — file is self-contained module + numpy).

Public surface (recap):
- `compute_auroc(predicted_errors, actual_errors) -> float` — sklearn AUROC under the hood. Returns 0.5 when predictions are uniform / data is degenerate.
- `compute_apgr`, `compute_cpt`, `compute_pgr_at_savings` — Pareto-curve metrics from the UniRoute paper.
- `compute_win_rate(routed, baseline) -> float` — fraction of prompts where router beats baseline.
- `RoutingMetrics` dataclass — collects all of the above plus `avg_cost`, `avg_quality`.

### T2 — Port `baselines.py`
Copy `<REF>/evaluation/baselines.py` → `router/evaluation/baselines.py`. Verbatim. Adjust imports to `router.models.llm_profile` if any.

Surface (recap): four baselines (`RandomBaseline`, `OracleBaseline`, `AlwaysStrongBaseline`, `AlwaysWeakBaseline`) each with `.predict(prompts) -> list[BaselineResult]`. Used by `RouterEvaluator` to compute win-rate.

### T3 — Port `cache.py` (rename from `response_cache.py`)
Copy `<REF>/evaluation/response_cache.py` → `router/evaluation/cache.py`. Then:

- Default storage path: `evals/_response_cache/cache.jsonl` (was `_response_cache/cache.jsonl` in the reference; same shape, repo-anchored).
- Create `evals/_response_cache/.gitignore` ignoring `*.jsonl` (cache is large + machine-specific; mirror what `evals/reports/` does).
- Surface unchanged: `ResponseCache(path)`, `.add(prompt, model_id, response, cost, latency_ms, is_correct?)`, `.get(prompt, model_id)`, `.has(prompt, model_id)`, `.get_all_models(prompt)`, `.coverage(model_id)`, `.save()`, `.populate_from_dataset(...)` (drop or rewrite to fail loudly — see Out of scope).
- `CachedResponse` dataclass keeps `(prompt_hash, model_id, response_text, cost_usd, latency_ms, is_correct: Optional[bool], metadata: dict)`.

### T4 — Port `evaluator.py` (slim)
Copy `<REF>/evaluation/evaluator.py` → `router/evaluation/evaluator.py`. Then:

- **Drop** any code path that does an LLM call. `RouterEvaluator.evaluate()` reads from `ResponseCache` only; if a `(prompt, model)` is missing, raise `CacheGapError` with the specific gap so the operator can run `tools/populate_response_cache.py` to fix it.
- Keep the Pareto λ sweep: `evaluate(router, lambdas=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0])` returns `EvaluationResult.pareto_curve` with one `ParetoPoint` per λ.
- Adjust imports: `from router.uniroute import UniRouteRouter, RoutingDecision`, etc.
- `EvaluationResult.summary()` adds a "cache_hit_ratio" line so logs surface the ratio per the roadmap DoD.

### T5 — Suite schema: introduce `kind`
Edit `evals/types.py`:
```python
class Suite(BaseModel):
    suite: str
    description: str = ""
    kind: Literal["agent", "router"] = "agent"   # NEW; defaults preserve backcompat
    goldens: list[str]
    rubrics: list[Rubric] = []
    aggregation: Literal["mean", "min", "max"] = "mean"
    # router-specific (only required when kind == "router")
    config_path: Optional[str] = None             # path to router_config_<n>.json
    baselines: list[str] = []                     # ["random", "oracle", "always_strong"]
    lambdas: list[float] = [0.0, 0.1, 0.5, 1.0, 2.0]
```
Add a validator: `kind == "router"` requires `config_path` to be set; else raise.

Update existing suite YAML loader to log a deprecation-free INFO line when `kind` is omitted (defaults to agent).

### T6 — `router_runner.py`
Create `evals/runners/router_runner.py`:

```python
def run_router_suite(suite: Suite, *, cache: ResponseCache) -> ReportSummary:
    """Run a kind='router' suite. Reads cache only — no LLM calls."""
    config_path = Path(suite.config_path)
    assigner, registry, _ = load_current_config_at(config_path)
    embedder = _get_or_init_embedder()
    router = UniRouteRouter(embedder, assigner, registry, cost_weight=0.0)

    goldens = load_goldens(suite.goldens)
    evaluator = RouterEvaluator(
        router=router,
        cache=cache,
        baselines=[BASELINES[name] for name in suite.baselines],
    )
    result: EvaluationResult = evaluator.evaluate(
        prompts=[g.prompt for g in goldens],
        lambdas=suite.lambdas,
    )
    report = _evaluation_result_to_report(result, suite, goldens)
    _persist_report(report)   # → evals/reports/<id>.json
    return report
```

`_evaluation_result_to_report` flattens the Pareto curve + RoutingMetrics into the existing `Report` shape so the UI's `/evals/reports/{id}` endpoint shows router evals next to agent evals naturally.

### T7 — Dispatcher in `runner.py`
Edit `evals/runners/runner.py`:
```python
def run_suite(suite: Suite) -> ReportSummary:
    if suite.kind == "router":
        from .router_runner import run_router_suite
        return run_router_suite(suite, cache=_global_cache())
    return run_agent_suite(suite)   # existing behavior, renamed if needed
```
Lazy import keeps the agent path free of the router subtree's torch dep.

### T8 — `cache_populator.py` + `tools/populate_response_cache.py`

`router/evaluation/cache_populator.py`:
```python
def harvest_from_reports(reports_dir: Path, cache: ResponseCache) -> int:
    """Walk evals/reports/*.json. For each report, extract (prompt, model_id, response, score)
    and add to cache with is_correct = score >= rubric_threshold.
    Returns count of rows added. Logs per-suite gap stats."""
```

`tools/populate_response_cache.py`:
```python
"""One-shot offline replay. Given goldens + model_ids, calls each model on each prompt
and writes (prompt, model_id, response) to the cache.

Usage:
  python tools/populate_response_cache.py \
      --goldens evals/golden \
      --models claude-haiku-4-5,claude-sonnet-4-5,gpt-5 \
      --out evals/_response_cache/cache.jsonl
"""
```
Uses the existing model clients (`AnthropicClient` from P15.3.1; OpenAI is still stubbed → script raises with the deferral message if asked for `gpt-*`).

This script is the **explicit cache seeder**. The cycle is:
1. Operator (or harness in P15.3.7) calls the script before the first router eval.
2. Cache fills.
3. `RouterEvaluator.evaluate()` reads from cache — fast, deterministic, replayable.

### T9 — Tests

Create `router/tests/test_metrics.py`:
- `test_auroc_returns_half_on_uniform_predictions` — synthetic uniform predictions → AUROC ≈ 0.5.
- `test_auroc_returns_above_half_on_separable` — predictions ordered by actual → AUROC ≈ 1.0.
- `test_apgr_smoke` — synthetic Pareto curve → APGR finite + monotonic in cost.
- `test_win_rate_basic` — routed beats baseline on 7/10 prompts → win_rate == 0.7.

Create `router/tests/test_cache.py`:
- `test_cache_add_get_round_trip` — add → get → equality.
- `test_cache_save_load` — round-trip JSONL.
- `test_cache_coverage` — 5 prompts × 2 models, coverage("model_a") == 5.
- `test_cache_gap_raises_in_evaluator` — empty cache + non-empty prompt set → `CacheGapError` with prompt+model in message.

Create `router/tests/test_evaluator.py`:
- `test_evaluate_empty_config_yields_auroc_half` — Ψ all zeros, fully populated cache → AUROC ≈ 0.5 ± tolerance. **This is the roadmap DoD.**
- `test_evaluate_fitted_config_yields_auroc_above_half` — Ψ fitted to match cache labels → AUROC > 0.5. **Roadmap DoD.**
- `test_evaluate_pareto_curve_monotonic` — at higher λ, avg_cost decreases. (Pareto curve direction sanity.)
- `test_evaluate_logs_cache_hit_ratio` — caplog captures `"cache_hit_ratio=1.0"` line at INFO.

Create `router/tests/test_router_runner.py`:
- `test_router_runner_writes_report` — feed a `kind="router"` suite + tmp cache + tmp config → `evals/reports/<id>.json` exists with router-specific fields.
- `test_runner_dispatches_on_kind` — `kind="agent"` calls existing path; `kind="router"` calls new path. Verify via mock.
- `test_router_suite_validation` — `kind="router"` without `config_path` raises Pydantic validation error.

### T10 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
python -m pytest router/tests/test_metrics.py router/tests/test_cache.py router/tests/test_evaluator.py router/tests/test_router_runner.py -v
python -m pytest router/tests/ -v   # full router test surface still green
python -m pytest -v                  # full project test surface still green
```

Manual smoke (optional): seed cache + run a router suite end-to-end:
```
python tools/populate_response_cache.py --goldens evals/golden --models claude-haiku-4-5
# write a kind: "router" suite YAML pointing at versions/router_config_v1.json (synthesized for the smoke)
python -m evals.runners --suite evals/suites/router_smoke_v0.yaml
ls evals/reports/   # confirm a new report appears
```

## Acceptance criteria (DoD)

1. `python -m pytest router/tests/test_metrics.py router/tests/test_cache.py router/tests/test_evaluator.py router/tests/test_router_runner.py -v` is green.
2. **Empty-config AUROC = 0.5** (within ε=0.05): `RouterEvaluator.evaluate(router_with_empty_psi)` over a populated cache returns `metrics.auroc ≈ 0.5`. (Roadmap DoD.)
3. **Fitted-config AUROC > 0.5**: `RouterEvaluator.evaluate(router_with_fitted_psi)` over the same cache returns `metrics.auroc > 0.5`. (Roadmap DoD.)
4. **Cache hit ratio is logged** at `INFO` on `router.evaluation.evaluator` for every `evaluate()` call. (Roadmap DoD.)
5. `evals/types.py` accepts `kind: "router"` suites; existing `kind: "agent"` suites still load and run unchanged.
6. `evals/runners/runner.py` dispatches `kind="router"` to `router_runner`; `kind="agent"` flows are unchanged.
7. A `kind="router"` suite end-to-end produces a standard `evals/reports/<id>.json` consumable by the existing `/v1/evals/reports/{id}` endpoint (+ UI Reports view).
8. `RouterEvaluator.evaluate()` with a missing `(prompt, model)` raises `CacheGapError` whose message names the specific gap (no silent zeros).
9. `tools/populate_response_cache.py --models claude-haiku-4-5 ...` runs and grows `evals/_response_cache/cache.jsonl` by N rows.
10. No regressions: full `python -m pytest` green.

## Risks / open questions

- **Cache cold-start.** First router eval needs `(prompt × model)` coverage. Two paths: (a) `tools/populate_response_cache.py` does a one-shot replay before the first eval (controlled cost, explicit), (b) `cache_populator.harvest_from_reports` salvages from existing eval reports (free but partial — reports cover one model per run, not the matrix). T8 + T9 ensure the operator can do (a). The harness in P15.3.7 will need a hook to call (a) automatically when a model joins the registry.
- **AUROC needs binary `is_correct` labels.** Cache rows must carry `is_correct: bool`. Sources: rubric scores from existing reports (`harvest_from_reports`) **or** judge verdicts from P15.3.5 (`PreferenceDataset` → A_correct, B_correct flags). When labels are missing, `compute_auroc` returns NaN. Document that "fitted-config AUROC > 0.5" DoD requires labeled cache; the synthetic test in T9 builds its own labels.
- **Pareto sweep cost (cache hits only).** All-cache eval is cheap (lookup + numpy). For 8 goldens × 4 models × 6 λ values = 192 routing decisions per evaluation, each ~ms. Acceptable.
- **Suite schema change is observable.** Existing UI consumers (Eval suites screen) read `Suite` shape. Adding optional fields with defaults keeps back-compat. Verify during T5 that the UI's `SuiteDetail` Pydantic model accepts the new fields without erroring (existing UI screens may need to ignore them — extra-fields-allowed mode).
- **`tools/populate_response_cache.py` cost.** Real cost: M models × N goldens × ~500 tokens average. Default goldens are 8 prompts; 3 models = 24 calls ≈ $0.05. Cheap. Becomes non-trivial only when goldens grow to thousands.
- **Cache as a single JSONL.** Easy to reason about, fast to load (<10 MB at thousands of rows). Won't scale to millions of rows; if it grows beyond a few hundred MB, switch to DuckDB-backed storage in a follow-up. Not a P15.3 concern.
- **`is_correct` label drift.** A judge that improves over time relabels old cache rows differently. Mitigation: store `judge_version` in `CachedResponse.metadata`; relabel rows on judge upgrade. P15.3.7 owns this; flag for that PLAN.
