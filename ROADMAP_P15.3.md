# P15.3 — Router (UniRoute, autonomous training)

Inspired by `arxiv 2502.08773` (UniRoute) for the decision rule and
`arxiv 2603.03329` (AutoHarness) for the self-improvement loop.
Reference impl: `/Users/diogovieira/Developer/open_project/OpenTracy/opentracy`.

> **Note.** Each sub-phase has its own `PLAN_P15.3.<n>.md` with the
> implementation detail. Where the PLAN supersedes this roadmap (e.g.
> renaming `kind="router"` → `kind="router_config"` to avoid colliding
> with the legacy `pipeline/route.yaml` knob, replacing the "write-through"
> PUT with the AHE-correct `record_manual_change()` path, dropping the
> Ψ heatmap / λ slider UI elements that don't exist in the design), the
> PLAN wins and this roadmap has been patched to reflect it.

## Locked decisions

- **Day-zero state:** empty. No `router_config` exists. `/run` falls back to `agent.models.default` until the harness fits the first config.
- **Retrain trigger:** Claude Code decides via MCP. No cron, no fixed thresholds. Drift/count signals are exposed as inputs; the brain interprets them.
- **Judge:** uses the same Agent SDK / MCP path the harness already uses (no separate Anthropic SDK call).
- **Augmentation budget:** `max_augmentation_samples = 500` per cycle (reference default).
- **Quality gate:** reference defaults — `min_auroc_improvement=0.0`, `min_win_rate=0.5`, `max_error_rate_increase=0.05`.
- **UI:** port the design's Router config panel verbatim. No layout/style changes unless extremely necessary.
- **AHE integration:** `router_config` is just another editable surface. Every change goes through the existing `proposer → critic → approver → executor → ledger` pipeline and surfaces in Evolution as `Lesson(kind="router_config")`. The legacy `kind="router"` (from `pipeline/route.yaml` knob edits) stays distinct.

## File layout (target)

```
router/
  __init__.py
  uniroute.py                      [P15.3.2 — port verbatim]
  core/
    embeddings.py                  [P15.3.1]
    clustering.py                  [P15.3.1]
    metrics.py                     [P15.3.1]
  models/
    llm_profile.py                 [P15.3.1]   # Ψ(h) vector
    llm_registry.py                [P15.3.1]
    llm_client.py                  [P15.3.1]
  training/
    kmeans.py                      [P15.3.3]
  feedback/
    trace_to_training.py           [P15.3.4]
    drift_detector.py              [P15.3.4]
    incremental.py                 [P15.3.4 — optional]
  augmentation/
    judge.py                       [P15.3.5]   # via Agent SDK
    goldens.py                     [P15.3.5]
    preference.py                  [P15.3.5]
  evaluation/
    evaluator.py                   [P15.3.6]
    metrics.py                     [P15.3.6]
    cache.py                       [P15.3.6]

harness/proposer/router_proposer.py [P15.3.7]   # the autotrainer pipeline,
                                                #  delegating to existing
                                                #  critic/approver/executor
harness/critics/router_critic.py    [P15.3.7]   # quality gate
harness/wakeup/                     [P15.3.9]   # scheduler + runner
harness/brain/transport.py          [P15.3.5]   # extracted complete() helper

techniques/routing/impl.py          [P15.3.8 — add UniRouteVariant]
runtime/server.py                   [P15.3.2 + P15.3.8 — endpoints inline]
runtime/embedder_pool.py            [P15.3.8 — process singleton]

harness/introspection/agent.py      [P15.3.9 — extend TOOLS list]
harness/introspection/lib.py        [P15.3.9 — add 2 tool implementations]
harness/introspection/mcp_server.py [P15.3.9 — register the 2 new tools]

ui/src/screens/Technical/RouterConfig.tsx
                                   [P15.3.10 — port from design verbatim]
ui/src/screens/Technical/RuleDrawer.tsx
                                   [P15.3.10 — drawer with UniRoute adapter]

versions/router_config_<n>.json    [data artifact, no new code]
ledger/decisions/router_<iso>.json [P15.3.9 — wakeup decision artifacts]
```

## Dependency graph

```
P15.3.1 ──┬── P15.3.2 ──┐
          ├── P15.3.3 ──┤
          ├── P15.3.4 ──┼── P15.3.7 ── P15.3.9
          ├── P15.3.5 ──┤      │
          └── P15.3.6 ──┘      └── P15.3.8 ── P15.3.10
```

P15.3.5 needs **a working harness brain transport**. The existing
`harness/introspection/agent.py` already provides one (Anthropic API when
`ANTHROPIC_API_KEY` is set, `claude --print` CLI fallback). P7.7 (Agent
SDK proper) is a future upgrade — when it lands, only
`harness/brain/transport.py` changes; the rest of P15.3 is unaffected.

---

## P15.3.1 — Core + models scaffolding

**Goal.** Land the math primitives and the `LLMProfile`/`LLMRegistry` data structures so every later phase can import them.

**Deliverable.**
- `router/core/embeddings.py` — `PromptEmbedder` (sentence-transformers `all-MiniLM-L6-v2`, local CPU, 384-dim).
- `router/core/clustering.py` — `ClusterAssigner` + `ClusterResult` (soft probabilities + hard one-hot).
- `router/core/metrics.py` — metric primitives (cross-entropy, MSE) used by Ψ.
- `router/models/llm_profile.py` — `LLMProfile` with `Ψ` vector of shape `(K,)`, `cost_per_1k_tokens`, `get_expected_error(phi)`, `get_cluster_error(c)`.
- `router/models/llm_registry.py` — registry keyed by `model_id` with `get_available_models()`.
- `router/models/llm_client.py` — abstract `LLMClient` + concrete `AnthropicClient` / `OpenAIClient` thin wrappers. (No Agent SDK here — those are the *judged* models, not the brain.)
- `versions/router_config_v0.json` — schema reference, empty Ψ, K=0. Documents the artifact format.

**Files referenced.** `core/embeddings.py`, `core/clustering.py`, `models/*.py` from reference repo. Port semantics verbatim; rename only what conflicts with this repo's namespaces.

**DoD.** `pytest router/tests/test_core.py router/tests/test_profile.py` green. `from router.core.embeddings import PromptEmbedder` works without runtime errors.

---

## P15.3.2 — UniRoute decision engine

**Goal.** A pure routing function: given a prompt, return `(selected_model, expected_error, reasoning)` without executing anything.

**Deliverable.**
- `router/uniroute.py` — port `UniRouteRouter` verbatim. Key methods: `route(prompt)`, `route_batch(prompts)`, `get_best_model_for_cluster(c)`, `analyze_routing_distribution(prompts)`.
- Cold-start fallback: if `registry` is empty or `cluster_assigner` is unfitted, `route()` raises `RouterColdStartError`. Caller decides what to do.
- Endpoint `POST /v1/router/decide` (in `runtime/routes/router.py`): body `{ prompt, allowed_models?, cost_weight_override? }`, returns full `RoutingDecision.to_dict()`. Used by debug tools and the UI.
- Endpoint `GET /v1/router/config` — returns current config metadata (version, K, model count, last fit time, λ).

**Deps.** P15.3.1.

**DoD.** Unit tests for the decision math (mock embedder + 2 fake profiles). Integration test against `/v1/router/decide` with a synthesized config.

---

## P15.3.3 — KMeans trainer + first-fit gate

**Goal.** Take a corpus of prompt embeddings → fit clusters → produce a serializable `ClusterAssigner`.

**Deliverable.**
- `router/training/kmeans.py` — port `kmeans_trainer.py`. Inputs: list of prompts (or `PromptDataset` once P15.3.4 introduces it) + K. Output: `KMeansTrainResult(assigner, silhouette, inertia, cluster_sizes, fitted_from)`.
- `router/training/snapshot.py` — `snapshot_clusters_only(result)` writes a **partial** `router_config_<n>.json` with centroids only and `model_psi={}`; the `current` pointer is **not** updated until P15.3.7's executor stitches Ψ in and promotes.
- First-fit gate: refuses to fit until corpus size ≥ `min_corpus_size` (default 200, configurable) and `N >= 2K`. Returns `NotEnoughDataError`. The Claude Code brain (P15.3.9) checks this and decides whether to wait or retrain.
- Cluster quality: log silhouette score + within-cluster sum of squares (sample capped at 5000 for O(N²)).

**Deps.** P15.3.1.

**DoD.** Fit → save → load round-trip preserves cluster assignments bit-for-bit. Silhouette score logged on every fit.

---

## P15.3.4 — Trace → training + drift detector

**Goal.** Make the autonomy loop *legible*: production traces become training data; deltas in input distribution become a drift signal.

**Deliverable.**
- `router/feedback/trace_to_training.py` — port semantics. Reads `traces/` from this project's existing trace store, transforms into `PromptDataset` + per-trace label (response quality from auto-eval if present, else null). Filters: must have terminal status, must have non-null `prompt`, dedup by hash.
- `router/feedback/drift_detector.py` — port. Computes drift score = mean cosine distance from new traces' embeddings to nearest current centroid. Output: `DriftReport(score, threshold_recommendation, n_new_samples, last_fit_age)`.
- `router/feedback/incremental.py` (optional) — incremental Ψ update without full refit. Land scaffolding only; not wired in P15.3.7 unless we hit eval cost issues.

**Deps.** P15.3.1.

**DoD.** Given 1k synthetic traces and a fitted assigner, `DriftDetector.report()` returns a numeric score and the corresponding threshold. `TraceToTraining.build()` round-trips a `PromptDataset` of expected size.

---

## P15.3.5 — Augmentation: judge via Agent SDK + goldens + preference data

**Goal.** Turn unlabeled traces into labeled pairs via LLM-as-judge, using **the same Agent SDK / MCP path the harness uses** so we don't have a second cerebro.

**Hard prereq.** P7.7 (Claude Code Agent SDK integration) must be live.

**Deliverable.**
- `router/augmentation/judge.py` — `LLMJudge.score(prompt, response_a, response_b) → win | loss | tie + rationale`. Implementation: dispatches to the same Agent SDK harness instance, with a system prompt template defined in `router/augmentation/judge_prompt.md`. Caps at `max_augmentation_samples=500` per cycle.
- `router/augmentation/goldens.py` — `GoldenAugmenter` extends the existing eval suite goldens with judged production pairs. Outputs an augmented `PromptDataset`.
- `router/augmentation/preference.py` — `PreferenceDataset` of `(prompt, winner, loser, rationale)` tuples used to refine Ψ via the preference-data path.

**Deps.** P15.3.1, P15.3.4, P7.7.

**DoD.** Smoke test: judge 5 trace pairs end-to-end, capture rationale, persist to `evals/preference_pairs/<run_id>.jsonl`. Failure-on-no-Agent-SDK is explicit.

---

## P15.3.6 — Evaluation harness for router configs

**Goal.** Score a router_config (current or candidate) against a fixed eval suite. This is the comparator the AHE quality gate needs.

**Deliverable.**
- `router/evaluation/evaluator.py` — `RouterEvaluator.evaluate(router) → EvaluationResult` with AUROC, win rate, avg error, per-cluster error, model distribution.
- `router/evaluation/metrics.py` — `RoutingMetrics` (AUROC, calibration, win rate vs each baseline).
- `router/evaluation/cache.py` — `ResponseCache` keyed by `(model_id, prompt_hash)` so re-evals don't re-call the LLM. Stored in `evals/_response_cache/`.
- Wire into the existing `evals/` substrate from P15.2: a router config eval becomes a special suite type `kind: "router"` with the candidate config as input.

**Deps.** P15.3.1, P15.3.2, P15.3.5.

**DoD.** Evaluating an empty config returns AUROC=0.5 (uninformative). Evaluating a fitted config on its training data returns AUROC > 0.5. Cache hit ratio logged.

---

## P15.3.7 — Router proposer in the AHE harness

**Goal.** Replace the reference's monolithic `auto_trainer.py` with a proposer that hands each step off to the *existing* harness machinery (critic, approver, executor, ledger).

**Deliverable.**
- `harness/proposer/router_proposer.py` — `RouterProposer.propose() → Proposal` with `mutations=[Mutation(file="versions/router_config_v<n>.json", payload=candidate_payload)]` and `source="claude_code"`. Internally:
  1. Pull traces since `router_config_current.created_at` (via `router/feedback/store_adapter.py`).
  2. Run `TraceToTraining.add_traces()` + `GoldenAugmenter.augment()` (cap 500).
  3. Re-fit clusters (`KMeansTrainer.train()`).
  4. Recompute Ψ via `harness/proposer/router_psi_compute.py` (bench Ψ from cache + production Ψ from traces + preference-data refinement, blended at `production_alpha=0.3`).
  5. Build inline `candidate_payload` carried on `Mutation.payload`.
- `harness/proposer/router_psi_compute.py` — Ψ blend math (split out of the reference's monolith).
- `harness/critics/router_critic.py` — `RouterCritic(Critic)` at `CriticStage.POST`, applies to `kind="router_config"`. Runs `RouterEvaluator.evaluate()` for current + candidate, applies the locked quality gate (AUROC≥0, win_rate≥0.5, max_err_increase≤0.05).
- `harness/types.py` — `kind_from_mutations()` learns `versions/router_config_*` → `"router_config"` (the legacy `"router"` for `pipeline/route.yaml` knobs stays distinct).
- Approver wiring: **zero code change** — existing `Policy.overrides[kind]` already accepts `kind="router_config"`. Documented `policy.yaml` example added.
- Executor: `harness/executor/promote.py:apply_router_candidate(payload)` writes `versions/router_config_<n>.json` (+ centroids `.npz` sidecar) atomically and flips the `current` pointer. Existing `promote()` flow gains a small `kind == "router_config"` dispatch.
- `Lesson(kind="router_config", proposal_source="claude_code" | "human")`. `voices.yaml` gets the new lesson kind templates.

**Deps.** P15.3.2, P15.3.3, P15.3.4, P15.3.5, P15.3.6, plus existing harness phases (P3, P4, P14).

**DoD.** End-to-end smoke: `python -m harness.proposer.router_proposer_smoke` → critic runs → approver decides → ledger gets a `Lesson(kind="router_config")` entry → `versions/router_config_v1.json` exists. Visible in Evolution timeline.

---

## P15.3.8 — Engine wiring + RoutingDecision in trace

**Goal.** Make `/run` actually use the router and surface the decision in the trace so the UI can show "why this model".

**Deliverable.**
- `techniques/routing/impl.py` — new variant `UniRouteVariant` alongside `SmallFirstVariant`. Reads `router_config_current`, calls `UniRouteRouter.route(prompt)`, sets `ctx.routing.model` + `ctx.routing.decision`. Cold-start raises `RouterColdStartError` → variant catches and falls back to `knobs.small` (or `knobs.default`), stamping `fallback_reason: "router_not_initialized"`.
- `runtime/embedder_pool.py` — process-singleton `PromptEmbedder`. Lazy + thread-safe; `warm()` hooked into agent compile when `route.variant == "uniroute"`.
- `runtime/executor/pipeline.py` — `StageRecord` gets optional `routing_decision: dict`. Existing `dataclasses.asdict` trace writer flows it through automatically.
- `runtime/types.py` — `StageOutcome` Pydantic gains `routing_decision: Optional[dict]`. `RunResponse.stages` exposes it.
- `agent/pipeline/route.yaml` — `variant: small_first` stays default; `uniroute` is opt-in.
- `runtime/server.py` — `PUT /router/config` for manual `λ` overrides via `harness.executor.promote.record_manual_change()` so the edit emits `Lesson(kind="router_config", proposal_source="human")`, bumps the version, and rolls back through `/v1/versions/{v}/rollback` like any other change. **Not write-through** — AHE-aligned manual edit pipeline.

**Deps.** P15.3.2, P15.3.7.

**DoD.** Hit `/run` with `route.variant: uniroute` → trace's route stage carries `routing_decision`. Cold-start path returns the fallback model with `fallback_reason: "router_not_initialized"`. `PUT /router/config {"cost_weight": x}` produces a visible Lesson + bumped version + rollback works.

---

## P15.3.9 — MCP tools: Claude Code decides retrain

**Goal.** No cron, no hardcoded threshold. Claude Code (acting as the brain via MCP) is the one that decides whether to fire `RouterProposer`.

**Deliverable.**
- `router/feedback/health.py` — `RouterHealth` dataclass + `compute_router_health()` (cold-start safe; reads config, traces, cache, drift).
- New MCP tools registered in `harness/introspection/agent.py:TOOLS` and exposed via `harness/introspection/mcp_server.py`:
  - `router_health_check()` → `RouterHealth.to_dict()`. Pure read.
  - `propose_router_retrain(rationale)` → triggers `RouterProposer.propose()` + critic + approver + executor + ledger. Returns `{action, lesson_id, version, reason?}`. Gated by `Policy.overrides["router_config"]`.
- Wake-up scheduler: `harness/wakeup/{scheduler.py, runner.py, prompt.py}`. Hook in `runtime/store/traces.py:write_trace()` calls `wakeup.scheduler.maybe_fire()` fire-and-forget when N traces accumulate (default 50, configurable via `HARNESS_ROUTER_WAKEUP_N`). Lockfile prevents concurrent wake-ups. Runner composes the prompt + health JSON, calls `harness.introspection.agent.introspect()` (reuses existing brain transport), captures Claude Code's response.
- Decision rationales persist to `ledger/decisions/router_wakeup_<iso>.json` for both proposed and skipped paths, so the UI can show "Claude Code declined retrain at T because drift was 0.04, below threshold he set".

**Deps.** P15.3.7, working harness brain transport (already exists; P7.7 is an additive upgrade).

**DoD.** Wake-up fires after N synthetic traces → Claude Code is invoked → either a `Lesson(kind="router_config")` shows up in Evolution OR a decision artifact at `ledger/decisions/router_wakeup_*.json` explains the skip. Both paths exit cleanly. `/run` is never blocked by the hook (<5ms overhead).

---

## P15.3.10 — UI Router config (port verbatim from design)

**Goal.** Plug real router data into the design's Router config panel, **without** modifying its layout, classes, paddings, or proportions.

**Design ↔ backend reconciliation.** The design is **rule-based** (declarative `if when then model` rules with share / cost / authorship); UniRoute is **embedding-based** (clusters + Ψ matrix). They're different paradigms. The earlier draft of this phase mentioned "Ψ heatmap, cluster distribution chart, λ slider, candidate queue" — none exist in the design and none are added here. UniRoute is mapped to the design's existing `isDefault: true` rule row; UniRoute internals render inside the existing `RuleDrawer`'s three tabs (Overview / Sample matches / History) using only the design's existing classes.

**Deliverable.**
- `ui/src/screens/Technical/RouterConfig.tsx` — port `RouterConfig` from `<DESIGN>/screens/Technical.jsx:670-861` verbatim (layout, classes, paddings, summary cards, filter pills, search, table, modal). Wire data to real endpoints.
- `ui/src/screens/Technical/RuleDrawer.tsx` — port `RuleDrawer` from `<DESIGN>:862-944`. When the rule is the UniRoute default, the Overview / Samples / History tabs render UniRoute-derived data via the design's existing `.sheet-section` + `.meta-grid` chrome — no new components.
- `ui/src/screens/Technical/NewRuleModal.tsx` — port `NewRuleModal` from `<DESIGN>:945-1006`. **Disabled** for v1 (manual rules engine deferred); same chrome with "Coming soon" submit copy.
- `ui/src/screens/Technical/TraceDrawer.tsx` (extend existing P15.1 component) — render `routing_decision` from P15.3.8 in a new `.sheet-section` using existing `.meta-grid` rows.
- New backend endpoints (proxied via `backend/channels/router/`):
  - `GET /v1/router/rules` — synthesized rules list (v1 returns the single UniRoute default row).
  - `GET /v1/router/candidates` — pending `Lesson(kind="router_config", status="awaiting_review")` entries.
  - `GET /v1/router/health` — same payload as the MCP tool.
- No fabrication: cold-start cells render `—`. Candidate queue rows link to the matching `/review/<lesson_id>`.

**Deps.** P15.3.8.

**DoD.** Switch view → Technical → Router config: page renders pixel-faithful to the design (zero diff on shape / spacing / colors; text content can differ). Default-rule row reflects UniRoute state — fitted state shows real K / drift / λ in the drawer's Overview tab; cold-start shows `—` and the cold-start rationale. Pending candidates link to Review. `+ Add routing rule` is visibly disabled with a "Coming soon" tooltip.

---

## Out of scope for P15.3

- Per-tenant or per-channel router configs (single global config for v1).
- Online incremental Ψ updates (`router/feedback/incremental.py` lands as scaffold only).
- Cost forecast / budget UI in Router config (deferred to a later phase).
- Auto-rollback on router regressions: persists in policy but doesn't fire (matches the existing auto-rollback deferral).
