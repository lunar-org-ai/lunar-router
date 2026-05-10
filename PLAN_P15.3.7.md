# PLAN — P15.3.7 · Router proposer in the AHE harness

| Field | Value |
|---|---|
| Phase | P15.3.7 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | P15.3.2 (uniroute), P15.3.3 (kmeans), P15.3.4 (trace_to_training, drift), P15.3.5 (judge), P15.3.6 (evaluator) |
| Unblocks | P15.3.8 (engine wiring), P15.3.9 (MCP `propose_router_retrain`) |
| Reference | `/Users/diogovieira/Developer/open_project/OpenTracy/opentracy/training/auto_trainer.py` (semantics only — pipeline structure changes) |

## Goal

Replace the reference's monolithic `auto_trainer.py` with a **proposer that
hands each step off to the existing harness machinery** (proposer →
critic → approver → executor → ledger). After this phase, every router
config change — auto or manual — is a regular AHE Lesson, rolls back via
`/versions/{v}/rollback`, and shows up in the Evolution timeline next to
prompt edits.

This phase ships **the autonomous promotion path**. It does **not** wire
routing into `/run` (that's P15.3.8 — runtime consumes the promoted
`router_config_current` pointer) and it does **not** decide *when* to
fire (that's P15.3.9 — Claude Code via MCP).

## AHE alignment recap

Per AutoHarness (arxiv 2603.03329):

- **Single editable surface** — `versions/router_config_<n>.json` is **one** versioned artifact (with sidecar `.npz` for centroids per P15.3.3 storage decision). Pointer `versions/router_config_current` references one version at a time.
- **Same pipeline for every change** — proposer → critic → approver → executor → ledger. Auto and manual edits travel the same route.
- **`proposal_source` attribution** — `"claude_code"` (auto) or `"human"` (manual via P15.3.8). Both surface as Lessons.
- **Rollback uniform** — existing `/versions/{v}/rollback` handles router config rollback the same as prompt rollback (executor extension in T4 ensures the rollback path knows how to flip the `current` pointer).

## Scope

### In scope
- `harness/types.py` (extend) — `kind_from_mutations()` learns `versions/router_config_*` → `"router_config"` (distinct from legacy `"router"` which targets `pipeline/route.yaml` knobs).
- `harness/proposer/router_proposer.py` — `RouterProposer.propose() → Proposal`. Orchestrates: pull traces (via `store_adapter`) → augment goldens (via `GoldenAugmenter`) → re-fit clusters (`KMeansTrainer`) → recompute Ψ (preference-data path + cache replay) → build candidate `router_config_<n>.json` payload → emit `Proposal(source="claude_code", mutations=[...], prediction=Prediction(...))`.
- `harness/proposer/router_psi_compute.py` — helper that builds Ψ tables: blends benchmark Ψ (from goldens via `RouterEvaluator` cache) with production Ψ (from `TraceToTraining.compute_psi_updates`) at `production_alpha=0.3` (locked default from P15.3 roadmap).
- `harness/critics/router_critic.py` — `RouterCritic(Critic)` at `CriticStage.POST` (after-eval). Runs `RouterEvaluator.evaluate()` for both current and candidate, computes `delta_auroc`, `delta_avg_error`, `delta_win_rate`, applies the locked quality gate (`min_auroc_improvement=0.0`, `min_win_rate=0.5`, `max_error_rate_increase=0.05`). Emits `CriticVerdict(passed=bool, reason=...)`.
- `harness/executor/promote.py` (extend) — `apply_router_candidate(candidate_payload: dict, agent_dir: Path)` writes `versions/router_config_<n>.json` (+ centroids `.npz`) and updates the `current` pointer atomically. Hooks into the existing `promote()` / `promote_queued()` paths so the lesson + ledger entry generation reuses what's there.
- `harness/observability/lessons.py` (verify) — confirm `Lesson(kind="router_config")` renders correctly in the Evolution feed; if the Lessons screen filters on kind enum, extend.
- `policies/policy.yaml` (extend example) — add a documented `overrides.router_config: review` entry showing operators how to opt UniRoute changes into auto vs review mode.
- `router/feedback/drift_detector.py` (P15.3.4 risk fix) — `RouterProposer` initializes `DriftDetector` baseline from the silhouette fit's intra-cluster distance, **not** from the first arbitrary `check()` call.
- `harness/proposer/router_proposer_smoke.py` — manual smoke harness (`python -m harness.proposer.router_proposer_smoke`) that runs the full pipeline against a synthetic trace set + tmp `versions/` and prints the resulting Lesson. Used for the DoD.
- Tests: `harness/proposer/tests/test_router_proposer.py`, `harness/critics/tests/test_router_critic.py`, `harness/executor/tests/test_apply_router_candidate.py`.

### Out of scope (deferred)
- `PUT /router/config` for manual `λ` overrides → **P15.3.8** (uses existing `record_manual_change()` to emit `Lesson(kind="router_config", proposal_source="human")`).
- Wiring `/run` to read `router_config_current` → **P15.3.8**.
- The trigger ("when to call `RouterProposer.propose()`") → **P15.3.9** (MCP tool + Claude Code wake-up loop).
- UI consumption of router Lessons in Evolution → already works (existing Evolution feed reads all kinds); no new UI needed in this phase.
- Online incremental Ψ updates → `router/feedback/incremental.py` stays a stub (P15.3.4 deliverable).

## Reference → target file map

| Reference | Target | Port mode |
|---|---|---|
| `training/auto_trainer.py` | **disassembled** | the orchestration logic moves into `harness/proposer/router_proposer.py`; the quality gate moves into `harness/critics/router_critic.py`; the promote step delegates to `harness/executor/promote.py`. No single target file — the reference's monolith is split to fit AHE. |
| `feedback/drift_detector.update_baseline()` | called from `RouterProposer` post-fit | reuse — sets baseline = silhouette intra-cluster distance |

## Pre-work

- Verify P15.3.6's `RouterEvaluator.evaluate()` works against a populated cache (smoke from that PLAN's T9).
- Verify P15.3.5's `LLMJudge` smoke ran successfully — preference pairs in `evals/preference_pairs/`.
- No new pyproject deps.

## Tasks (atomic, ordered)

### T1 — Extend `kind_from_mutations` for `router_config`
Edit `harness/types.py:kind_from_mutations`:

```python
def kind_from_mutations(mutations: list[str]) -> str:
    files = {m.split(":")[0] for m in mutations}
    if any("versions/router_config" in f for f in files):
        return "router_config"          # NEW — UniRoute config
    if any("retrieve" in f for f in files):
        return "rag"
    if any("rerank" in f for f in files):
        return "rerank"
    if any("route" in f for f in files):
        return "router"                 # legacy — pipeline/route.yaml knobs
    ...
```

The check for `versions/router_config` must come **before** the legacy `"route"` check; otherwise the `route` substring in `router_config` would trigger the legacy branch.

Add a unit test in `harness/tests/test_kind_from_mutations.py`:
- `versions/router_config_v3.json` → `"router_config"`.
- `pipeline/route.yaml` → `"router"` (legacy unchanged).
- `agent/prompts/system.md` → `"prompt"` (sanity).

### T2 — `RouterProposer`
Create `harness/proposer/router_proposer.py`:

```python
from harness.types import Proposal, Prediction, Mutation
from router.feedback.store_adapter import iter_traces_since
from router.feedback.trace_to_training import TraceToTraining
from router.feedback.drift_detector import DriftDetector
from router.augmentation.judge import LLMJudge
from router.augmentation.goldens import GoldenAugmenter
from router.training.kmeans import KMeansTrainer
from router.training.gate import check_first_fit_eligibility
from router.config_io import load_current_config, get_current_version
from router.evaluation.cache import ResponseCache
from .router_psi_compute import compute_blended_psi


@dataclass
class RouterProposerConfig:
    min_corpus_size: int = 200
    target_k: Optional[int] = None         # None → auto via sqrt(N/2) heuristic
    max_augmentation_samples: int = 500
    production_alpha: float = 0.3
    judge_model: Optional[str] = None      # None → harness brain default


class RouterProposer:
    def __init__(self, *, embedder, registry, judge, cache, cfg=RouterProposerConfig()):
        self.embedder = embedder
        self.registry = registry
        self.judge = judge
        self.cache = cache
        self.cfg = cfg

    def propose(self) -> Proposal:
        """Build a router_config candidate. Returns a Proposal ready for
        the harness loop (or callable directly from P15.3.9)."""

        # 1. Pull traces since current config's created_at.
        since = self._current_config_age_iso()
        traces = list(iter_traces_since(
            since_iso=since,
            embedder=self.embedder,
            assigner=None,           # cold-start: cluster_id=-1 sentinel
        ))

        # 2. First-fit gate.
        eligible, reason = check_first_fit_eligibility(
            corpus_size=len(traces),
            min_corpus_size=self.cfg.min_corpus_size,
            requested_k=self.cfg.target_k,
        )
        if not eligible:
            raise NotEnoughDataError(reason)

        # 3. Augment goldens with judged production pairs.
        augmenter = GoldenAugmenter(
            judge=self.judge,
            max_samples=self.cfg.max_augmentation_samples,
        )
        augmented = augmenter.augment(...)
        # → preference dataset persisted to evals/preference_pairs/<run_id>.jsonl

        # 4. Fit clusters.
        prompts = [t.input_text for t in traces if t.input_text]
        k = self.cfg.target_k or _sqrt_k_heuristic(len(prompts))
        trainer = KMeansTrainer(self.embedder, num_clusters=k)
        fit_result = trainer.train(
            prompts,
            fitted_from={
                "source": "production_traces",
                "n_traces": len(traces),
                "earliest": traces[0].metadata.get("timestamp"),
                "latest": traces[-1].metadata.get("timestamp"),
            },
        )

        # 5. Recompute Psi.
        psi_table = compute_blended_psi(
            assigner=fit_result.assigner,
            registry=self.registry,
            traces=traces,
            preference_dataset=augmented.preference_dataset,
            cache=self.cache,
            production_alpha=self.cfg.production_alpha,
        )

        # 6. Build candidate payload.
        next_version = (get_current_version() or 0) + 1
        candidate_payload = {
            "version": next_version,
            "k": fit_result.k,
            "centroids": fit_result.assigner.centroids,   # serialized via T4 hook
            "model_psi": psi_table,
            "cost_weight": 0.0,                            # default; manual override via P15.3.8
            "embedder_model": self.embedder.provider.model_name,
            "embedding_dim": self.embedder.dimension,
            "fitted_from": fit_result.fitted_from,
            "created_at": fit_result.fitted_at,
            "metadata": {
                "phase": "P15.3.7",
                "silhouette": fit_result.silhouette,
                "inertia": fit_result.inertia,
                "production_alpha": self.cfg.production_alpha,
            },
        }

        # 7. Initialize drift baseline so P15.3.4's risk doesn't bite.
        # (Fix: don't let DriftDetector self-baseline on first check.)
        intra_dist = _intra_cluster_distance(fit_result)
        # Persist baseline alongside the candidate; the proposer's caller in
        # P15.3.9 picks it up and instantiates DriftDetector(baseline=intra_dist).
        candidate_payload["drift_baseline"] = intra_dist

        # 8. Build Proposal for the AHE pipeline.
        target_path = f"versions/router_config_v{next_version}.json"
        return Proposal(
            mutations=[Mutation(
                file=target_path,
                describe=f"router_config v{next_version}: K={k}, N={len(prompts)}, sil={fit_result.silhouette:.3f}",
                payload=candidate_payload,           # executor reads this
            )],
            description=f"UniRoute router_config v{next_version}",
            source="claude_code",
            metadata={
                "candidate_payload_inline": True,    # signals executor to read from mutation.payload
                "cache_hits": ...,                   # populated by compute_blended_psi
                "judge_pairs": len(augmented.preference_dataset),
            },
            prediction=Prediction(
                rubric="overall",
                expected_delta=fit_result.silhouette - 0.0,   # rough — refined by critic
                rationale=(
                    f"K={k} clusters fit on {len(prompts)} production traces "
                    f"with silhouette {fit_result.silhouette:.3f}. Expecting "
                    f"AUROC lift from 0.5 (cold-start) toward something nontrivial."
                ),
                confidence=0.4,
            ),
        )

    def _current_config_age_iso(self) -> Optional[str]:
        """Return current config's created_at ISO, or None if no current config."""
```

The function `compute_blended_psi` lives in `harness/proposer/router_psi_compute.py` (T3 below). Helpers `_sqrt_k_heuristic` and `_intra_cluster_distance` are private to the proposer module.

### T3 — `router_psi_compute.py`
Create `harness/proposer/router_psi_compute.py`:

```python
def compute_blended_psi(
    *,
    assigner,
    registry,
    traces,
    preference_dataset,
    cache,
    production_alpha=0.3,
) -> dict[str, list[float]]:
    """Compute Psi[model_id] = K-vector of expected error per cluster.

    Steps:
      1. Bench Psi: for each (cluster, model), pull cached responses for prompts
         in that cluster; aggregate is_correct → bench_psi[cluster, model] = error rate.
      2. Production Psi: TraceToTraining.compute_psi_updates(traces) groups production
         traces by (cluster, model) and computes empirical error rate.
      3. Preference Psi: PreferenceDataset weights — for each (chosen_model, rejected_model)
         pair in cluster c, push -Δ to chosen_psi[c] and +Δ to rejected_psi[c].
      4. Blend: psi[m] = (1-α) * bench_psi[m] + α * prod_psi[m] (per cluster).

    Returns dict {model_id: list_of_K_floats}. Caller serializes to JSON.
    """
```

This is the meat of the recompute step. Reference's `auto_trainer.py:_improve_profiles` and `_refine_from_preferences` are the source — port the math, not the orchestration.

### T4 — Executor extension: `apply_router_candidate`
Edit `harness/executor/promote.py` to add:

```python
def apply_router_candidate(
    candidate_payload: dict,
    *,
    versions_dir: Path = Path("versions"),
) -> tuple[Path, Path]:
    """Write router_config_<n>.json + centroids sidecar atomically; update the
    current pointer. Returns (json_path, centroids_path).

    Atomic write: stage to versions/.staging/, then os.replace to final.
    Pointer update: write versions/router_config_current.tmp, fsync, replace.
    """
```

Then the existing `promote()` flow gets a small dispatch:

```python
def promote(candidate_id, ..., kind: Optional[str] = None):
    if kind is None:
        kind = kind_from_mutations(...)
    if kind == "router_config":
        # Pull payload from the proposal's mutation.payload
        payload = _load_router_payload_for(candidate_id)
        json_path, npz_path = apply_router_candidate(payload)
        mutations_desc = [f"versions/router_config_v{payload['version']}.json"]
    else:
        # Existing behavior — copy candidate's agent_dir/ on top of live agent/
        ...
```

The existing `record_manual_change()` already passes through `kind=`; P15.3.8 will reuse it for the manual `PUT /router/config` path.

### T5 — Critic: `RouterCritic`
Create `harness/critics/router_critic.py`:

```python
from harness.critics.base import Critic, CriticStage, CriticVerdict
from router.evaluation.evaluator import RouterEvaluator
from router.evaluation.cache import ResponseCache
from router.uniroute import UniRouteRouter
from router.config_io import load_current_config, load_config_from_payload


@dataclass
class RouterQualityGate:
    min_auroc_improvement: float = 0.0
    min_win_rate: float = 0.5
    max_error_rate_increase: float = 0.05


class RouterCritic(Critic):
    name = "router_quality_gate"
    stage = CriticStage.POST          # after candidate eval
    applies_to_kinds = {"router_config"}

    def __init__(self, *, embedder, cache: ResponseCache, gate: RouterQualityGate = RouterQualityGate()):
        ...

    def evaluate(self, ctx) -> CriticVerdict:
        candidate_payload = ctx.proposal.mutations[0].payload
        candidate_router = self._build_router_from_payload(candidate_payload)

        try:
            current_assigner, current_registry, current_lambda = load_current_config()
            current_router = UniRouteRouter(self.embedder, current_assigner, current_registry, current_lambda)
        except RouterConfigNotFoundError:
            current_router = None     # cold-start

        eval_candidate = RouterEvaluator(candidate_router, self.cache).evaluate(...)
        eval_current = (
            RouterEvaluator(current_router, self.cache).evaluate(...)
            if current_router else None
        )

        delta_auroc = (
            eval_candidate.metrics.auroc - eval_current.metrics.auroc
            if eval_current else eval_candidate.metrics.auroc - 0.5
        )
        # ... compute delta_avg_error, delta_win_rate similarly

        passed = (
            delta_auroc >= self.gate.min_auroc_improvement
            and eval_candidate.metrics.win_rate >= self.gate.min_win_rate
            and (eval_current is None or
                 (eval_candidate.metrics.avg_error - eval_current.metrics.avg_error)
                 <= self.gate.max_error_rate_increase)
        )

        return CriticVerdict(
            critic=self.name,
            passed=passed,
            reason=(
                f"delta_auroc={delta_auroc:+.4f}, "
                f"win_rate={eval_candidate.metrics.win_rate:.2f}, "
                f"avg_error={eval_candidate.metrics.avg_error:.4f}"
            ),
            payload={
                "delta_auroc": delta_auroc,
                "delta_win_rate": ...,
                "delta_avg_error": ...,
                "candidate_metrics": eval_candidate.metrics.to_dict(),
                "current_metrics": eval_current.metrics.to_dict() if eval_current else None,
            },
        )
```

Cold-start case: when no current config exists, the critic compares candidate against the implicit AUROC=0.5 baseline — a candidate that's no better than random gets rejected.

### T6 — Approver wiring (zero code change, one yaml line)
The existing `harness/approver/policy.py:decide()` already reads `policy.overrides[kind]`. Document in `policies/README.md` (or wherever the policy schema lives):

```yaml
mode: review
overrides:
  router_config: review     # require human review for router config changes
  # router_config: auto     # opt into auto-promotion when delta_auroc > min_auroc_improvement
```

T8 includes a test that the approver picks up `overrides["router_config"]` correctly.

### T7 — Lesson + ledger emit
Verify `harness/executor/promote.build_lesson()` already supports `kind="router_config"`. From the file inspection: it does — `kind` is a free-form string field on `Lesson`.

Add a `voice` template in `harness/executor/voices.yaml`:

```yaml
router_config:
  promoted: "I refit my routing — clusters look better and the eval lifted by Δ {delta_auroc:+.3f}."
  rejected: "I tried a refit but it didn't beat my current setup, so I kept what I had."
  human_promoted: "I tweaked my router weights manually."
  human_rejected: "I rejected my own tweak — wasn't an improvement."
```

The voice generator already routes by kind; this fills the new bucket.

### T8 — Drift baseline propagation (P15.3.4 risk fix)
The proposer stamps `drift_baseline` into `candidate_payload`. The executor's `apply_router_candidate` writes it into the persisted `router_config_<n>.json`. P15.3.9's `router_health_check` reads it back to instantiate `DriftDetector(baseline=...)` correctly — never `baseline=None`.

Add a unit test: synthesize a fit, propose, promote, then load the persisted config and confirm `drift_baseline` is a finite float matching the silhouette's intra-cluster distance.

### T9 — End-to-end smoke
Create `harness/proposer/router_proposer_smoke.py`:

```python
def main():
    """Synthetic end-to-end:
      1. Seed traces/raw/<today>.jsonl with 250 fake rows across 2 models.
      2. Seed evals/_response_cache/cache.jsonl with labeled responses.
      3. Instantiate RouterProposer with a FakeJudge + small embedder.
      4. proposal = proposer.propose()
      5. critic = RouterCritic(...); verdict = critic.evaluate(ctx_for(proposal))
      6. assert verdict.passed in (True, False)   # honest pass/fail
      7. promote(proposal, verdict)
      8. assert versions/router_config_v1.json exists
      9. assert ledger has Lesson(kind="router_config", proposal_source="claude_code")
    """
```

This satisfies the roadmap DoD ("End-to-end: trigger `RouterProposer.propose()` on a synthetic trace set → critic runs → approver decides → ledger gets a `Lesson(kind="router")` entry → `versions/router_config_v1.json` exists. Visible in Evolution timeline.")

### T10 — Tests

`harness/proposer/tests/test_router_proposer.py`:
- `test_first_fit_gate_blocks_under_min` — 100 synthetic traces → `NotEnoughDataError`.
- `test_propose_emits_proposal_with_router_config_kind` — `kind_from_mutations(p.mutations) == "router_config"`.
- `test_propose_initializes_drift_baseline` — payload's `drift_baseline` is a finite float.
- `test_psi_blend_respects_alpha` — α=0 returns bench-only Ψ; α=1 returns production-only.

`harness/critics/tests/test_router_critic.py`:
- `test_critic_passes_on_clear_win` — synthesize candidate with AUROC=0.8 vs current 0.6 → `passed=True`.
- `test_critic_fails_on_regression` — candidate AUROC=0.55 vs current 0.7 → `passed=False`.
- `test_critic_cold_start_compares_against_05` — current=None, candidate AUROC=0.7 → passes.
- `test_critic_quality_gate_uses_locked_thresholds` — verify gate values match the locked roadmap defaults.

`harness/executor/tests/test_apply_router_candidate.py`:
- `test_apply_writes_atomically` — write → fsync → pointer update; verify intermediate state never visible.
- `test_apply_round_trip_preserves_centroids` — write payload → load_current_config() → centroids byte-equal to input.
- `test_apply_appends_to_ledger` — after `promote()` with kind=router_config, ledger has the entry.

`harness/tests/test_kind_from_mutations.py`:
- See T1.

`harness/tests/test_policy_router_overrides.py`:
- `test_policy_overrides_router_config` — `Policy(overrides={"router_config": "auto"})` → `decide(outcome)` returns `AUTO_APPROVE` when critic passed.

### T11 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
python -m pytest harness/proposer/tests harness/critics/tests harness/executor/tests harness/tests -v
python -m pytest router/tests/ -v   # P15.3.1–6 still green
python -m harness.proposer.router_proposer_smoke   # end-to-end smoke
ls versions/router_config_v1.json
```

## Acceptance criteria (DoD)

1. **End-to-end smoke passes:** `python -m harness.proposer.router_proposer_smoke` runs cleanly, produces `versions/router_config_v1.json`, and emits a `Lesson(kind="router_config", proposal_source="claude_code")` visible via `/v1/lessons` (and therefore the Evolution timeline).
2. `RouterProposer.propose()` raises `NotEnoughDataError` when corpus < `min_corpus_size`.
3. `RouterCritic.evaluate(ctx)` returns `CriticVerdict.passed=True` for a clear-win candidate, `False` for a regression. Cold-start (no current config) uses AUROC=0.5 as the implicit baseline.
4. `kind_from_mutations(["versions/router_config_v3.json"])` returns `"router_config"`. Legacy `kind_from_mutations(["pipeline/route.yaml"])` still returns `"router"`.
5. `policies/policy.yaml` with `overrides.router_config: review` makes the approver queue router promotions for human review even at global `mode: auto`. With `overrides.router_config: auto` and the critic passing, the approver auto-promotes.
6. `apply_router_candidate(payload)` writes `versions/router_config_<n>.json` atomically; the `current` pointer is updated after the file is durable on disk.
7. The persisted `router_config_<n>.json` round-trips: `load_current_config()` returns an assigner whose centroids are byte-equal to the proposer's input.
8. The persisted config carries `drift_baseline` set from the silhouette's intra-cluster distance — P15.3.9 will read it without re-baselining on first `check()` call.
9. Existing critics (`ScopeCritic`, `EvalLiftCritic`, `NoCriticalRegression`) **do not** apply to `kind="router_config"` proposals (or pass trivially) — verify by running the smoke and inspecting verdicts.
10. Voice templates in `voices.yaml` produce non-empty first-person narration for the new lesson kind.
11. No regressions: full `python -m pytest` green.

## Risks / open questions

- **`kind="router"` collision.** The existing `kind_from_mutations` returns `"router"` for `pipeline/route.yaml`. We rename our new kind to `"router_config"` to disambiguate. The roadmap originally said `"router"`; this PLAN supersedes it. Update `ROADMAP_P15.3.md` Lesson kind references when this lands.
- **Candidate payload as inline `Mutation.payload`.** Existing AHE proposers branch by copying `agent_dir/` to a candidate dir. We don't branch a directory — we hold the candidate `router_config_<n>.json` payload inline on the `Mutation`. The executor's dispatch in T4 reads it from there. This is a small departure from the existing pattern. Acceptable because the artifact is one JSON file, not a directory tree. Document in T2.
- **Cache cold-start automation.** `RouterCritic` reads from `ResponseCache`. If a model in the candidate's registry has zero cache rows, the evaluator's `CacheGapError` aborts the critic. Mitigation: `RouterProposer.propose()` calls `cache_populator.harvest_from_reports()` first, then if gaps remain calls `tools/populate_response_cache.py` for the missing models in-process. Adds latency (one-shot replay) but keeps the loop autonomous. T2 documents this with a `_ensure_cache_coverage()` step.
- **Judge cost in proposer path.** `GoldenAugmenter.augment(max_samples=500)` plus eval cache replay can cost $1-3 per propose. Acceptable for a Claude-Code-triggered cycle (P15.3.9 caps cadence). Loud log line at INFO with cost estimate so the operator can watch.
- **Approver per-kind overrides — back-compat.** Existing operators have `policy.yaml` files in production that don't mention `router_config`. They get the global `mode` by default — fine. T6 documents how to opt in.
- **Synthetic smoke vs real brain.** `router_proposer_smoke.py` uses a `FakeJudge` so it runs without a brain. Real brain integration is exercised by P15.3.5's smoke (which T9 of P15.3.5 already covers). The DoD here uses synthetic to avoid a circular dep.
- **Manual-edit pipeline alignment is P15.3.8's job.** This phase makes auto-promotion AHE-conformant. Manual `PUT /router/config` path lands in P15.3.8 by calling `record_manual_change(apply_edit=write_router_payload, kind="router_config", source="human", ...)`. The existing function already takes the right shape — no new executor code needed in P15.3.8 for the manual path.
- **Drift baseline persistence.** Stored on the config artifact (not in a separate file) so rollback semantics are clean: when you roll back to v3, you also restore v3's drift baseline. T8 verifies this round-trips.
