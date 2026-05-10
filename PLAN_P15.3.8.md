# PLAN — P15.3.8 · Engine wiring + RoutingDecision in trace

| Field | Value |
|---|---|
| Phase | P15.3.8 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | P15.3.2 (uniroute decide), P15.3.7 (router_config_current promotion) |
| Unblocks | P15.3.10 (UI consumes routing_decision from traces) |
| Reference | none — this phase is integration-only |

## Goal

Make `/run` actually consume `router_config_current` and surface the
`RoutingDecision` in every trace so the UI's Trace drawer can render
"why this model". Extend `PUT /v1/router/config` for manual `λ` overrides
**through the AHE pipeline** (`record_manual_change()`), not write-through.

This phase is **pure integration** — no new concepts. It pulls together
P15.3.2 (the decide function) + P15.3.7 (the promoted config artifact) +
the existing pipeline executor + the existing manual-edit machinery.

## AHE alignment

The roadmap originally said `PUT /router/config` would be **write-through**.
That violates AutoHarness — manual operator edits must travel the same
versioned pipeline as auto edits. **This PLAN supersedes the roadmap on
that point.** `PUT` calls `harness.executor.promote.record_manual_change()`
which:

- snapshots the current router_config first
- bumps the version
- writes the new config (with the λ override applied)
- emits `Lesson(kind="router_config", proposal_source="human")`
- updates the ledger

Result: a manual λ tweak shows up in Evolution next to auto retrains, and
rolls back via `/v1/versions/{v}/rollback` like any other change.

## Scope

### In scope
- `techniques/routing/impl.py` (extend) — new variant `uniroute` alongside the existing `small_first`. Reads `router_config_current`, calls `UniRouteRouter.route(prompt)`, sets `ctx.routing.model` to the selected model and `ctx.routing.decision` to the full `RoutingDecision` dict.
- `runtime/executor/pipeline.py` (extend) — `StageRecord` gets an optional `routing_decision: dict | None` field; populated when the route stage runs the `uniroute` variant.
- `runtime/store/traces.py` (verify) — JSONL writer already serializes `StageRecord` via dataclass `asdict`; the new field flows through automatically. Add a regression test.
- `runtime/server.py` (extend) — `PUT /router/config` endpoint: accepts `{cost_weight: float}` body, calls `record_manual_change(apply_edit=_apply_lambda_override, kind="router_config", source="human", ...)`. Returns the new version number + the resulting `RouterConfigView`.
- `runtime/server.py` (extend response model) — `RunResponse.stages` includes `routing_decision` per stage so the UI Trace drawer can render it without a second round-trip.
- `agent/agent.yaml` documentation — note that `routing.variant: uniroute` opts the pipeline into UniRoute. `small_first` stays the default for backward compatibility.
- `runtime/embedder_pool.py` — lazy module-level `PromptEmbedder` singleton (one instance per process). Warmed during agent compile so first `/run` doesn't eat the model-load latency.
- `backend/channels/router/config.ts` (extend) — proxy `PUT` alongside the `GET` from P15.3.2.
- Tests: integration `tests/test_uniroute_pipeline.py` (full /run path), unit `tests/test_put_router_config_goes_through_pipeline.py`.

### Out of scope (deferred)
- UI rendering of `routing_decision` in Trace drawer → **P15.3.10** (with the rest of the Router config screen port).
- Per-tenant / per-channel routing → single global config for v1.
- Streaming the decision over SSE → existing `/v1/traces/stream` doesn't carry per-stage detail; not P15.3.8's job.
- Cost-aware λ auto-tuning → λ stays operator-controlled; the harness retrain owns clusters + Ψ but not λ. Auto-tuning λ is a future phase.
- Live config hot-reload → on `PUT`, the next `/run` reads the new config (cache flushed). No mid-request hot-swap.

## File map

| Existing | Edits |
|---|---|
| `techniques/routing/impl.py` | add `UniRouteVariant` class alongside existing variants |
| `runtime/executor/pipeline.py` | add `routing_decision` field to `StageRecord` |
| `runtime/types.py` | extend `StageOutcome` (already exposed via `RunResponse.stages`) |
| `runtime/server.py` | add `PUT /router/config`, extend `RunResponse` shape |
| `agent/pipeline/route.yaml` | document opt-in to `uniroute` variant; keep `small_first` default |
| `backend/channels/router/config.ts` | add PUT handler |

| New |
|---|
| `runtime/embedder_pool.py` |

## Pre-work

- P15.3.2 + P15.3.7 must be merged. `versions/router_config_v1.json` exists from P15.3.7's smoke (or re-run it before integration tests).
- Verify `record_manual_change()` already accepts `kind="router_config"` cleanly — checked in P15.3.7 PLAN; revisit during T7.

## Tasks (atomic, ordered)

### T1 — `UniRouteVariant` in `techniques/routing/impl.py`
Add a new variant class alongside the existing `SmallFirstVariant`:

```python
class UniRouteVariant:
    """Route via UniRoute (P15.3). Falls back to default model on cold-start."""

    def __init__(self, *, default_model: str, embedder_pool):
        self.default_model = default_model
        self.embedder_pool = embedder_pool

    def execute(self, ctx: Context) -> Context:
        from router.config_io import load_current_config
        from router.errors import RouterConfigNotFoundError, RouterColdStartError
        from router.uniroute import UniRouteRouter

        try:
            assigner, registry, lam = load_current_config()
        except RouterConfigNotFoundError:
            ctx.routing = RoutingState(
                model=self.default_model,
                decision={
                    "selected_model": self.default_model,
                    "fallback_reason": "router_not_initialized",
                    "cold_start": True,
                },
            )
            return ctx

        embedder = self.embedder_pool.get()
        router = UniRouteRouter(embedder, assigner, registry, cost_weight=lam)
        try:
            decision = router.route(ctx.request)
        except RouterColdStartError as e:
            ctx.routing = RoutingState(
                model=self.default_model,
                decision={
                    "selected_model": self.default_model,
                    "fallback_reason": f"cold_start: {e}",
                    "cold_start": True,
                },
            )
            return ctx

        ctx.routing = RoutingState(
            model=decision.selected_model,
            decision={
                "selected_model": decision.selected_model,
                "expected_error": decision.expected_error,
                "cost_adjusted_score": decision.cost_adjusted_score,
                "all_scores": decision.all_scores,
                "cluster_id": decision.cluster_id,
                "cluster_probabilities": decision.cluster_probabilities.tolist(),
                "reasoning": decision.reasoning,
                "cold_start": False,
            },
        )
        return ctx
```

Register the variant in the routing factory so `route.yaml` can select `variant: uniroute`.

### T2 — `embedder_pool.py`
Create `runtime/embedder_pool.py`:

```python
"""Process-singleton PromptEmbedder. Lazy init; warm on agent compile.

Reasons:
- Sentence-transformers model takes ~1-3s to load. We don't want every
  /run to pay that.
- Multiple stages might want embeddings later; one cache is enough.
"""

import threading
from typing import Optional


class EmbedderPool:
    def __init__(self):
        self._embedder: Optional[PromptEmbedder] = None
        self._lock = threading.Lock()

    def get(self) -> "PromptEmbedder":
        if self._embedder is not None:
            return self._embedder
        with self._lock:
            if self._embedder is None:
                from router.core.embeddings import (
                    PromptEmbedder,
                    SentenceTransformerProvider,
                )
                provider = SentenceTransformerProvider()
                self._embedder = PromptEmbedder(provider)
        return self._embedder

    def warm(self) -> None:
        """Pre-load the embedder model so the first /run is fast."""
        _ = self.get()
        # Trigger one no-op embedding to ensure tokenizer + weights are paged in
        _ = self._embedder.embed("warmup")


_pool: Optional[EmbedderPool] = None


def get_pool() -> EmbedderPool:
    global _pool
    if _pool is None:
        _pool = EmbedderPool()
    return _pool
```

Hook `EmbedderPool.warm()` into the agent compile path in `runtime/compiler/builder.py` (only when `route.variant == "uniroute"` so users on `small_first` don't pay the warmup cost).

### T3 — Extend `StageRecord` + `Context.routing`
Edit `runtime/executor/pipeline.py`:

```python
@dataclass
class StageRecord:
    stage: str
    technique: str
    variant: str
    duration_ms: float
    docs_in: int
    docs_out: int
    response_set: bool
    routing_model: Optional[str] = None
    routing_decision: Optional[dict] = None    # NEW — populated by uniroute variant
    error: Optional[str] = None
```

Edit `runtime/protocols.py` (or wherever `RoutingState` lives):

```python
@dataclass
class RoutingState:
    model: str
    decision: Optional[dict] = None    # the dict from UniRouteVariant.execute
```

In `PipelineExecutor.run`, when assembling each `StageRecord`:

```python
records.append(StageRecord(
    ...,
    routing_model=ctx.routing.model if ctx.routing else None,
    routing_decision=(ctx.routing.decision if ctx.routing else None),
    error=err,
))
```

The trace JSONL writer in `runtime/store/traces.py` uses `dataclasses.asdict`, so the new field flows through automatically. Verify in T8.

### T4 — Extend `StageOutcome` Pydantic + `RunResponse`
Edit `runtime/types.py`:

```python
class StageOutcome(BaseModel):
    stage: str
    technique: str
    variant: str
    duration_ms: float
    docs_in: int
    docs_out: int
    routing_model: Optional[str] = None
    routing_decision: Optional[dict] = None    # NEW
    error: Optional[str] = None
```

The `/run` handler in `runtime/server.py` already passes through `s.routing_model`; add `routing_decision=s.routing_decision`.

### T5 — `route.yaml` opt-in
Document in `agent/pipeline/route.yaml`:

```yaml
# variant choices: small_first (default), uniroute (P15.3)
variant: small_first   # change to "uniroute" to enable the trained router
knobs:
  small: claude-haiku-4-5
  big: claude-sonnet-4-5
  confidence_threshold: 0.7
  # uniroute reads versions/router_config_current; cold-start falls back
  # to knobs.small (or knobs.default if defined).
```

Existing deployments using `small_first` are untouched. New `uniroute` adopters change one line.

### T6 — `PUT /router/config` via `record_manual_change`
Edit `runtime/server.py`:

```python
class RouterConfigUpdate(BaseModel):
    cost_weight: Optional[float] = None
    # future: allowed_models override, etc.

class RouterConfigUpdateResponse(BaseModel):
    version: int
    config: RouterConfigView


@app.put("/router/config", response_model=RouterConfigUpdateResponse)
async def put_router_config(req: RouterConfigUpdate) -> RouterConfigUpdateResponse:
    from router.config_io import load_current_config_payload, save_config_payload
    from harness.executor.promote import record_manual_change

    try:
        current_payload = load_current_config_payload()
    except RouterConfigNotFoundError:
        raise HTTPException(
            status_code=409,
            detail="cannot edit a config that doesn't exist yet — wait for the harness to fit one",
        )

    new_payload = dict(current_payload)
    if req.cost_weight is not None:
        new_payload["cost_weight"] = req.cost_weight
        new_payload["version"] = current_payload["version"] + 1
        new_payload["metadata"] = {
            **current_payload.get("metadata", {}),
            "previous_cost_weight": current_payload["cost_weight"],
            "manual_edit_phase": "P15.3.8",
        }

    summary = (
        f"manual lambda override: {current_payload['cost_weight']} -> {req.cost_weight}"
        if req.cost_weight is not None
        else "manual router_config edit (no fields changed)"
    )

    def apply_edit():
        save_config_payload(new_payload)   # atomic write + pointer flip

    lesson = record_manual_change(
        apply_edit=apply_edit,
        kind="router_config",
        summary=summary,
        mutations_desc=[f"versions/router_config_v{new_payload['version']}.json"],
        voice=f"I tweaked my routing weights manually — λ is now {req.cost_weight}.",
    )

    return RouterConfigUpdateResponse(
        version=new_payload["version"],
        config=_build_router_config_view(new_payload),
    )
```

Key points:
- `record_manual_change()` already snapshots, bumps version, writes ledger, emits Lesson with `proposal_source="human"`. We pass `apply_edit` as the callback that does the actual file write.
- The `apply_edit` body uses `save_config_payload()` (sibling to `load_current_config_payload()` from P15.3.2). Atomic write + pointer flip — same code path the auto-promote in P15.3.7 uses.
- 409 if no current config exists — operators can't manually edit before the first auto-fit.
- Lesson kind = `"router_config"` matches the auto-promote kind from P15.3.7. Evolution timeline shows manual λ overrides next to auto retrains.

### T7 — Backend proxy: PUT
Edit `backend/channels/router/config.ts` (created in P15.3.2):

```typescript
// existing GET stays; add PUT
router.put('/v1/router/config', authMiddleware, async (c) => {
  const body = await c.req.json();
  const r = await fetch(`${RUNTIME_URL}/router/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return c.body(await r.text(), r.status, { 'Content-Type': r.headers.get('content-type') ?? 'application/json' });
});
```

### T8 — Tests

`tests/test_uniroute_pipeline.py` (integration, end-to-end):
- `test_run_with_uniroute_cold_start` — no `versions/router_config_current` exists; route variant=`uniroute`; `/run` succeeds; trace's route stage has `routing_decision.fallback_reason == "router_not_initialized"`; `routing_model == knobs.small`.
- `test_run_with_uniroute_fitted_config` — synthesize `router_config_v1.json` + pointer; `/run` succeeds; `routing_decision.cluster_id` is a non-negative int; `routing_decision.selected_model` is one of the registered models; trace JSONL on disk includes the `routing_decision` field.
- `test_run_with_small_first_unchanged` — variant=`small_first`; trace `routing_decision` is `None` (not the new path); routing_model unchanged from existing behavior.

`tests/test_put_router_config_goes_through_pipeline.py`:
- `test_put_creates_lesson_with_proposal_source_human` — `PUT /router/config {"cost_weight": 0.5}` → response 200; `/v1/lessons` lists a new lesson with `kind="router_config"`, `proposal_source="human"`.
- `test_put_bumps_version` — version goes from N to N+1; `versions/router_config_vN+1.json` exists; current pointer flipped.
- `test_put_409_when_cold_start` — no config exists; PUT returns 409 with explanatory detail.
- `test_put_rolled_back_via_versions_endpoint` — after PUT, call `POST /v1/versions/{n-1}/rollback`; current pointer goes back to v{n-1}; subsequent `/run` uses the rolled-back λ.

`tests/test_embedder_pool.py`:
- `test_pool_singleton` — two `get_pool()` calls return the same instance.
- `test_warm_eats_first_load` — `warm()` then `get().embed("x")` is fast (no cold-load).

### T9 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
python -m pytest tests/test_uniroute_pipeline.py tests/test_put_router_config_goes_through_pipeline.py tests/test_embedder_pool.py -v
python -m pytest -v   # full suite

# Manual smoke (with backend + UI dev servers up):
curl -X PUT -H "Authorization: Bearer dev" -H "Content-Type: application/json" \
  -d '{"cost_weight": 0.5}' http://localhost:8002/v1/router/config | jq
# → returns the new version + RouterConfigView
# Open Evolution screen → confirm a Lesson(kind=router_config, proposal_source=human) appears
# Click into the lesson → voice should narrate the manual edit in first person

curl -X POST -H "Authorization: Bearer dev" -H "Content-Type: application/json" \
  -d '{"request": "What is your refund policy?"}' http://localhost:8002/v1/run | jq '.stages[].routing_decision'
# → with uniroute variant + fitted config: full RoutingDecision dict
# → with cold-start: {"fallback_reason": "router_not_initialized", "cold_start": true}
```

## Acceptance criteria (DoD)

1. **`/run` with `route.variant: uniroute` succeeds** in both cold-start and fitted-config states. Cold-start path stamps `fallback_reason` and uses `agent.models.default` (or `knobs.small`).
2. **Trace's route stage carries `routing_decision`**: with a fitted config, the dict has `selected_model`, `expected_error`, `cluster_id`, `all_scores`, `reasoning`, `cold_start: false`. Field is absent (None) when variant is `small_first` or no router_config exists.
3. **`PUT /v1/router/config` with `{"cost_weight": x}`** emits a `Lesson(kind="router_config", proposal_source="human")` visible in `/v1/lessons` and the Evolution timeline.
4. **`PUT /v1/router/config` bumps version** — `versions/router_config_v(N+1).json` exists; current pointer flipped; previous version still on disk for rollback.
5. **`POST /v1/versions/{n-1}/rollback`** restores the prior router_config and the next `/run` uses its λ.
6. **Cold-start PUT returns 409** with a clear "fit a config first" detail.
7. **`small_first` variant unchanged** — existing deployments don't see new behavior; `routing_decision` is `None` on those traces.
8. **Embedder warmup happens on agent compile** when variant is `uniroute`. First `/run` doesn't pay the model-load cost. Verified by timing the first request after agent compile.
9. **Backend `PUT /v1/router/config` proxies cleanly** with the same auth as other endpoints.
10. No regressions: full `python -m pytest` green; existing `/run` tests continue to pass.

## Risks / open questions

- **Embedder load on first agent compile.** ~1-3s extra startup. Acceptable for a long-running runtime. If it ever matters, gate the warmup behind an env var (`OPENTRACY_PRELOAD_EMBEDDER=1`).
- **`Context.routing` schema drift.** Existing `Context.routing` may already have a fixed shape. T3 extends it; verify no other stage reads `ctx.routing.decision` and breaks. Default to `None` so non-uniroute paths don't see the new field.
- **`record_manual_change()` writes to `agent_dir`.** Existing implementation snapshots `agent_dir` (where prompts live). For router_config we don't want to copy the whole `agent/` tree on each manual λ edit — that bloats `versions/`. Mitigation: pass `agent_dir=Path("/tmp/null")` or extend `record_manual_change` to skip the agent-dir snapshot when `kind == "router_config"`. Verify during T6; if the function doesn't accept that, add a small `_router_config_specific_record_manual_change()` helper that mirrors the function's body minus the agent-dir snapshot.
- **Pointer flip atomicity under concurrent reads.** If `/run` reads the pointer while `PUT` is mid-flip, we get a race. Mitigation: pointer flip via `os.replace` is atomic on POSIX; `/run`'s read happens once per request and tolerates either pre- or post-flip state. Document in T6.
- **`routing_decision` size in trace JSONL.** With K=32 and 8 models, `all_scores` + `cluster_probabilities` adds ~600 bytes per trace. Across millions of traces this matters for storage. Compact representation (round to 4 decimal places) keeps it sane. Decide during T3.
- **UI consumes `routing_decision` in P15.3.10.** This phase exposes the field; rendering is later. The decision lives in the trace and the existing `/v1/traces/{id}` returns the full StageRecord, so the UI can already see it via the JSON debug view in DevTools — proper rendering lands in P15.3.10.
- **AHE alignment of PUT correctness.** This PLAN supersedes the roadmap's "write-through" wording. ROADMAP_P15.3.md should be patched to reflect that — small follow-up edit during T6.
