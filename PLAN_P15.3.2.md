# PLAN — P15.3.2 · UniRoute decision engine

| Field | Value |
|---|---|
| Phase | P15.3.2 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | P15.3.1 (core + models scaffolding) |
| Unblocks | P15.3.7 (router_proposer), P15.3.8 (engine wiring), debug surfaces in UI |
| Reference | `/Users/diogovieira/Developer/open_project/OpenTracy/opentracy/router/uniroute.py` |

## Goal

Make `router.uniroute.UniRouteRouter` a pure, callable decision function:
given a prompt + a fitted `router_config`, return
`(selected_model, expected_error, all_scores, reasoning)` **without
executing the LLM**. Expose the function via two HTTP endpoints so the UI
and debug tools can poke at it independently of `/run`.

This phase ships the **decision math + read-only HTTP surface**. It does
**not** wire routing into `/run` (that's P15.3.8) and does **not** train
or fit anything (that's P15.3.3 / P15.3.7).

## Scope

### In scope
- `router/uniroute.py` — port `UniRouteRouter` + `RoutingDecision` + `RoutingStats` verbatim from reference; adjust import paths for the new namespace.
- `router/errors.py` — `RouterColdStartError` (raised when registry empty or assigner unfitted).
- `router/config_io.py` — load/save `router_config_<n>.json` artifacts; resolve "current" via a `versions/router_config_current` symlink (or a marker file fallback if symlinks are awkward on the user's setup).
- `runtime/types.py` (extend) — Pydantic models: `RouterConfigView`, `RouterDecideRequest`, `RouterDecideResponse`.
- `runtime/server.py` (extend) — `GET /router/config`, `POST /router/decide`, plus update the docstring endpoint listing at the top of the file.
- `backend/channels/router/` (new) — `config.ts`, `decide.ts` proxy handlers; register in the main backend router.
- `router/tests/test_uniroute.py` — unit tests for the decision math with mock embedder + 2 fake profiles.
- `router/tests/test_endpoints.py` — integration tests against `POST /router/decide` and `GET /router/config` using FastAPI's `TestClient`.

### Out of scope (deferred)
- Wiring routing into `/run` execution path → **P15.3.8**.
- Cluster fitting / Ψ population → **P15.3.3**, **P15.3.5**, **P15.3.6**.
- `PUT /router/config` for manual `λ` overrides → **P15.3.8** (explicitly added in that phase, not here).
- `DELETE` / rollback endpoints → handled implicitly via the existing `/versions/{v}/rollback` machinery in P15.3.7.
- UI consumption of `/router/decide` → **P15.3.10** (UI port).

## Reference → target file map

| Reference (`opentracy/...`) | Target (`opentracy_new_mode/...`) | Port mode |
|---|---|---|
| `router/uniroute.py` | `router/uniroute.py` | verbatim — only fix relative imports (`..core.embeddings` → `router.core.embeddings`, etc.) |
| — (new) | `router/errors.py` | new file — `RouterColdStartError`, `RouterConfigNotFoundError` |
| — (new) | `router/config_io.py` | new file — `load_current_config()`, `save_config()`, `get_current_version()` |

The reference's `uniroute.py` is fully re-usable as-is. The new files
(`errors.py`, `config_io.py`) glue it to this project's `versions/` ledger
without polluting the ported file.

## Pre-work

None beyond P15.3.1 deliverables. No new pyproject deps.

Verify P15.3.1 is green before starting: `python -m pytest router/tests/`
must pass, including the slow embedder test under `OPENTRACY_RUN_SLOW=1`.

## Tasks (atomic, ordered)

### T1 — Port `uniroute.py`
Copy `<REF>/router/uniroute.py` → `router/uniroute.py`. Fix imports:
```python
from router.core.embeddings import PromptEmbedder
from router.core.clustering import ClusterAssigner, ClusterResult
from router.models.llm_profile import LLMProfile
from router.models.llm_registry import LLMRegistry
from router.models.llm_client import LLMClient, LLMResponse
```
Keep all three public classes (`RoutingDecision`, `RoutingStats`, `UniRouteRouter`) and all methods (`route`, `route_batch`, `route_and_execute`, `get_best_model_for_cluster`, `analyze_routing_distribution`). No behavioral edits.

### T2 — Define error types
Create `router/errors.py`:
```python
class RouterError(Exception): ...
class RouterColdStartError(RouterError):
    """Raised when route() is called before a router_config exists."""
class RouterConfigNotFoundError(RouterError):
    """Raised when config_io can't find a router_config artifact."""
class RouterConfigInvalidError(RouterError):
    """Raised when a router_config_<n>.json fails schema validation."""
```
Wire into `uniroute.py`: in `__init__`, raise `RouterColdStartError` if registry is empty; in `route()`, raise the same if `cluster_assigner` is unfitted (detect via `assigner.num_clusters == 0` or the equivalent guard the reference uses).

### T3 — `router/config_io.py`
Surface:
```python
def load_current_config() -> tuple[ClusterAssigner, LLMRegistry, float]:
    """Load (assigner, registry, cost_weight) from versions/router_config_current.

    Raises RouterConfigNotFoundError if no current symlink/marker exists.
    Raises RouterConfigInvalidError if the JSON fails schema validation.
    """

def save_config(version: int, assigner, registry, cost_weight, fitted_from: dict) -> Path:
    """Atomic write of versions/router_config_<version>.json + update current pointer."""

def get_current_version() -> Optional[int]:
    """Return version int from the current pointer, or None if cold-start."""
```
Implementation notes:
- Schema matches `versions/router_config_v0.json` from P15.3.1.
- `model_psi` field stores Ψ vectors as nested arrays (JSON-friendly); reconstructs `LLMProfile` via `from_dict()`.
- "Current pointer" is `versions/router_config_current` (symlink) with a fallback to `versions/router_config_current.txt` (single-line file containing the version int) if the OS rejects symlinks. Detect once at import, log which path is used.
- Saves are atomic via the `tempfile + os.replace` pattern already used elsewhere in this repo (verify by grepping `os.replace` in `harness/executor/`).

### T4 — Pydantic models in `runtime/types.py`
Add three models:

```python
class RouterConfigView(BaseModel):
    version: Optional[int]              # None if cold-start
    k: int                              # 0 if cold-start
    model_count: int
    cost_weight: float
    embedder_model: str
    embedding_dim: int
    last_fit_at: Optional[str]          # ISO 8601
    fitted_from: Optional[dict]         # {n_traces, n_goldens, ...}
    cold_start: bool                    # convenience flag

class RouterDecideRequest(BaseModel):
    prompt: str
    allowed_models: Optional[list[str]] = None
    cost_weight_override: Optional[float] = None

class RouterDecideResponse(BaseModel):
    selected_model: str
    expected_error: float
    cost_adjusted_score: float
    all_scores: dict[str, float]
    cluster_id: int
    cluster_probabilities: list[float]
    reasoning: str
    cold_start: bool                    # always false here; cold-start raises 503
```

### T5 — `GET /router/config` endpoint
Add to `runtime/server.py`:
```python
@app.get("/router/config", response_model=RouterConfigView)
async def get_router_config() -> RouterConfigView:
    try:
        assigner, registry, lam = load_current_config()
    except RouterConfigNotFoundError:
        return RouterConfigView(
            version=None, k=0, model_count=0, cost_weight=0.0,
            embedder_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384, last_fit_at=None, fitted_from=None,
            cold_start=True,
        )
    # ... happy path
```
Cold-start returns 200 with `cold_start=True`, **not** 404 — UI needs the metadata to render the empty-state correctly per P15.3.10.

### T6 — `POST /router/decide` endpoint
Add to `runtime/server.py`:
```python
@app.post("/router/decide", response_model=RouterDecideResponse)
async def post_router_decide(req: RouterDecideRequest) -> RouterDecideResponse:
    try:
        assigner, registry, lam = load_current_config()
    except RouterConfigNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="router_cold_start: no fitted config; call /router/decide after the harness fits a config",
        )
    embedder = _get_or_init_embedder()  # lazy singleton
    router = UniRouteRouter(embedder, assigner, registry, cost_weight=lam)
    decision = router.route(req.prompt, req.allowed_models, req.cost_weight_override)
    return RouterDecideResponse(
        selected_model=decision.selected_model,
        expected_error=decision.expected_error,
        cost_adjusted_score=decision.cost_adjusted_score,
        all_scores=decision.all_scores,
        cluster_id=decision.cluster_id,
        cluster_probabilities=decision.cluster_probabilities.tolist(),
        reasoning=decision.reasoning,
        cold_start=False,
    )
```
The lazy embedder singleton (`_get_or_init_embedder`) lives module-scope so the first call eats the model-load latency and subsequent calls are fast. Same pattern P15.3.8 will reuse for `/run`.

### T7 — Update `server.py` docstring
Append to the endpoint listing at the top of `runtime/server.py`:
```
  GET  /router/config                  — current router_config metadata; cold-start safe.
  POST /router/decide                  — score a prompt against the router; no LLM call.
```

### T8 — Backend proxy channel
Create `backend/channels/router/` with two files matching existing channel conventions (cf. `backend/channels/agent/`):

- `config.ts` — `GET /v1/router/config` → proxies to runtime `:8001/router/config`.
- `decide.ts` — `POST /v1/router/decide` → proxies to runtime `:8001/router/decide`.

Both inherit the existing `Authorization: Bearer dev` requirement from the backend's auth middleware.

### T9 — Register backend routes
Add the two handlers to backend's main router (likely `backend/index.ts` or `backend/routes.ts` — verify during execution by mirroring how `agent/config.ts` is registered).

### T10 — Decision-math unit tests
Create `router/tests/test_uniroute.py`:

```python
def test_route_picks_lowest_error_at_zero_lambda():
    """With cost_weight=0, picks the model with lowest expected error."""
    # 2 fake profiles, K=3
    # profile_a.psi = [0.1, 0.5, 0.3]; cost = 0.01
    # profile_b.psi = [0.5, 0.1, 0.3]; cost = 0.001
    # phi = [1, 0, 0] (hard cluster 0) → A wins (0.1 < 0.5)
    # phi = [0, 1, 0] (hard cluster 1) → B wins (0.1 < 0.5)

def test_route_factors_cost_at_high_lambda():
    """High cost_weight tips toward cheaper model even if accuracy is worse."""

def test_route_uses_soft_assignment():
    """Soft phi = [0.6, 0.4] picks the one with lower expected error under blend."""

def test_cold_start_raises():
    """Empty registry → RouterColdStartError on construction."""

def test_route_respects_allowed_models():
    """allowed_models=['b'] forces B even if A is better."""

def test_routing_stats_update():
    """Sequential decisions update RoutingStats counters."""

def test_get_best_model_for_cluster():
    """get_best_model_for_cluster(c) matches what route() picks for hard phi."""

def test_analyze_distribution_smoke():
    """analyze_routing_distribution returns dict with expected keys."""
```
Use a `FakeEmbedder` returning a hardcoded embedding and a `FakeKMeansAssigner` returning a fixed `ClusterResult` to keep tests fast. No model download.

### T11 — Endpoint integration tests
Create `router/tests/test_endpoints.py`:

```python
def test_get_router_config_cold_start(client):
    """GET /router/config returns 200 with cold_start=True when no config exists."""

def test_get_router_config_fitted(client, tmp_versions_dir):
    """GET /router/config returns full metadata when versions/ has a config."""

def test_post_router_decide_cold_start_503(client):
    """POST /router/decide returns 503 router_cold_start when no config."""

def test_post_router_decide_happy_path(client, tmp_versions_dir, monkeypatch_embedder):
    """POST /router/decide returns full RoutingDecision shape with a fitted config."""

def test_post_router_decide_rejects_unknown_model_in_allowed(client, ...):
    """POST /router/decide with allowed_models=['nonexistent'] returns 400/422."""
```
Use FastAPI `TestClient` from `runtime/server.py:app`. `tmp_versions_dir` fixture writes a synthetic `router_config_v1.json` + current pointer; `monkeypatch_embedder` swaps in a `FakeEmbedder` so tests don't download the real model.

### T12 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
uv sync --extra router
python -m pytest router/tests/ -v
# Backend smoke (manual):
cd backend && OPENTRACY_API_KEY=dev npm run start &
curl -s -H "Authorization: Bearer dev" http://localhost:8002/v1/router/config | jq
curl -s -H "Authorization: Bearer dev" -H "Content-Type: application/json" \
  -d '{"prompt": "test"}' http://localhost:8002/v1/router/decide
```

## Acceptance criteria (DoD)

1. `python -m pytest router/tests/test_uniroute.py router/tests/test_endpoints.py -v` is green (all 13+ tests pass).
2. `GET /router/config` returns `cold_start=true` on a clean checkout (no `versions/router_config_current` exists).
3. `POST /router/decide` returns 503 with `router_cold_start` detail on cold-start.
4. With a synthesized `versions/router_config_v1.json` + current pointer:
   - `GET /router/config` returns the metadata correctly (version=1, k matches, model_count matches).
   - `POST /router/decide` returns a full `RoutingDecision` shape; `selected_model` is one of the registered IDs; `cluster_probabilities` sums to ~1.0.
5. `runtime/server.py` docstring lists the two new endpoints.
6. Backend proxy works: `curl localhost:8002/v1/router/config -H "Authorization: Bearer dev"` returns the same body as the runtime port (modulo any wrapping).
7. No regressions: `python -m pytest` (full suite) is still green.
8. `from router.uniroute import UniRouteRouter, RoutingDecision` works without runtime errors.

## Risks / open questions

- **Symlink portability for `router_config_current`.** macOS + Linux dev boxes handle symlinks fine, but if the project ever ships in a Docker layer that uses overlayfs or a Windows dev container, symlinks can be unreliable. Fallback to a `.txt` marker file is implemented in T3; pick the path during T3 by probing `os.symlink` once.
- **Embedder load latency on first `/router/decide`.** First call after server start eats ~1–3s for the MiniLM load. Acceptable for a debug endpoint, but worth a warm-up hook later (P15.3.8 will need it for `/run`).
- **`RoutingDecision.to_dict()` shape vs Pydantic.** The reference's `to_dict()` outputs `cluster_probabilities` as a list (already JSON-friendly) but `all_scores` as a `dict[str, float]` — fine. Verify during T6 that nothing leaks numpy types into the Pydantic response (numpy floats break FastAPI's JSON encoder in some versions; cast at the boundary).
- **Schema drift between `router_config_v0.json` (doc) and `config_io.save_config()` writes.** P15.3.1 wrote v0 as documentation. If T3 picks a different field layout, update v0 and the schema_doc note. The two must stay in sync.
- **Backend channel naming.** `backend/channels/router/` may collide with existing routing-related code if the backend already has a `router.ts` for its own internal use. Verify during T8 by `ls backend/`; rename to `routerconfig/` if there's a collision.
