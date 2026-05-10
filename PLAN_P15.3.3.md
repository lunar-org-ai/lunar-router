# PLAN — P15.3.3 · KMeans trainer + first-fit gate

| Field | Value |
|---|---|
| Phase | P15.3.3 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | P15.3.1 (core + models scaffolding), P15.3.2 (`config_io.save_config`) |
| Unblocks | P15.3.7 (router_proposer) |
| Reference | `/Users/diogovieira/Developer/open_project/OpenTracy/opentracy/training/kmeans_trainer.py` |

## Goal

Take a corpus of prompt embeddings → fit clusters → produce a serializable
`KMeansClusterAssigner` plus a quality report (silhouette + inertia +
cluster-size distribution).

This phase ships the **clustering training loop**, the **first-fit gate**
that holds back fitting until enough data accumulates, and the **on-disk
hand-off** to the partial `router_config_<n>.json` shape from P15.3.2.

It does **not** populate Ψ (per-model error tables) — that's P15.3.6 — and
it does **not** decide *when* to fit (that's P15.3.9, the Claude Code
brain talking via MCP).

## Scope

### In scope
- `router/training/__init__.py` — package marker.
- `router/training/kmeans.py` — `KMeansTrainer` + `KMeansPlusPlusInitializer` + `analyze_clusters`, ported from reference, slimmed to operate on `list[str]` (no `PromptDataset` dep yet).
- `router/training/result.py` — `KMeansTrainResult` dataclass: `(assigner, silhouette, inertia, k, n_samples, cluster_sizes, fitted_at, embedder_model_id, fitted_from)`.
- `router/training/gate.py` — `check_first_fit_eligibility(corpus_size, min_corpus_size=200) -> tuple[bool, str]` + `NotEnoughDataError`.
- `router/errors.py` (extend from P15.3.2) — add `NotEnoughDataError`, `KMeansFitError`.
- `router/training/snapshot.py` — wrapper that takes a `KMeansTrainResult` and emits a partial `router_config_<n>.json` via `config_io.save_config()`. Ψ stays empty (`{}`) — P15.3.6 fills it later.
- `router/tests/test_kmeans.py` — unit tests covering gate, training, K-selection via silhouette, save/load round-trip, `analyze_clusters` shape.

### Out of scope (deferred)
- Ψ population per cluster × model → **P15.3.6**.
- Trace → training-set conversion → **P15.3.4** (introduces `PromptDataset`; this phase stays on flat `list[str]` to avoid a circular dep).
- The trigger that fires fitting → **P15.3.9** (`router_health_check` MCP tool + Claude Code decision).
- Refit-vs-incremental update strategy → **P15.3.4 optional / P15.3.7**.
- Embedding caching across fits — `PromptEmbedder` cache from P15.3.1 already covers this; nothing extra here.

## Reference → target file map

| Reference | Target | Port mode |
|---|---|---|
| `training/kmeans_trainer.py` (`KMeansTrainer`, `KMeansPlusPlusInitializer`, `analyze_clusters`) | `router/training/kmeans.py` | partial — port classes; **drop** `PromptDataset` argument; accept `list[str]` directly |
| — | `router/training/result.py` | new |
| — | `router/training/gate.py` | new |
| — | `router/training/snapshot.py` | new |

## Pre-work

None. P15.3.1 already added `numpy`, `scikit-learn`, `sentence-transformers`. Reuse.

Verify before starting:
- `python -m pytest router/tests/test_core.py router/tests/test_uniroute.py -v` is green.
- `from router.config_io import save_config` works (P15.3.2 deliverable).

## Tasks (atomic, ordered)

### T1 — Port `kmeans_trainer.py` (slim)
Copy `<REF>/training/kmeans_trainer.py` → `router/training/kmeans.py`. Then:

- **Drop** the `PromptDataset` import and replace its uses:
  - `train(training_set: PromptDataset)` → `train(prompts: list[str])`. Remove the `training_set.get_prompts()` call; iterate `prompts` directly.
  - `train_with_validation(training_set, validation_set)` → `train_with_validation(train_prompts: list[str], val_prompts: list[str])`.
  - `analyze_clusters(dataset)` → `analyze_clusters(prompts: list[str])`.
- Fix imports to `router.core.embeddings` / `router.core.clustering`.
- Wrap the `verbose=True print()` calls with a module-level `logger = logging.getLogger("router.training.kmeans")` and emit at `INFO`. Keep `verbose` param as a passthrough that toggles a stream handler if the caller wants raw stdout (default off).
- Keep `KMeansPlusPlusInitializer` and `analyze_clusters` verbatim (modulo the dataset → list edit).

### T2 — Add `KMeansTrainResult` dataclass
Create `router/training/result.py`:
```python
@dataclass(frozen=True)
class KMeansTrainResult:
    assigner: KMeansClusterAssigner
    k: int
    n_samples: int
    silhouette: float            # NaN if N < 2 * K (sklearn requirement)
    inertia: float
    cluster_sizes: dict[int, int]
    embedder_model_id: str
    fitted_at: str               # ISO 8601 UTC
    fitted_from: dict            # {"source": "production_traces", "n_traces": ..., "earliest": ..., "latest": ...}
                                 # OR {"source": "synthetic", "note": "..."} for test fits

    def summary(self) -> str:
        """One-line human summary used in logs + ledger entries."""
        return (
            f"K={self.k} N={self.n_samples} "
            f"silhouette={self.silhouette:.4f} inertia={self.inertia:.2f} "
            f"sizes_min={min(self.cluster_sizes.values())} "
            f"sizes_max={max(self.cluster_sizes.values())}"
        )
```

### T3 — First-fit gate + `NotEnoughDataError`
Extend `router/errors.py`:
```python
class NotEnoughDataError(RouterError):
    """Corpus too small to fit a cluster model. Wait for more traces."""

class KMeansFitError(RouterError):
    """KMeans failed to converge or produced degenerate clusters."""
```

Create `router/training/gate.py`:
```python
def check_first_fit_eligibility(
    corpus_size: int,
    min_corpus_size: int = 200,
    requested_k: Optional[int] = None,
) -> tuple[bool, str]:
    """Decide whether the corpus is large enough to fit.

    Returns (eligible, reason). If eligible=False the reason explains why
    so the Claude Code brain (P15.3.9) can surface it in its rationale.
    """
    if corpus_size < min_corpus_size:
        return False, f"corpus_size={corpus_size} < min_corpus_size={min_corpus_size}"
    if requested_k is not None:
        if corpus_size < 2 * requested_k:
            return False, f"corpus_size={corpus_size} < 2 * K={requested_k} (silhouette undefined)"
        if requested_k < 2:
            return False, f"K={requested_k} < 2"
    return True, "ok"
```

### T4 — Wrap `train()` to return `KMeansTrainResult`
Change `KMeansTrainer.train()` signature:
```python
def train(
    self,
    prompts: list[str],
    *,
    fitted_from: dict,                 # caller supplies provenance
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    silhouette_sample: int = 5000,     # cap for O(N²) silhouette calc
) -> KMeansTrainResult: ...
```

Body: compute embeddings → fit `sklearn.KMeans` → wrap centroids in `KMeansClusterAssigner` → compute silhouette on a random subsample of `silhouette_sample` (or all if `N < silhouette_sample`) → compute cluster_sizes via `analyze_clusters` → return `KMeansTrainResult`.

If KMeans fails to converge (`kmeans.n_iter_ >= max_iter` and inertia jumps), raise `KMeansFitError` with the diagnostic.

`train_with_validation()` analogously returns the best `KMeansTrainResult` (not `(assigner, k)` tuple). Internally it runs the K loop and picks the highest silhouette.

### T5 — Snapshot: emit partial `router_config_<n>.json`
Create `router/training/snapshot.py`:
```python
def snapshot_clusters_only(
    result: KMeansTrainResult,
    cost_weight: float = 0.0,
    bump_version: bool = True,
) -> Path:
    """Write a router_config artifact with centroids set, model_psi empty.

    P15.3.6 will write a sibling file router_psi_<n>.json with the Ψ tables
    and the executor in P15.3.7 stitches them into router_config_<n>.json
    under the current pointer. This snapshot is intermediate — not a valid
    'current' config until Ψ is added.
    """
```

It calls `config_io.save_config()` from P15.3.2 with:
- `centroids = result.assigner.centroids.tolist()`
- `k = result.k`
- `model_psi = {}` (empty — Ψ not computed yet)
- `cost_weight = 0.0` (ignored at this stage)
- `embedder_model = result.embedder_model_id`
- `embedding_dim = result.assigner.embedding_dim`
- `fitted_from = result.fitted_from`
- `created_at = result.fitted_at`
- `metadata = {"phase": "P15.3.3", "stage": "clusters_only", "silhouette": result.silhouette}`

The "current" pointer is **not** updated by this function. Only the executor in P15.3.7 promotes a stitched config to current.

### T6 — Logging surface
At the top of `router/training/kmeans.py`:
```python
logger = logging.getLogger("router.training.kmeans")
```
Emit at INFO:
- Start of fit: `f"fit start n={n} k={k} embedder={embedder_id}"`.
- Per-K progress in `train_with_validation`: `f"k={k} silhouette={s:.4f} inertia={i:.2f}"`.
- Final: `result.summary()`.

These show up in the runtime log + get captured by the harness ledger when P15.3.7 calls into here.

### T7 — Unit tests
Create `router/tests/test_kmeans.py`:

```python
def test_gate_blocks_under_min():
    eligible, reason = check_first_fit_eligibility(corpus_size=100, min_corpus_size=200)
    assert not eligible
    assert "100" in reason and "200" in reason

def test_gate_blocks_when_n_lt_2k():
    eligible, reason = check_first_fit_eligibility(corpus_size=200, min_corpus_size=200, requested_k=120)
    assert not eligible

def test_gate_passes_at_threshold():
    eligible, reason = check_first_fit_eligibility(corpus_size=200, min_corpus_size=200, requested_k=8)
    assert eligible

def test_train_basic_with_fake_embedder():
    # Generate 3 well-separated synthetic Gaussians in 16-dim space.
    # Use a FakeEmbedder that returns the pre-computed embedding by string lookup.
    # Train K=3 → silhouette > 0.5, cluster_sizes roughly balanced.

def test_train_round_trip_bit_for_bit():
    # Train on synthetic data → save assigner → load assigner → assign 100 vectors.
    # Assignments must match position-by-position.

def test_train_with_validation_selects_best_k():
    # 4 well-separated Gaussians → K candidates [2, 3, 4, 5, 6].
    # Best K must be 4 (highest silhouette on val set).

def test_train_logs_silhouette(caplog):
    with caplog.at_level(logging.INFO, logger="router.training.kmeans"):
        result = trainer.train(prompts, fitted_from={"source": "synthetic"})
    assert any("silhouette" in r.message for r in caplog.records)

def test_analyze_clusters_shape():
    stats = analyze_clusters(prompts, assigner, embedder, top_n=2)
    assert stats["num_clusters"] == K
    assert sum(stats["cluster_sizes"].values()) == len(prompts)
    assert all(len(v) <= 2 for v in stats["cluster_examples"].values())

def test_snapshot_writes_partial_config(tmp_path, monkeypatch):
    monkeypatch.setattr("router.config_io._VERSIONS_DIR", tmp_path)
    result = trainer.train(prompts, fitted_from={"source": "synthetic"})
    path = snapshot_clusters_only(result)
    payload = json.loads(path.read_text())
    assert payload["k"] == result.k
    assert payload["model_psi"] == {}
    assert "centroids" in payload and len(payload["centroids"]) == result.k
```

All tests use a `FakeEmbedder` that maps prompt → pre-computed numpy vector. No model download in the test path; the slow MiniLM tests stay isolated to P15.3.1's `test_core.py`.

### T8 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
python -m pytest router/tests/test_kmeans.py -v
python -m pytest router/tests/ -v   # full router test surface still green
```

## Acceptance criteria (DoD)

1. `python -m pytest router/tests/test_kmeans.py -v` is green.
2. **Round-trip is bit-for-bit:** train → save → load → assign N vectors produces position-identical `cluster_id` and `cluster_probabilities` (numpy `array_equal` on both, modulo the soft-prob tolerance the reference already accepts).
3. **Silhouette is logged on every fit** at `INFO` level on `router.training.kmeans`.
4. **First-fit gate refuses below 200**: `check_first_fit_eligibility(199, 200)` returns `(False, ...)`.
5. **First-fit gate refuses N < 2K**: `check_first_fit_eligibility(200, 200, requested_k=120)` returns `(False, ...)`.
6. **`snapshot_clusters_only()` writes a partial config** with `model_psi={}` and centroids matching `result.assigner.centroids.tolist()`. The "current" pointer is **not** updated (this is intermediate).
7. No regressions: full `python -m pytest` green.
8. `from router.training.kmeans import KMeansTrainer` and `from router.training.gate import check_first_fit_eligibility` work without runtime errors.

## Risks / open questions

- **Silhouette O(N²).** sklearn's `silhouette_score` is quadratic in N. Capped at `silhouette_sample=5000` random subsample (matches sklearn's own guidance). Above that, results are still representative; below it, exact. T4 documents the cap.
- **Bit-for-bit save/load.** The reference's `KMeansClusterAssigner.save()` uses `np.save` for centroids. Loading a `.npy` and re-asserting equality is reliable. **But** if `snapshot_clusters_only()` writes centroids as JSON arrays via `tolist()`, float64 → Python float → JSON → float64 round-trip can drop the last bit. Mitigation: snapshot writes JSON for *metadata* but stores centroids in a sibling `.npz` file referenced from JSON (`"centroids_path": "router_config_<n>_centroids.npz"`). T5 must spell this out before implementation; if the reviewer prefers all-JSON, switch to base64-encoded float64 bytes.
- **Default K.** Reference default is K=100. For day-zero corpus of 200 prompts, K=100 means ~2 prompts/cluster — degenerate. The harness brain in P15.3.9 is supposed to pick K, but T3's gate also enforces `N >= 2K`. Document a recommended K formula in the gate's docstring (`K ≈ sqrt(N/2)`, capped 4..32) but don't enforce — Claude Code reads it as a hint, not a rule.
- **Silhouette undefined for K=1.** `train_with_validation` must filter `k_values` to drop K<2; document.
- **`fitted_from` provenance.** Required arg in `train()` so the caller can't forget to record where the data came from. The proposer in P15.3.7 will pass `{source: "production_traces", n_traces: ..., earliest: iso, latest: iso}`. Tests pass `{source: "synthetic"}`. The ledger needs this to render Lessons of `kind="router"` correctly.
- **`PromptDataset` deferral.** Reference's trainer takes `PromptDataset`. We accept `list[str]`. When P15.3.4 introduces `PromptDataset`, retrofit the trainer to accept either — `list[str]` or a `PromptDataset` whose `.get_prompts() -> list[str]`. Don't introduce that polymorphism here; P15.3.4 owns it.
