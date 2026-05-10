# PLAN — P15.3.1 · Core + models scaffolding

| Field | Value |
|---|---|
| Phase | P15.3.1 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | — (foundation phase) |
| Unblocks | P15.3.2, P15.3.3, P15.3.4, P15.3.6 |
| Reference | `/Users/diogovieira/Developer/open_project/OpenTracy/opentracy/{core,models}` |

## Goal

Land the math primitives and the `LLMProfile` / `LLMRegistry` data structures
so every later sub-phase can `from router.core ...` and
`from router.models ...` without runtime errors.

This phase ships **no behavior end-to-end**. It is a prerequisite library
phase: imports work, save/load round-trips work, and a smoke test suite
passes. Routing only happens in P15.3.2; training only happens in P15.3.3.

## Scope

### In scope
- `router/__init__.py` — package marker.
- `router/core/embeddings.py` — `PromptEmbedder` + `SentenceTransformerProvider` (local CPU, `all-MiniLM-L6-v2`).
- `router/core/clustering.py` — `ClusterResult` + abstract `ClusterAssigner` + `KMeansClusterAssigner`.
- `router/core/metrics.py` — metric primitives (cross-entropy, MSE, AUROC helpers).
- `router/models/llm_profile.py` — `LLMProfile` (Ψ container + save/load).
- `router/models/llm_registry.py` — `LLMRegistry` (register / get / filter / save / load).
- `router/models/llm_client.py` — `LLMResponse` + abstract `LLMClient` + concrete `AnthropicClient`.
- `versions/router_config_v0.json` — canonical schema document for the artifact.
- `router/tests/test_core.py`, `router/tests/test_profile.py` — smoke tests.
- `pyproject.toml` — new deps: `numpy`, `scikit-learn`, `sentence-transformers` (under a `[router]` extra to avoid forcing torch on users who don't need routing).

### Out of scope (deferred to later sub-phases)
- `OpenAIEmbeddingProvider` (skip — local-only for v1).
- `LearnedMapClusterAssigner` (skip — KMeans is the only cluster method needed for the first router_config).
- `OpenAIClient`, `MistralClient` (stub with `NotImplementedError`; land when first non-Anthropic routing target ships).
- Any router decision logic (P15.3.2).
- Any training / fitting (P15.3.3).
- Any wiring into `runtime/engine.py` (P15.3.8).

## Reference → target file map

| Reference (`opentracy/...`) | Target (`opentracy_new_mode/...`) | Port mode |
|---|---|---|
| `core/embeddings.py` | `router/core/embeddings.py` | partial — keep `EmbeddingProvider` Protocol + `PromptEmbedder` + `SentenceTransformerProvider`; **drop** `OpenAIEmbeddingProvider` |
| `core/clustering.py` | `router/core/clustering.py` | partial — keep `ClusterResult` + abstract `ClusterAssigner` + `KMeansClusterAssigner`; **drop** `LearnedMapClusterAssigner` |
| `core/metrics.py` | `router/core/metrics.py` | verbatim |
| `models/__init__.py` | `router/models/__init__.py` | verbatim (re-exports) |
| `models/llm_profile.py` | `router/models/llm_profile.py` | verbatim |
| `models/llm_registry.py` | `router/models/llm_registry.py` | verbatim |
| `models/llm_client.py` | `router/models/llm_client.py` | partial — keep `LLMResponse` + abstract `LLMClient` + `AnthropicClient`; **stub** `OpenAIClient` and `MistralClient` |

"Verbatim" means: copy the file, fix imports (`..core.x` → `router.core.x`, etc.), keep behavior bit-for-bit. No refactors.

## Pre-work — dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
router = [
  "numpy>=1.26",
  "scikit-learn>=1.5",
  "sentence-transformers>=3.0",
]
```

Notes:
- `sentence-transformers` pulls `torch` (CPU wheel ~700 MB). Acceptable for a dev install; `[router]` is opt-in so plain backend installs stay slim.
- `numpy` already lives in `[rag]`. Keep it there too — duplicate decl is fine.
- `anthropic>=0.40` already a core dep. No change needed.

Validate the install:
```
uv sync --extra router
python -c "import sentence_transformers, sklearn, numpy"
```

## Tasks (atomic, ordered)

### T1 — Add `[router]` extra to `pyproject.toml`
Edit `pyproject.toml`. Add the optional-dependencies block above. Run `uv sync --extra router`. Verify imports succeed.

### T2 — Create `router/` package
```
router/
  __init__.py
  core/__init__.py
  models/__init__.py
  tests/__init__.py
```
All four `__init__.py` files start empty.

### T3 — Port `core/metrics.py`
Copy `<REF>/core/metrics.py` → `router/core/metrics.py`. No edits beyond import path (none expected — it has no internal imports).

### T4 — Port `core/embeddings.py` (slim)
Copy `<REF>/core/embeddings.py` → `router/core/embeddings.py`. Then:
- **Delete** the `OpenAIEmbeddingProvider` class entirely. Leave a `# OpenAIEmbeddingProvider deferred — see ROADMAP_P15.3.md` comment.
- Keep `EmbeddingProvider` Protocol, `PromptEmbedder`, `SentenceTransformerProvider`.
- Default `SentenceTransformerProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")`.

### T5 — Port `core/clustering.py` (slim)
Copy `<REF>/core/clustering.py` → `router/core/clustering.py`. Then:
- **Delete** `LearnedMapClusterAssigner` entirely. Leave deferral comment referencing ROADMAP.
- Keep `ClusterResult`, abstract `ClusterAssigner`, `KMeansClusterAssigner`.

### T6 — Port `models/llm_profile.py` verbatim
Copy `<REF>/models/llm_profile.py` → `router/models/llm_profile.py`. No drops, no edits.

### T7 — Port `models/llm_registry.py` verbatim
Copy `<REF>/models/llm_registry.py` → `router/models/llm_registry.py`. Fix imports: `from .llm_profile import LLMProfile` (already correct in reference).

### T8 — Port `models/llm_client.py` (slim + stubs)
Copy `<REF>/models/llm_client.py` → `router/models/llm_client.py`. Then:
- Keep `LLMResponse`, abstract `LLMClient`, `AnthropicClient` verbatim.
- Replace bodies of `OpenAIClient.generate()` and `MistralClient.generate()` with `raise NotImplementedError("OpenAI client deferred — wire when a non-Anthropic routing target ships.")`. Keep the `__init__` and properties so the surface is honest.
- The `AnthropicClient.generate()` should mirror the SDK call style already used in `runtime/server.py` (same SDK version).

### T9 — Schema doc: `versions/router_config_v0.json`
Write a canonical empty schema:
```json
{
  "version": 0,
  "k": 0,
  "centroids": null,
  "model_psi": {},
  "cost_weight": 0.0,
  "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "min_corpus_size": 200,
  "created_at": null,
  "fitted_from": null,
  "metadata": {
    "schema_doc": "ROADMAP_P15.3.md",
    "phase": "P15.3.1",
    "note": "Day-zero placeholder. Real configs replace this when the harness fits the first one."
  }
}
```
This file is intentionally never read at runtime — runtime detects "no config" via missing `router_config_current` symlink. The file documents the on-disk shape so P15.3.3 / P15.3.7 know what to write.

### T10 — Smoke tests
Create `router/tests/test_core.py`:
- `test_embedder_dimension`: `PromptEmbedder(SentenceTransformerProvider()).dimension == 384`. Marked `@pytest.mark.slow` because of model download. Skip on CI unless `OPENTRACY_RUN_SLOW=1` is set.
- `test_embedder_caches`: same string returns same numpy array, identity-cached.
- `test_kmeans_round_trip`: fit a KMeans externally (use `sklearn.cluster.KMeans`), construct `KMeansClusterAssigner(centroids)`, save → load, compare assignments on a 50-vector batch bit-for-bit.
- `test_cluster_result_one_hot`: probabilities of shape (K,), `to_one_hot()` returns expected indicator vector.
- `test_metrics_smoke`: cross-entropy and MSE return finite floats on a 2-vector input.

Create `router/tests/test_profile.py`:
- `test_llm_profile_basic`: construct `LLMProfile(model_id="x", psi_vector=np.array([0.1, 0.2, 0.3]), cost_per_1k_tokens=0.001, num_validation_samples=100, cluster_sample_counts=np.array([30, 40, 30]))`, check `num_clusters == 3`, `get_cluster_error(0) == 0.1`.
- `test_llm_profile_save_load`: round-trip via `LLMProfile.save / load`. JSON content sanity-checked.
- `test_llm_profile_dim_mismatch`: psi_vector length != cluster_sample_counts length raises ValueError in `__post_init__`.
- `test_registry_basic`: register two profiles, `get_all()` returns 2, `__contains__` works, `get_default()` returns the one set as default.
- `test_anthropic_client_init`: `AnthropicClient(model_id="claude-haiku-4-5", api_key="dummy")` instantiates without making a network call. `cost_per_1k_tokens` returns the configured value.
- `test_openai_client_stub`: `OpenAIClient(...).generate("hi")` raises `NotImplementedError` with the deferral message.

### T11 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
uv sync --extra router
python -m pytest router/tests/ -v
```
All tests pass (slow tests skip without env flag; T11 explicitly runs them once with `OPENTRACY_RUN_SLOW=1` to confirm the model download path works).

## Acceptance criteria (DoD)

A reviewer can verify P15.3.1 ships by:

1. `uv sync --extra router` succeeds on a clean checkout.
2. `python -c "from router.core.embeddings import PromptEmbedder, SentenceTransformerProvider; from router.core.clustering import KMeansClusterAssigner, ClusterResult; from router.models.llm_profile import LLMProfile; from router.models.llm_registry import LLMRegistry; from router.models.llm_client import AnthropicClient, LLMResponse"` — no `ImportError`, no `ModuleNotFoundError`.
3. `python -m pytest router/tests/ -v` is green for fast tests.
4. `OPENTRACY_RUN_SLOW=1 python -m pytest router/tests/ -v` is green including the embedder download/dimension test.
5. `versions/router_config_v0.json` exists, parses as JSON, includes the documented keys.
6. The deferred classes (`OpenAIEmbeddingProvider`, `LearnedMapClusterAssigner`, full `OpenAIClient`, `MistralClient`) are explicitly marked deferred with a comment pointing to `ROADMAP_P15.3.md`.

No runtime endpoint changes. No UI changes. No harness wiring. P15.3.2 is the next phase that turns this library into something callable.

## Risks / open questions

- **Torch wheel size.** ~700 MB CPU torch + ~80 MB miniLM weights. First `uv sync --extra router` is slow. Acceptable trade-off for a local-only embedder. If it becomes a problem, the fallback is the deferred `OpenAIEmbeddingProvider` (paid per call, zero local install).
- **Model download on first import.** `SentenceTransformerProvider.__init__` triggers a HF download if the model isn't cached. Tests gate this behind `OPENTRACY_RUN_SLOW=1`. The `versions/router_config_v0.json` doc-config commits us to `all-MiniLM-L6-v2` — switching later means a forced full retrain, so worth re-checking the choice before P15.3.3 ships.
- **Anthropic SDK drift.** `AnthropicClient.generate()` ports verbatim from the reference, but the reference might pin an older SDK. Verify against the SDK version pinned in this project (`anthropic>=0.40`) and adapt minimally if needed (e.g., new content-block schema).
- **`LLMProfile` save format.** Reference saves with `np.save` for the array fields and JSON for metadata side-by-side. Confirm during T6 — if the format is split-files, decide whether to keep that or move to a single JSON-with-base64-arrays for ledger ergonomics. Default: keep verbatim, revisit only if P15.3.7 snapshotting hits issues.
