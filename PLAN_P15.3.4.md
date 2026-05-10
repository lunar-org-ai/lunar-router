# PLAN — P15.3.4 · trace → training + drift detector

| Field | Value |
|---|---|
| Phase | P15.3.4 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | P15.3.1 (core + models scaffolding) |
| Unblocks | P15.3.5 (judge), P15.3.7 (router_proposer), P15.3.9 (MCP `router_health_check`) |
| Reference | `/Users/diogovieira/Developer/open_project/OpenTracy/opentracy/{data,feedback}` |

## Goal

Make the autonomy loop **legible**: production traces become a structured
training signal (per-cluster, per-model error rates), and shifts in input
distribution become a quantified drift score. Both surface as inputs the
Claude Code brain in P15.3.9 reads via MCP to decide if a retrain is
worth proposing.

This phase ships **read paths only**: it converts existing traces and
embeddings into typed reports. It does **not** trigger fits (P15.3.9), it
does **not** run the proposer pipeline (P15.3.7), and it does **not** judge
or augment data (P15.3.5).

## Scope

### In scope
- `router/data/__init__.py` + `router/data/dataset.py` — port `PromptSample` + `PromptDataset` (the type previously deferred from P15.3.3).
- `router/feedback/__init__.py` — package marker.
- `router/feedback/trace_to_training.py` — port `TraceRecord`, `ProductionPsiUpdate`, `TraceToTraining`. Adapt: drop ClickHouse path, expose `add_traces(list[TraceRecord])` as the only ingest.
- `router/feedback/store_adapter.py` — **new**. Reads this repo's `traces/` JSONL (and the DuckDB view in `runtime/store/traces.py` when bulk queries make sense) and yields `TraceRecord` values with `cluster_id` filled by embedding + assigning each prompt against a provided assigner. Cold-start mode: if no assigner is provided, yields records with `cluster_id=None` so the first-ever fit still has prompts to chew on.
- `router/feedback/drift_detector.py` — port `DriftReport` + `DriftDetector` verbatim.
- `router/feedback/incremental.py` — scaffold only. Raises `NotImplementedError`. Docstring points to ROADMAP. Lands so P15.3.7 imports don't break later.
- `router/tests/test_dataset.py`, `test_trace_to_training.py`, `test_drift_detector.py`, `test_store_adapter.py`.

### Out of scope (deferred)
- Online incremental Ψ updates (`incremental.py` only scaffolded; real impl deferred unless P15.3.7 hits eval cost issues).
- Latency / cost violations as error signals — `is_error` here is a boolean derived from stage errors only. Latency-based error signals stay a future enhancement.
- Embedding cache shared across `store_adapter` calls — `PromptEmbedder` already caches per-process; cross-process cache deferred.
- Webhook-style live drift alerts — drift surfaces only via pull (`/router/health` MCP tool) in P15.3.9.

## Reference → target file map

| Reference | Target | Port mode |
|---|---|---|
| `data/dataset.py` (`PromptSample`, `PromptDataset`) | `router/data/dataset.py` | verbatim — adjust imports |
| `feedback/trace_to_training.py` (`TraceRecord`, `ProductionPsiUpdate`, `TraceToTraining`) | `router/feedback/trace_to_training.py` | verbatim — drop ClickHouse query path, keep `add_traces()` ingest |
| `feedback/drift_detector.py` (`DriftReport`, `DriftDetector`) | `router/feedback/drift_detector.py` | verbatim |
| `feedback/incremental_updater.py` | `router/feedback/incremental.py` | **stub** — class + method signatures preserved, bodies raise `NotImplementedError("deferred — see ROADMAP_P15.3.md")` |
| — | `router/feedback/store_adapter.py` | new (this repo's trace store is filesystem JSONL + DuckDB, not ClickHouse) |

## Pre-work

None. P15.3.1 deps cover this phase.

Verify before starting:
- P15.3.1 + P15.3.2 + P15.3.3 green: `python -m pytest router/tests/ -v`.
- This repo has trace data for adapter tests: `ls traces/raw/*.jsonl` shows at least one file.

## Tasks (atomic, ordered)

### T1 — Port `PromptDataset`
Copy `<REF>/data/dataset.py` → `router/data/dataset.py`. Verbatim — `PromptSample` + `PromptDataset` with `__init__`, `__len__`, `__iter__`, `__getitem__`, `get_prompts`, `get_categories`, `filter_by_category`, `split`, `sample`, `save`, `load`, `from_list`, `__repr__`.

Create `router/data/__init__.py` re-exporting both names.

Retrofit `router/training/kmeans.py` (P15.3.3) to **also** accept `PromptDataset` in `train()` / `train_with_validation()`:
```python
def train(
    self,
    prompts: list[str] | PromptDataset,
    *,
    fitted_from: dict, ...
) -> KMeansTrainResult:
    if isinstance(prompts, PromptDataset):
        prompts = prompts.get_prompts()
    ...
```
This is the deferred polymorphism the P15.3.3 PLAN flagged. Single-line guard, no other changes.

### T2 — Port `TraceRecord` + `ProductionPsiUpdate` + `TraceToTraining`
Copy `<REF>/feedback/trace_to_training.py` → `router/feedback/trace_to_training.py`. Then:

- **Remove** any ClickHouse-specific code (`_query_clickhouse`, etc. — verify by `grep -i clickhouse <REF>/feedback/trace_to_training.py`).
- Keep `add_trace(TraceRecord)`, `add_traces(list[TraceRecord])`, `compute_psi_updates() -> list[ProductionPsiUpdate]`, `blend_with_profiles(profiles, alpha)`, `reset()`.
- Adjust imports to `router.models.llm_profile`.

The adapter (T3) is what feeds `add_traces()`.

### T3 — `store_adapter.py` — read this repo's traces
Create `router/feedback/store_adapter.py`:

```python
from collections.abc import Iterator
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from router.core.embeddings import PromptEmbedder
from router.core.clustering import ClusterAssigner
from .trace_to_training import TraceRecord

logger = logging.getLogger("router.feedback.store_adapter")

_TRACES_RAW = Path("traces/raw")  # repo-relative; resolve from project root


def iter_traces_since(
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
    *,
    embedder: Optional[PromptEmbedder] = None,
    assigner: Optional[ClusterAssigner] = None,
) -> Iterator[TraceRecord]:
    """Stream traces from JSONL partitions, yielding TraceRecord values.

    cluster_id is filled when both embedder + assigner are provided.
    Cold-start callers (no fitted assigner yet) pass embedder=None,
    assigner=None and get records with cluster_id=-1 (sentinel for "unassigned").
    Those records are still useful for *fitting* clusters but NOT for
    computing Psi updates (TraceToTraining.add_trace will skip cluster_id=-1).
    """

    files = _select_partition_files(since_iso, until_iso)
    for path in files:
        with path.open() as f:
            for raw_line in f:
                row = json.loads(raw_line)
                rec = _row_to_trace_record(row, embedder, assigner)
                if rec is not None:
                    yield rec


def _row_to_trace_record(
    row: dict,
    embedder: Optional[PromptEmbedder],
    assigner: Optional[ClusterAssigner],
) -> Optional[TraceRecord]:
    """Convert one JSONL row → TraceRecord, or None if essential fields missing."""

    request = row.get("request")
    if not request:
        return None  # can't embed, skip

    # selected_model: pull from stages[].routing_model (last non-null wins)
    stages = row.get("stages") or []
    selected = next(
        (s["routing_model"] for s in reversed(stages) if s.get("routing_model")),
        None,
    )
    if not selected:
        return None  # no model attribution → can't update Psi for any model

    # is_error: any stage with error != null
    is_error = any(s.get("error") for s in stages)

    # cluster_id: embed + assign if we can, else sentinel -1
    cluster_id = -1
    if embedder is not None and assigner is not None:
        emb = embedder.embed(request)
        cluster_id = assigner.assign(emb).cluster_id

    return TraceRecord(
        request_id=row["trace_id"],
        selected_model=selected,
        cluster_id=cluster_id,
        is_error=is_error,
        latency_ms=float(row.get("duration_ms") or 0.0),
        total_cost_usd=0.0,  # TODO: derive from token usage when token accounting lands
        input_text=request,
        output_text=row.get("response"),
        error_category=_first_error_category(stages),
        metadata={"timestamp": row.get("timestamp")},
    )


def _select_partition_files(since_iso, until_iso) -> list[Path]: ...
def _first_error_category(stages) -> Optional[str]: ...
```

The adapter is intentionally **thin** — it knows the JSONL schema in this repo's `traces/raw/` and nothing else. If the trace schema evolves, only this file changes.

### T4 — Port `DriftDetector` verbatim
Copy `<REF>/feedback/drift_detector.py` → `router/feedback/drift_detector.py`. Adjust imports to `router.core.clustering`. No behavioral edits.

Surface (recap):
- `DriftDetector(assigner, baseline_distance=None, drift_threshold=1.5, outlier_threshold=0.2, outlier_distance_multiplier=3.0)`
- `.check(embeddings: np.ndarray) -> DriftReport`
- `.update_baseline(new_baseline: float)` — called by P15.3.7 after a successful retrain to reset the drift baseline.

### T5 — `incremental.py` scaffold
Create `router/feedback/incremental.py`:

```python
"""Online incremental Psi update.

Deferred. See ROADMAP_P15.3.md "Out of scope" — this module is scaffolded
so harness/proposer/router_proposer.py (P15.3.7) can reference the import
path without breaking. Real implementation lands only if P15.3.7 hits
eval-cost issues that incremental updates would solve.
"""

from .trace_to_training import ProductionPsiUpdate
from router.models.llm_profile import LLMProfile


class IncrementalPsiUpdater:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "IncrementalPsiUpdater deferred — see ROADMAP_P15.3.md"
        )

    def update(self, profile: LLMProfile, update: ProductionPsiUpdate) -> LLMProfile:
        raise NotImplementedError(...)
```

### T6 — Tests

Create `router/tests/test_dataset.py`:
- `test_prompt_dataset_basic` — construct from list of dicts, len matches, iter works.
- `test_prompt_dataset_split` — `train, val = ds.split(0.8)` ratios add up.
- `test_prompt_dataset_save_load` — round-trip JSON file.

Create `router/tests/test_trace_to_training.py`:
- `test_add_traces_groups_by_cluster_and_model` — synthesize 50 `TraceRecord`s across 3 models × 4 clusters, call `compute_psi_updates()`, assert the matrix shapes match.
- `test_blend_with_profiles_respects_alpha` — α=0 returns benchmark profile unchanged; α=1 returns pure production Ψ; α=0.3 blends.
- `test_psi_update_skips_cluster_neg_one` — records with `cluster_id=-1` are excluded from Ψ math.
- `test_reset_clears_state` — after `reset()`, `compute_psi_updates()` returns empty list.

Create `router/tests/test_drift_detector.py`:
- `test_drift_under_baseline_no_reclustering` — synthetic embeddings near centroids → `needs_reclustering=False`.
- `test_drift_above_threshold_triggers_reclustering` — embeddings 3× baseline distance → `needs_reclustering=True`.
- `test_drift_outlier_fraction` — sprinkle 30% far-away points → `outlier_fraction ≥ 0.3`.
- `test_update_baseline` — manually update; subsequent `check()` uses new baseline.

Create `router/tests/test_store_adapter.py`:
- Use a `tmp_path` with a synthesized JSONL of 5 trace rows (mirroring the schema verified in T3 from `traces/raw/2026-05-09.jsonl`).
- `test_iter_traces_cold_start` — embedder=None, assigner=None → 5 records, all `cluster_id=-1`.
- `test_iter_traces_with_assigner` — pass a fitted `KMeansClusterAssigner` + `FakeEmbedder` → records have `cluster_id ∈ [0, K)`.
- `test_iter_traces_skips_unattributed` — a row with no `routing_model` in any stage is skipped.
- `test_iter_traces_marks_errors` — a row with stage error → `TraceRecord.is_error=True`.
- `test_partition_filtering_by_date` — 3 partition files (2026-05-07, 08, 09), `since_iso="2026-05-08"` skips the 2026-05-07 file.

### T7 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
python -m pytest router/tests/test_dataset.py router/tests/test_trace_to_training.py router/tests/test_drift_detector.py router/tests/test_store_adapter.py -v
python -m pytest router/tests/ -v   # full router test surface still green
```

## Acceptance criteria (DoD)

1. `python -m pytest router/tests/test_dataset.py router/tests/test_trace_to_training.py router/tests/test_drift_detector.py router/tests/test_store_adapter.py -v` is green.
2. `from router.data.dataset import PromptDataset; ds = PromptDataset.load(path); ds.save(other)` round-trips on a 100-sample dataset.
3. `TraceToTraining.compute_psi_updates()` on 50 synthetic records returns one `ProductionPsiUpdate` per distinct `selected_model`; each update's `psi_vector` has length K matching the assigner used at adapter time.
4. `DriftDetector.check(embeddings)` returns a `DriftReport` with all finite numbers and an honest `needs_reclustering` boolean derived from `drift_ratio > drift_threshold OR outlier_fraction > outlier_threshold`.
5. `iter_traces_since(...)` reads `traces/raw/*.jsonl` from this repo's actual data and yields `TraceRecord` values without raising. (Smoke test the live data path once: `python -c "from router.feedback.store_adapter import iter_traces_since; print(sum(1 for _ in iter_traces_since()))"` prints a number.)
6. `IncrementalPsiUpdater(...)` raises `NotImplementedError` with a deferral message.
7. P15.3.3's `KMeansTrainer.train` accepts both `list[str]` and `PromptDataset` (the polymorphism deferred from that phase).
8. No regressions: full `python -m pytest` green.

## Risks / open questions

- **Trace schema brittleness.** `store_adapter.py` knows the exact JSONL field names from `traces/raw/<date>.jsonl`. If P15.x evolves the schema (e.g., moves `routing_model` off stages onto a top-level field), this file breaks first. Mitigation: tests in T6 use a synthesized JSONL pinned to the schema we verified live in T3 — schema drift fails loudly with a clear stack trace pointing at `_row_to_trace_record`.
- **Cold-start cluster_id sentinel.** Reference assumes every record has a real `cluster_id`. We use `-1` for "unassigned". `TraceToTraining.add_trace` must filter these out before Ψ math; T6 verifies. If we forget to filter, `psi_vector` indexing under `-1` blows up because numpy treats it as last-index — a silent bug. Belt-and-suspenders: assert `cluster_id >= 0` at top of `add_trace`.
- **Latency / cost as error signals.** Reference docstring mentions both. We keep `is_error` as boolean stage-error only and stub `total_cost_usd=0.0` until token accounting lands. Document this gap in `TraceRecord`'s docstring so future-us doesn't assume cost-aware Ψ already exists.
- **Drift baseline bootstrapping.** `DriftDetector(baseline_distance=None)` triggers auto-baseline from the first `check()` call. P15.3.7 must initialize the detector with the silhouette-fit's intra-cluster distance as the baseline, not let `check()` self-baseline on the first arbitrary embedding batch. T6 doesn't test this — flag for P15.3.7 PLAN.
- **DuckDB vs JSONL for bulk reads.** This phase reads JSONL directly because the adapter is meant to stream over many traces. The DuckDB view in `runtime/store/traces.py` is faster for filtered queries but adds a dependency we don't need yet. If P15.3.7 finds bulk reads slow at scale (>100k traces), revisit and switch the adapter to a DuckDB query without changing `TraceToTraining` upstream.
- **Embedding throughput at adapter time.** Embedding 10k traces at MiniLM speeds (~500 emb/s on CPU) takes ~20s. Acceptable for a one-shot retrain trigger; not acceptable inside a request loop. P15.3.7 must run the adapter offline, not in `/run`.
