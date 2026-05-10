# PLAN — P15.3.9 · MCP tools: Claude Code decides retrain

| Field | Value |
|---|---|
| Phase | P15.3.9 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | P15.3.4 (drift_detector), P15.3.7 (RouterProposer), existing harness brain transport |
| Unblocks | autonomous operation — once this lands the loop closes |
| Reference | none — this phase is policy + glue |

## Goal

Close the autonomy loop: **no cron, no hardcoded threshold**. The harness
exposes router health via MCP; Claude Code (running on the user's machine,
via the same transport the harness already uses) reads the JSON, decides
whether a retrain is worth proposing, and either calls
`propose_router_retrain()` or persists a "skip" rationale.

This phase ships **the trigger**. It does **not** train (P15.3.7), score
(P15.3.6), or judge (P15.3.5) — those run when this phase fires them.

## AHE alignment

Per AutoHarness: "the brain decides; the harness records." This phase is
the literal implementation of that principle for routing:
- Claude Code is the **only** entity that calls `propose_router_retrain`.
- Every wake-up writes an artifact — either a `Lesson(kind="router_config")`
  (when retrain proposed and approved) or a `Decision(kind="router_skip")`
  (when Claude Code declined, with its rationale).
- The Evolution timeline can show both, so operators see *why* the brain
  did or didn't act, not just the outcomes.

## Scope

### In scope
- `router/feedback/health.py` — `RouterHealth` dataclass + `compute_router_health()` function. Reads `versions/router_config_current`, `traces/`, `evals/_response_cache/`, runs `DriftDetector.check()` on a recent batch. Returns the snapshot the MCP tool serves.
- `harness/introspection/lib.py` (extend) — `router_health_check()` and `propose_router_retrain()` Python implementations, sibling to the existing `list_recent_promotions`, `get_lesson`, etc.
- `harness/introspection/agent.py` (extend) — register the two tools in the `TOOLS` list with their JSON schemas.
- `harness/introspection/mcp_server.py` (extend) — wire the MCP server to expose the new tools (mirrors how existing tools are exposed; usually mechanical).
- `harness/wakeup/__init__.py`, `harness/wakeup/scheduler.py` — count-threshold wake-up: tracks traces-since-last-wakeup, fires when `n_traces >= threshold` (default 50, configurable via `HARNESS_ROUTER_WAKEUP_N`). In-process lockfile prevents concurrent wake-ups.
- `harness/wakeup/prompt.py` — the wake-up prompt template the harness sends Claude Code: "Here is the current router health JSON. Do you want to call `propose_router_retrain`? If not, explain why in one paragraph."
- `harness/wakeup/runner.py` — invokes Claude Code via the existing `harness.introspection.agent.introspect()` (dual transport from P15.3.5's extraction; reuses the brain). Captures the model's response + any tool calls. Returns `WakeupOutcome(action: "proposed"|"skipped", lesson_id?, rationale)`.
- `runtime/store/traces.py` (extend) — hook in `write_trace()` that increments a process-global counter and fires `wakeup.scheduler.maybe_fire()` when threshold crossed. Async / fire-and-forget so it never blocks `/run`.
- `ledger/decisions/` (new directory) — append-only `router_<iso>.json` files for skip rationales. Format mirrors `ledger/entries/`.
- `ledger/writer.py` (extend) — `write_decision(kind: str, payload: dict) -> Path`.
- Tests: `harness/wakeup/tests/test_scheduler.py`, `harness/introspection/tests/test_router_tools.py`, `tests/test_decision_persistence.py`.

### Out of scope (deferred)
- Adaptive threshold (e.g., decrease `wakeup_n` when drift is rising) — fixed default for v1.
- Per-tenant wake-ups → single global cadence.
- Cost-aware throttling (skip wake-up if budget exhausted) → future phase.
- UI rendering of `Decision(kind="router_skip")` artifacts → P15.3.10 may pick them up; not blocked on this phase.
- Force a wake-up via API → `POST /v1/router/wakeup` deferred. Operators can kick the harness manually for now via `python -m harness.wakeup.runner`.
- Multi-step Claude Code planning (chain of tool calls before deciding) → first version is single decision per wake-up.

## Reference → target file map

This phase is integration; no reference repo equivalent.

| New |
|---|
| `router/feedback/health.py` |
| `harness/wakeup/__init__.py`, `scheduler.py`, `prompt.py`, `runner.py` |
| `ledger/decisions/` directory |

| Existing | Edits |
|---|---|
| `harness/introspection/lib.py` | + `router_health_check`, `propose_router_retrain` |
| `harness/introspection/agent.py` | extend `TOOLS` list + dispatcher |
| `harness/introspection/mcp_server.py` | expose new tools |
| `runtime/store/traces.py` | hook `write_trace` |
| `ledger/writer.py` | + `write_decision` |

## Pre-work

- P15.3.7 must be merged — `RouterProposer` and `RouterCritic` must be callable.
- P15.3.5's brain transport (`harness/brain/transport.py`) must be live, or the existing `harness/introspection/agent.py:_call_claude_code_cli` must work.
- Verify `versions/router_config_current` resolution path is stable (P15.3.2 deliverable).

## Tasks (atomic, ordered)

### T1 — `router/feedback/health.py`
Create the data layer:

```python
@dataclass
class RouterHealth:
    cold_start: bool
    version: Optional[int]
    k: Optional[int]
    model_count: Optional[int]
    cost_weight: Optional[float]
    last_fit_at: Optional[str]
    last_fit_age_hours: Optional[float]
    trace_count_since_last_fit: int
    drift_score: Optional[float]              # None if cold-start
    drift_baseline: Optional[float]
    needs_reclustering: Optional[bool]
    current_avg_error: Optional[float]        # from latest router eval report
    current_win_rate: Optional[float]
    cluster_distribution: Optional[dict[int, int]]   # from a sample of recent traces
    fitted_from: Optional[dict]
    sample_size: int                          # how many traces went into drift calc

    def to_dict(self) -> dict: ...


def compute_router_health(
    *,
    drift_sample_size: int = 200,
    embedder=None,
) -> RouterHealth:
    """Build the snapshot. Cheap — drift sample capped at 200 traces."""
```

Implementation details:
- Cold-start (no `router_config_current`): return `RouterHealth(cold_start=True, ...)` with everything else `None` except `trace_count_since_last_fit` (which counts all traces ever — useful for the Claude Code first-fit decision).
- Fitted state: load config, count traces since `created_at`, embed a sample of `drift_sample_size` recent traces, run `DriftDetector(baseline=config['drift_baseline'])` → drift score. Read `current_avg_error` and `current_win_rate` from the most recent `evals/reports/<id>.json` whose suite has `kind="router"`.
- All file/disk I/O is best-effort; missing files yield `None` fields, never exceptions.

### T2 — MCP tool implementations
Add to `harness/introspection/lib.py`:

```python
def router_health_check() -> dict:
    """Pure read tool. Returns RouterHealth.to_dict(). No side effects."""
    from router.feedback.health import compute_router_health
    return compute_router_health().to_dict()


def propose_router_retrain(rationale: str = "") -> dict:
    """Write tool. Triggers RouterProposer.propose() → critic → approver →
    executor → ledger. Returns {action, lesson_id, version, rationale_recorded}.

    Gated by Policy: if global mode == 'off' or overrides['router_config'] == 'off',
    refuses with a clear error. Otherwise runs the full pipeline.

    The rationale arg is captured into the resulting Lesson's metadata so
    operators can see why Claude Code thought a retrain was warranted.
    """
    from harness.proposer.router_proposer import RouterProposer, RouterProposerConfig
    from harness.critics.router_critic import RouterCritic
    from harness.approver import Policy, decide
    from harness.executor.promote import promote
    from router.errors import NotEnoughDataError

    policy = Policy.from_yaml()
    if policy.mode_for("router_config") == "off":
        return {
            "action": "blocked",
            "reason": "policy mode = off for router_config",
            "lesson_id": None,
        }

    proposer = _build_router_proposer()    # uses harness brain + cache + registry
    try:
        proposal = proposer.propose()
    except NotEnoughDataError as e:
        return {
            "action": "blocked",
            "reason": f"not_enough_data: {e}",
            "lesson_id": None,
        }

    critic = RouterCritic(...)
    verdict = critic.evaluate(_ctx_for(proposal))
    outcome = LoopOutcome(proposal=proposal, verdicts=[verdict], ...)
    decision = decide(outcome, policy)

    if decision == ApprovalDecision.AUTO_APPROVE:
        lesson = promote(...)   # P15.3.7's extension
        return {"action": "promoted", "lesson_id": lesson.id, "version": lesson.version, "rationale": rationale}
    if decision == ApprovalDecision.QUEUE_HUMAN:
        # Build a queued lesson the same way existing flows do.
        lesson = build_queued_lesson(...)
        return {"action": "queued", "lesson_id": lesson.id, "rationale": rationale}
    # REJECT
    return {"action": "rejected", "reason": verdict.reason, "lesson_id": None}
```

`_build_router_proposer` is a small helper that resolves the embedder (via `runtime.embedder_pool` from P15.3.8), the registry (from agent config), and the brain-driven judge (from P15.3.5). One place to thread dependencies; tests use a substitute.

### T3 — Register tools in MCP surface
Edit `harness/introspection/agent.py:TOOLS`:

```python
TOOLS = [
    # ... existing entries (list_recent_promotions, get_lesson, ...) ...
    {
        "name": "router_health_check",
        "description": (
            "Read-only snapshot of the router's current state. Returns: "
            "cold_start flag, version, K, model_count, cost_weight, "
            "trace_count_since_last_fit, drift_score, last_fit_age_hours, "
            "current_avg_error, current_win_rate, needs_reclustering. "
            "Use this before deciding whether to call propose_router_retrain."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "propose_router_retrain",
        "description": (
            "Trigger a router config retrain. Runs proposer → critic → "
            "approver → executor → ledger. Returns the resulting Lesson ID "
            "(if promoted/queued) or a blocked-reason. Caller should pass a "
            "short rationale explaining why a retrain is warranted; this is "
            "captured into the Lesson's metadata for operator review."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "Why you think a retrain is needed (1-2 sentences).",
                },
            },
            "required": [],
        },
    },
]
```

Add the two tool names to `_execute_tool`'s dispatcher dict so Claude Code's tool calls resolve.

Edit `harness/introspection/mcp_server.py` to register the same two tools. The existing pattern (FastMCP `@server.tool` or equivalent) is mechanical; mirror what `get_lesson` does.

### T4 — Wake-up scheduler
Create `harness/wakeup/scheduler.py`:

```python
import os
import threading
from pathlib import Path

_LOCK_PATH = Path("/tmp/opentracy_router_wakeup.lock")
_TRACE_COUNTER_PATH = Path(".harness/wakeup_counter.txt")
_DEFAULT_THRESHOLD = int(os.getenv("HARNESS_ROUTER_WAKEUP_N", "50"))

_lock = threading.Lock()


def increment_trace_counter() -> int:
    """Bump the persisted counter atomically; returns new value."""
    _TRACE_COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        n = 0
        if _TRACE_COUNTER_PATH.exists():
            n = int(_TRACE_COUNTER_PATH.read_text().strip() or "0")
        n += 1
        _TRACE_COUNTER_PATH.write_text(str(n))
        return n


def reset_trace_counter() -> None:
    with _lock:
        _TRACE_COUNTER_PATH.write_text("0")


def maybe_fire(threshold: int = _DEFAULT_THRESHOLD) -> None:
    """If counter >= threshold AND no wake-up currently running, fire one
    asynchronously and reset the counter."""
    n = increment_trace_counter()
    if n < threshold:
        return

    # Try to acquire the cross-process lockfile (best-effort; fcntl on POSIX).
    try:
        _acquire_lock()
    except _LockHeldError:
        # Previous wake-up still running; skip and let the next trace try again.
        return

    reset_trace_counter()
    # Fire-and-forget: spawn a thread (or asyncio task) so /run never blocks.
    threading.Thread(target=_run_wakeup_with_lock_release, daemon=True).start()
```

`_acquire_lock` uses `fcntl.flock(LOCK_EX | LOCK_NB)`; raises `_LockHeldError` if another process holds it. The lock is released in the thread's finally block.

Important: never `time.sleep` in `maybe_fire` itself — it's called from the trace write path and must return instantly.

### T5 — Wake-up prompt + runner
Create `harness/wakeup/prompt.py`:

```python
WAKEUP_PROMPT = """\
You are the autonomous brain of the OpenTracy router. {n_traces} new traces
have been recorded since your last decision. Here is the current router
health:

{health_json}

Decide:
- If you think a retrain would meaningfully improve routing, call
  `propose_router_retrain` with a 1-2 sentence rationale explaining why.
- If you think the current config is still good (drift low, recent eval
  strong, not enough data, etc.), DO NOT call the tool. Reply with one
  paragraph explaining why you're skipping. That paragraph will be
  persisted as your decision rationale.

Be honest. Skipping is fine and often correct. Don't propose a retrain
just because you were woken up.
"""
```

Create `harness/wakeup/runner.py`:

```python
def run_wakeup(threshold: int = 50) -> WakeupOutcome:
    """Compose the prompt with current health, send to Claude Code via
    the existing harness brain, capture outcome, persist decision."""
    from router.feedback.health import compute_router_health
    from harness.introspection.agent import introspect
    from ledger.writer import write_decision

    health = compute_router_health()
    prompt = WAKEUP_PROMPT.format(
        n_traces=threshold,
        health_json=json.dumps(health.to_dict(), indent=2),
    )

    # introspect() already does tool-use loops with the registered tools,
    # so if Claude Code calls propose_router_retrain, that side-effect happens
    # inside introspect's tool dispatcher.
    result = introspect(prompt)

    # Inspect what happened: did a router_config Lesson get emitted in this
    # window? If yes, we treat the wake-up as "proposed". Else "skipped"
    # and capture the model's final text as the rationale.
    proposed_lesson_id = _last_router_config_lesson_since(result.started_at)
    outcome = WakeupOutcome(
        action="proposed" if proposed_lesson_id else "skipped",
        lesson_id=proposed_lesson_id,
        rationale=result.text,
        health_snapshot=health.to_dict(),
        timestamp=now_utc_iso(),
    )

    # Persist the decision, regardless of outcome.
    write_decision(
        kind="router_wakeup",
        payload=outcome.to_dict(),
    )
    return outcome
```

The runner is callable via `python -m harness.wakeup.runner` (with a CLI in `__main__.py`) for manual operator-triggered wake-ups during testing.

### T6 — Trace writer hook
Edit `runtime/store/traces.py`:

```python
def write_trace(rec: ExecutionRecord) -> str:
    # ... existing JSONL write ...
    trace_id = ...

    # P15.3.9 hook: bump the wake-up counter. Fire-and-forget; never blocks.
    try:
        from harness.wakeup.scheduler import maybe_fire
        maybe_fire()
    except Exception:
        # Wake-up plumbing should never break /run. Log and continue.
        logger.exception("wakeup hook failed (non-fatal)")

    return trace_id
```

### T7 — `write_decision` ledger writer
Edit `ledger/writer.py`:

```python
def write_decision(kind: str, payload: dict) -> Path:
    """Append a decision artifact to ledger/decisions/<kind>_<iso>.json.

    Decisions are distinct from entries: they record what the brain CHOSE,
    not what the system DID. A 'skip' is a valid decision and worth persisting
    so the Evolution timeline can render 'Claude Code declined retrain at T'.
    """
    decisions_dir = LEDGER_DIR / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = decisions_dir / f"{kind}_{ts}.json"

    full = {
        "kind": kind,
        "timestamp": ts,
        "payload": payload,
    }
    path.write_text(json.dumps(full, indent=2))
    return path
```

Schema for `kind="router_wakeup"`:
```json
{
  "kind": "router_wakeup",
  "timestamp": "20260510T143022Z",
  "payload": {
    "action": "skipped" | "proposed" | "blocked",
    "lesson_id": "L-..." | null,
    "rationale": "Drift is 0.04, well below baseline. Most recent eval was 6 hours ago. Skipping.",
    "health_snapshot": { ... },
    "timestamp": "..."
  }
}
```

### T8 — Tests

`harness/wakeup/tests/test_scheduler.py`:
- `test_increment_persists` — counter bumps across function calls.
- `test_maybe_fire_below_threshold_doesnt_fire` — threshold=50, after 49 increments no thread spawned (mock `_run_wakeup_with_lock_release`).
- `test_maybe_fire_at_threshold_fires_once` — at 50th increment, thread spawned exactly once; counter resets to 0.
- `test_concurrent_wakeups_skip_when_locked` — simulate two simultaneous threshold-crossings; only one wake-up runs.
- `test_lockfile_released_on_failure` — wake-up that raises still releases the lock.

`harness/introspection/tests/test_router_tools.py`:
- `test_router_health_check_cold_start` — no config → returns dict with `cold_start: True`.
- `test_router_health_check_fitted` — synthesize config → returns full dict; finite numbers; `drift_score` non-null.
- `test_propose_router_retrain_blocked_by_policy` — policy `mode: off` → returns `action: blocked` with reason; no Lesson written.
- `test_propose_router_retrain_blocked_by_data_gate` — fewer traces than `min_corpus_size` → returns `action: blocked` with `not_enough_data` reason.
- `test_propose_router_retrain_promotes_on_clear_win` — synthesize state where critic passes → returns `action: promoted`, lesson_id set, `versions/router_config_v2.json` exists.

`tests/test_decision_persistence.py`:
- `test_write_decision_creates_file` — `write_decision("router_wakeup", {...})` writes `ledger/decisions/router_wakeup_<iso>.json`.
- `test_skip_outcome_persists_rationale` — synthesize a wakeup that skips; assert the decision file exists with the rationale text.
- `test_proposed_outcome_persists_lesson_id` — synthesize a wakeup that promotes; the decision file's `payload.lesson_id` matches the new Lesson's ID.

`tests/test_wakeup_does_not_block_run.py` (integration):
- Time the `/run` request before and after the wake-up hook fires. Hook addition must not measurably block (delta < 5ms). Verifies the fire-and-forget pattern.

### T9 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
python -m pytest harness/wakeup/tests harness/introspection/tests tests/test_decision_persistence.py tests/test_wakeup_does_not_block_run.py -v
python -m pytest -v   # full suite

# Manual end-to-end (with backend + UI dev servers up + brain reachable):
HARNESS_ROUTER_WAKEUP_N=5 ./scripts/run_runtime.sh &
# Send 5 prompts to /run
for i in 1 2 3 4 5; do
  curl -s -X POST -H "Authorization: Bearer dev" -H "Content-Type: application/json" \
    -d "{\"request\": \"test prompt $i\"}" http://localhost:8002/v1/run > /dev/null
done
# Wait a few seconds for the wake-up subprocess to finish
sleep 5
# Confirm: either a router_config Lesson appeared in /v1/lessons OR a skip decision exists
ls ledger/decisions/router_wakeup_*.json | tail -1
cat $(ls ledger/decisions/router_wakeup_*.json | tail -1) | jq
```

## Acceptance criteria (DoD)

1. **Wake-up fires after N traces:** with `HARNESS_ROUTER_WAKEUP_N=5`, sending 5 `/run` requests triggers exactly one wake-up. Trace counter resets to 0. Test in T8 verifies.
2. **Wake-up paths exit cleanly:**
   - **Proposed path:** Claude Code calls `propose_router_retrain`; a `Lesson(kind="router_config", proposal_source="claude_code")` exists in `/v1/lessons`; `versions/router_config_v(N+1).json` written; decision artifact at `ledger/decisions/router_wakeup_*.json` records `action: "proposed"` + the lesson_id.
   - **Skipped path:** Claude Code does not call the tool; decision artifact records `action: "skipped"` + the model's rationale paragraph; no new Lesson, no version bump.
   - **Blocked path:** policy off OR not enough data OR critic fails → decision artifact records `action: "blocked"` with reason; no Lesson.
3. **Wake-up never blocks `/run`:** `tests/test_wakeup_does_not_block_run.py` shows < 5ms hook overhead.
4. **`router_health_check` MCP tool returns honest data:** cold-start returns `cold_start: true` and `drift_score: null`; fitted state returns finite numbers for all fields.
5. **`propose_router_retrain` is policy-gated:** `policies/policy.yaml` with `mode: off` causes the tool to return `action: "blocked"` without invoking the proposer.
6. **Decision artifacts are queryable:** `ls ledger/decisions/router_wakeup_*.json` shows one file per wake-up. Schema matches T7. Operators can grep them to understand why retrains did or didn't happen.
7. **Concurrent wake-ups don't double-fire:** lockfile test in T8 confirms only one runs at a time.
8. **Lockfile released on crash:** wake-up that raises still releases the lock (T8 test).
9. **No regressions:** full `python -m pytest` green; existing `/run` and `/v1/introspect` paths unchanged.
10. **Decision rationale is non-empty:** for skip outcomes, the persisted `rationale` field is at least one sentence (non-trivial). Empty rationales fail the smoke test.

## Risks / open questions

- **Wake-up runs in process vs subprocess.** Current plan: thread inside the runtime process, calling `introspect()` synchronously (the introspect call itself spawns `claude --print` if CLI transport). If the runtime crashes mid-wake-up, the lockfile leaks. Mitigation: lockfile on `/tmp` includes process PID; on next start, sweep stale locks. Document in T4.
- **Cost of every wake-up.** Each invocation = one Claude Code call (read tool + decision) + possibly one full retrain ($1-3 if proposed). At `HARNESS_ROUTER_WAKEUP_N=50`, with ~500 traces/day, that's 10 wake-ups/day = ~$5-30/day. Configurable, but worth flagging in `policies/README.md`.
- **`introspect()` reuse vs new helper.** `harness/introspection/agent.py:introspect()` is the existing entry — it does tool-use loops with the full TOOLS list, including the introspection tools (list_recent_promotions, etc). We piggyback on that for wake-ups. The brain has access to *all* introspection tools, not just the two new ones. That's fine — Claude Code can read context (recent lessons, predictions, drift) before deciding.
- **Drift baseline propagation.** `compute_router_health` reads `drift_baseline` from the persisted config (set by `RouterProposer` in P15.3.7's T8). If the field is missing on an older config, fall back to auto-baseline from the current sample's mean — log a WARNING noting we should re-promote to fix.
- **Threshold tuning.** Default 50. If too low → too many wake-ups, costs balloon. If too high → drift not detected fast enough. Document operator knob in README; suggest starting at 50 and adjusting based on cost dashboards.
- **Race on first deployment.** Day-zero: no config, no traces. First 50 traces trigger first wake-up. Claude Code sees `cold_start: true`, `trace_count: 50`, decides if 50 is enough to fit (the `min_corpus_size` is 200 in P15.3.3; below 200 → `propose_router_retrain` returns `not_enough_data: blocked`). Document this behavior — Claude Code might propose anyway and learn the gate exists.
- **`introspect()` tool calls and the lockfile.** If Claude Code's tool call to `propose_router_retrain` itself takes >60s, the lockfile blocks. Acceptable — concurrent retrains would clobber each other. Document timeout in T5 (~600s ceiling).
- **Decision artifact size.** `health_snapshot` adds ~1KB per decision; over a year that's ~3.5MB. Acceptable. Compaction (rotate to `ledger/decisions/archive/<year>/`) deferred.
- **Failure mode: no brain available.** `select_transport()` returns `none` if neither API key nor CLI is reachable. `run_wakeup` should detect this and write `action: "blocked"` with `reason: "no_brain_available"` rather than crashing. T8 test for this case.
