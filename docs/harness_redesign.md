# Harness Redesign — Plan

## Context

The current harness is **alive as a library, dead as a system**. Agents run when called (clustering, trace scanning, metrics suggestion), but nothing runs *itself*: the scheduler defaults off, the operator defaults off, memory is wired but never written, the UI page exists but isn't routed. Agent outputs evaporate after each call — they don't compound.

We are not going to rewrite the harness. We are going to:

1. Give it a spine (one trigger engine instead of four orchestrators).
2. Give it a memory that compounds (structured ledger instead of free-text notes).
3. Make it observable (ledger-driven dashboards, routed UI).
4. *Then* — and only then — layer auto-research on top.

---

## Sequencing

**Phase 1 — Observability Foundation.** Make the system traceable, measurable, and auditable. Turn on schedulers one at a time, each tied to a user-declared objective, each writing a structured row. Value: users get a live feedback layer; we accumulate data.

**Phase 1.5 — Claude Code as Operator.** Expose the harness's read + write surface through an MCP server so Claude Code becomes the human-operator interface over the running system. Before we hand the operator role to an auto-researcher (Phase 2), we first validate the *interface* by giving a human Claude-Code-driven access to everything auto-research will eventually touch — ledger queries, chain drill-down, policy enable/disable, manual recipe triggers. This step catches gaps in the harness's tool surface and in the ledger's expressiveness that would otherwise be discovered only when auto-research silently fails.

**Phase 2 — Auto-Research.** Reads the Phase 1 ledger. Proposes parameter mutations scored against the same objectives. Small module because the hard parts — what "better" means, how we know it moved, why it moved — are already solved, and the tool surface is already shaped correctly by the Phase 1.5 operator work.

Do not skip ahead. An auto-research layer on sparse data or an unproven tool surface hallucinates causality.

---

## The Observability Contract

Observability is a **system contract**, not an objective. It either holds or it doesn't.

**Three invariants.** If any is broken, observability is broken.

1. **Every scheduler run writes one structured row.** Always. Never free-text.
2. **Every action is traceable back to a signal.** Chain of causation via `parent_id`.
3. **Every objective is a queryable time series with annotations.** Plot it; every action the system took shows up as a marker.

**Acceptance test for "observability is done enough":** when a scheduler run fires, can you point at the resulting row and answer "did this move the objective, yes/no"? If yes, move on. Coverage is not the test.

**The trap to avoid.** Observability becoming the mechanism that *is* the work. Instrument only in service of a decision you are about to make. If nobody will read a metric, don't emit it.

---

## Primitives

Small vocabulary on purpose.

| Primitive | What it is | Produced by |
|---|---|---|
| **Objective** | User-declared goal with a deterministic compute function | Humans (config) |
| **Sensor** | Deterministic code that watches the world, emits signals. No LLM. | Harness core |
| **Signal** | Ledger entry: "something changed" | Sensors |
| **Policy** | Declarative `signal_type → task_recipe` mapping (YAML) | Humans (Phase 1), auto-research (Phase 2) |
| **Task recipe** | Ordered list of agent calls + actions + guardrails (YAML) | Humans |
| **Agent** | `.md` file, stateless, returns structured JSON. Writes result to ledger. | Existing registry |
| **Action** | Concrete side-effect (run eval, queue training, promote config). Code, not an agent. | Harness core |
| **Ledger entry** | Append-only structured row with typed payload + `parent_id` | Every primitive above |

---

## Data Models

### Objective

```yaml
id: cost_per_successful_completion
description: >
  USD cost per completion where judge_score >= 0.8, averaged 7d.
  Per tenant, per domain.
compute_fn: objectives.cost_per_success   # Python callable
baseline: 0.0042           # frozen at objective creation
target: 0.0030
unit: USD
update_cadence: hourly
dimensions: [tenant, domain]
owner_agents: [training_advisor, metrics_suggester]
guardrails:
  - no_regression_worse_than: 5%   # cannot promote if this slice regresses
  - min_sample_size: 200
```

Rules:
- Compute function must be deterministic and side-effect-free.
- Baseline is frozen when the objective is created. Progress is always measured against it.
- Three to five objectives, not twenty. Adding a sixth requires retiring one.

### Ledger entry (run log row)

```
id              uuid
ts              timestamp
type            enum: signal | run | observation | proposal | decision | action | lesson
objective_id    fk -> objectives (nullable; signals may be objective-agnostic)
subject         string  (cluster_id, model, dataset_id, trace_id, ...)
agent           string  (null if type is signal/action)
parameters_in   jsonb
data            jsonb   (typed payload by entry type)
parent_id       uuid    (chain of causation; null for root signals)
tags            string[]
duration_ms     int
cost_usd        numeric
outcome         enum: ok | failed | rolled_back | skipped
```

One table. One append-only insert per primitive execution. No updates. Root signals have `parent_id = null`; everything else points back to its cause.

### Guardrail outcome (inline in ledger `data` for `decision` rows)

```json
{
  "objective_deltas": {"cost_per_success": -0.0003, "p95_latency": +12},
  "guardrails_checked": ["no_regression_worse_than_5pct", "min_sample_size"],
  "guardrails_violated": [],
  "verdict": "promote"
}
```

---

## Directory Layout

```
opentracy/harness/
├── core/                # keep
│   ├── registry.py      # .md agent loader (existing)
│   ├── runner.py        # agent executor (existing)
│   └── toolkit.py       # tools available to agents (existing)
├── ledger/              # NEW — replaces memory_store.py
│   ├── store.py         # append-only writer + indexed reader
│   ├── entry.py         # typed entries
│   └── query.py         # chain/thread/time-window queries
├── objectives/          # NEW
│   ├── definitions/     # *.yaml — one file per objective
│   └── compute.py       # compute_fn implementations
├── sensors/             # NEW — deterministic signal producers
│   ├── cost.py
│   ├── coverage.py
│   ├── quality.py
│   └── cadence.py       # time-based triggers
├── triggers/            # NEW — the spine
│   ├── engine.py        # replaces scheduler.py + operator.py + scheduled_trainer.py
│   └── policies/        # *.yaml — signal_type → task_recipe
├── tasks/
│   ├── executor.py      # runs task queue, enforces budget
│   └── recipes/         # *.yaml — ordered agent calls + actions
├── agents/              # existing .md files, reorganized by role
│   ├── inspectors/      # cluster_labeler, coherence_scorer, outlier_detector, trace_scanner, dataset_summarizer
│   ├── proposers/       # merge_checker, metrics_suggester, eval_generator, training_advisor
│   ├── critics/         # NEW — review proposals
│   └── narrators/       # NEW (Phase 2) — distill ledger chains into lessons
├── actions/             # NEW — code side-effects
│   ├── run_eval.py
│   ├── queue_training.py
│   └── promote_config.py
└── api.py               # /v1/harness/{ledger, signals, tasks, policies, agents, objectives}
```

**What gets deleted after migration:** `operator.py`, `scheduler.py`, `trace_scanner.py` (decomposed), `training_advisor.py` (decomposed into proposer + action), `memory_store.py`, `feedback/scheduled_trainer.py`.

**What stays:** `registry.py`, `runner.py`, `toolkit.py`, existing `agents/*.md` bodies (just moved into role subfolders).

---

## Phase 1 Execution Plan

Each step is useful standalone. Do not merge them.

### Step 1 — Agree three objectives (week 1)

Write three objective YAML files end-to-end. Compute functions + baselines + targets + guardrails. Do not proceed until these are landed and the compute functions return numbers against real data.

Candidates to pick from:
- `cost_per_successful_completion` (tenant × domain, 7d rolling)
- `eval_pass_rate` (dataset × judge, per run)
- `p95_latency_routed` (production traffic, per model)
- `judge_score_on_domain` (cluster, weekly)
- `student_cost_savings_vs_teacher` (per deployed student)

Pick three. Defer the rest.

### Step 2 — Ledger schema + store (week 1)

- Backend: SQLite for the harness (small, transactional, heavy-read-by-chain). *Not* ClickHouse — that's for traces, which have an append-heavy analytical access pattern.
- One table. Append-only. Indexed on `(objective_id, ts)`, `(parent_id)`, `(type, ts)`.
- `ledger.store.append(entry)`, `ledger.query.chain(root_id)`, `ledger.query.time_series(objective_id, window)`.

Dual-write to old `MemoryStore` for one cycle to compare — don't delete yet.

### Step 3 — Wire one scheduler end-to-end (week 2)

Pick `trace_scanner` — it's the best-fleshed-out chain today. Reshape it as:

1. Sensor: `cadence_sensor` fires every N hours → emits `signal(type=cadence, objective_id=judge_score_on_domain)`.
2. Trigger engine matches policy `cadence@judge_score_on_domain → recipe: trace_scan_and_evaluate`.
3. Task recipe: `inspectors/trace_scanner` → `proposers/eval_generator` → `critics/budget_justifier` → `actions/run_eval`.
4. Each step writes a ledger row with `parent_id` pointing at its predecessor.

Enforce a budget cap. This flow proves the shape end-to-end.

### Step 4 — Objective dashboard UI (week 3)

- Route `HarnessPage.tsx` (currently orphaned in `ui/src/features/harness/`).
- Three tabs: Objectives, Ledger, Tasks.
- Objectives tab: one plot per objective with action markers on the x-axis. Click a marker → drill into the ledger chain that produced it.
- Acceptance test: "Why did objective X move on Tuesday?" answerable in under a minute from the UI.

### Step 5 — Enable remaining schedulers (weeks 4–5)

One at a time. Each one becomes a policy, each one tied to an objective, each one behind a per-objective daily budget cap.

Order:
1. Clustering pipeline's inline agent calls → policy `new_traces_threshold → cluster_and_label`.
2. Metrics suggester → policy `new_dataset → suggest_metrics`.
3. Training advisor → policy `cost_drift → evaluate_student_distillation`.

After each lands, delete the old orchestration code it replaces.

### Step 6 — Data accumulation (weeks 6–10)

Resist the urge to build auto-research here. Let the ledger fill. Watch objectives move. Fix whatever the dashboards expose (they will expose things).

---

## Phase 1.5 — Claude Code as Operator (weeks 8–10, overlaps Step 6)

### Why this exists

Auto-research (Phase 2) is "Claude Code without a human keyboard." Before handing that power to an autonomous loop, we validate the same tool surface with a human in it. Two benefits:

1. **Tool-surface validation.** The MCP tools we expose now are exactly what auto-research will later consume. If a tool is missing, awkward, or gives ambiguous output, we learn it while a human is there to course-correct. Discovering tool gaps in Phase 2 means discovering them via silent bad decisions.
2. **Operator-in-the-loop is the right default.** Most production incidents in the first months will want a human investigating, not an agent mutating config. Claude Code over MCP gives us that — the same conversational UI that answered "why did X move?" in the dashboard is now usable programmatically: "close the signal chain for signal-abc, approve the queued training, then disable cost_drift policy for 24h."

### What it looks like

```
  [Operator in Claude Code session]
       ↓ MCP (stdio)
  [opentracy-harness MCP server]
       ↓ in-process imports
  [Ledger | Triggers | Recipes | Policies]
```

Claude Code session example:
> — Why did `p95_latency_ms` jump yesterday afternoon?
> — Pull the signal chain from 14:00 and explain.
> — Ok, disable `latency_drift_to_trace_scanner` for 6h while I look into the provider outage.
> — On resume, run `trace_scan_and_evaluate` manually against the last 24h.

### MCP tool surface (minimal viable)

Tools expose only the harness's existing read/write primitives — no new behavior.

**Read:**
- `list_objectives` — YAML definitions + latest value per objective.
- `get_objective_time_series(id, hours)` — measurements + markers, same shape as the dashboard endpoint.
- `list_ledger_entries(type, objective_id, agent, limit)` — filtered listing, newest first.
- `get_ledger_entry(id)` / `get_ledger_chain(id)` — single entry / full causal chain.
- `list_policies()` / `list_recipes()` / `list_agents()` — what's loaded from YAML.
- `describe_policy(id)` / `describe_recipe(id)` — full YAML dump with loaded state.

**Write (human-authored, not auto):**
- `set_policy_enabled(id, enabled, reason)` — runtime enable/disable with the reason written to the ledger as a `decision` row.
- `run_recipe(recipe_id, parent_id?)` — manual dispatch for incident response. Writes the same ledger shape as auto-driven runs so the chain stays uniform.
- `acknowledge_signal(id, note)` — writes a `decision` row chaining to the signal, letting the operator record "I saw this; no action needed."

Every write emits a ledger row tagged `mcp_operator` so the dashboard differentiates human-via-Claude-Code from auto-driven causation. No write operates outside the ledger.

### Out of scope for Phase 1.5

- Autonomous decision making. Claude Code here is operator-triggered. It reads freely; it writes only when explicitly told by the human.
- Direct mutation of YAML files. Policy disable via MCP is runtime-only and expires on process restart unless persisted via a separate human commit.
- Cross-session state. Each Claude Code session is stateless on the MCP side; durability lives in the ledger.

### Acceptance

Phase 1.5 is done when:

- A non-maintainer can clone the repo, run `claude mcp add opentracy-harness python -m opentracy.harness.mcp_server`, and answer "why did X move on Tuesday?" within one Claude Code session without reading any other source file.
- Every read the Phase 2 auto-research loop will need is available as an MCP tool (so Phase 2 becomes "same tools, no human in the loop").
- At least one real incident has been investigated through this path, producing a lesson in the ledger that Phase 2 can later consume.

### Open question: executor choice

Three ways to wire Claude Code into the harness. Phase 1.5 commits to the **first**, but leaves room for the others:

1. **MCP server exposing harness primitives (this phase).** Claude Code on the operator's laptop connects via MCP. No change to how recipes execute in production — cheap Mistral/OpenAI LLM calls still drive the autonomous loop. Claude Code only runs when the human invokes it.
2. **Hybrid: Claude-Agent-SDK-backed agents in specific recipes (deferred).** Certain agent roles — `critics` and, later, `narrators` — benefit from tool use and multi-turn reasoning. We keep `.md` agent definitions but add an `engine: claude-agent-sdk` field that the `AgentRunner` dispatches differently. Gated per-agent so cost stays bounded.
3. **Full replacement of the recipe executor with Claude Code (not recommended).** Every agent step becomes a Claude Code invocation. Loses the YAML-declarative structure that makes the harness reviewable, multiplies cost by ~20×, and couples production behavior to a heavyweight subprocess. Flagged here only so the option is considered and dismissed on record.

---

## Phase 2 — Auto-Research (sketch, not spec)

Kicks in only after Phase 1 is solid.

### Central config file

Single source of truth for the optimization surface. Read by the SDK/engine to actually run; read by auto-research to propose mutations.

```yaml
live:
  routing:
    similarity_threshold: 0.78
    cost_quality_alpha: 0.6
    fallback_policy: cheapest_first
  clustering:
    k_bounds: [20, 80]
    min_cluster_size: 15
    coherence_threshold: 0.72
  distillation:
    default_teacher: openai/gpt-4o-mini
    lora_rank: 16
    bond_beta: 0.1
    judge_cutoff: 0.8
  eval:
    judge_model: anthropic/claude-sonnet-4-6
    default_metrics: [similarity, latency, cost, llm_judge]

candidate:
  # pending mutation under evaluation — written by auto-research,
  # read by the objective function, decided on by the critic
  null_or:
    id: exp-abc123
    diff_from_live: {...}
    hypothesis: "higher similarity threshold reduces miss-routes on JS domain"

guardrails:
  max_cost_regression_pct: 5
  min_sample_size: 200
  per_domain_no_regression: true

budget:
  daily_experiment_cap_usd: 20
  weekly_experiment_cap_usd: 100
  remaining_today: 14.37
```

### Objective function

Single entrypoint. Deterministic code. Returns a vector, not a scalar.

```python
def objective(config: dict) -> ObjectiveResult:
    """Run the config against held-out data, return scores + guardrail verdicts."""
    # scores: {objective_id -> float}
    # guardrail_verdicts: {guardrail_id -> pass|fail}
```

### The loop (one more policy)

Auto-research is not a new system — it's a policy:

```
signal: central_file_promotion_window_open
→ recipe:
    - proposers/autoresearcher   # reads ledger + current config → emits candidate
    - actions/run_objective      # evaluates candidate against objectives
    - critics/auto_research_critic  # checks guardrails, compares to live
    - actions/promote_or_reject
```

The experiment ledger *is* the harness ledger — no separate storage.

### New agents for Phase 2

- **`proposers/autoresearcher`** — reads recent ledger + `lesson` entries + current config, emits a candidate mutation with a hypothesis.
- **`critics/auto_research_critic`** — blocks promotion if guardrails fail or if the mutation re-explores territory the ledger says failed before.
- **`narrators/lesson_summarizer`** — periodically reads closed ledger chains and writes `lesson` entries (e.g. "cost drift on JS → student distillation gave 3× savings, 0% regression"). Feeds future proposers.

---

## What We Keep / Delete / Add

| Status | Component |
|---|---|
| **Keep** | `registry.py`, `runner.py`, `toolkit.py`, agent `.md` bodies |
| **Keep** | `/v1/harness/agents`, `/v1/harness/run/{name}` endpoints |
| **Delete** | `memory_store.py`, `operator.py`, `scheduler.py`, `trace_scanner.py`, `training_advisor.py`, `feedback/scheduled_trainer.py` |
| **Add** | `ledger/`, `objectives/`, `sensors/`, `triggers/`, `tasks/`, `actions/` |
| **Add** | `critics/`, `narrators/` agent roles |
| **Restructure** | `agents/` into `inspectors/`, `proposers/`, `critics/`, `narrators/` |
| **Route** | `HarnessPage.tsx` + Ledger + Objectives tabs |

---

## Risks & Guardrails

**Observability mechanism-is-the-work.** Ship value, not telemetry. Every metric emitted must answer a decision someone is about to make. Kill unread metrics.

**Objective drift.** Aspirational objectives ("improve quality") are poison. If you cannot write the SQL, it is not an objective. Three objectives with compute functions beat twenty poetic ones.

**Budget runaways.** Turning schedulers on without caps burns money. Per-objective daily caps *before* enabling, enforced by the trigger engine, logged into the ledger.

**Auto-research prematurity.** Do not start Phase 2 before the ledger has four to six weeks of real data and at least three objectives moving through promotions. Premature auto-research hallucinates causality.

**Consolidation risk.** Deleting `operator.py`, `scheduler.py`, `scheduled_trainer.py` in the middle of the migration breaks things. Rule: new trigger engine ships feature-parity for one orchestrator at a time. Delete the old one only after the new one has owned its traffic for a full week.

---

## Acceptance Tests

**Phase 1 is done when:**
- Three objectives have populated time series with at least four weeks of data.
- Every scheduler emits a structured ledger row per run.
- Every action has a chain back to a signal via `parent_id`.
- The UI answers "why did objective X move on date Y" in under a minute.
- `operator.py`, `scheduler.py`, `scheduled_trainer.py`, `memory_store.py` are deleted.

**Phase 2 is done when:**
- Auto-research has proposed, evaluated, and either promoted or rejected at least ten candidate mutations against real objectives.
- At least one narrated `lesson` has been fed back into a proposer and visibly shaped a later proposal.
- The central config file is the only source of truth for the optimization surface (no orphan defaults in other modules).

---

## Open Questions

- **Ledger backend.** SQLite (assumed) vs Postgres. SQLite if single-node; Postgres if the harness spans processes. Decide before Step 2.
- **Policy language.** YAML + Jinja is probably enough. If a policy needs real logic, write a task recipe or an agent instead. Resist Turing-completeness.
- **Budget accounting.** Ledger-as-source-of-truth (budget = sum of ledger `cost_usd`) vs separate counter. Ledger-as-source is simpler; keep it.
- **Signal bar.** "New trace arrived" is not a signal. "Cost on domain X moved > 2σ over 7d baseline" is. Codify the bar in a sensor-writing guide before week 2.
- **Who declares objectives — operator or platform user?** Affects UX. Default: platform user via UI; operator can seed defaults.

---

## Immediate Next Artifacts

Before any code lands:

1. Three objective YAMLs with working `compute_fn`s against current data.
2. Ledger entry Pydantic model + SQLite schema.
3. One policy YAML (the `trace_scanner` rewire) end-to-end.

Everything downstream is constrained by these three. Get them right and the rest is mechanical.
