# PLAN — P15.3.10 · UI Router config (port verbatim from design)

| Field | Value |
|---|---|
| Phase | P15.3.10 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | P15.3.2 (`/v1/router/config`), P15.3.7 (router_config Lessons), P15.3.8 (`routing_decision` in traces), P15.3.9 (decision artifacts) |
| Unblocks | — (closes the P15.3 roadmap) |
| Reference | `/tmp/design_unpack/opentracy-front-end-remix/project/screens/Technical.jsx:670-1006` (RouterConfig + RuleDrawer + NewRuleModal) |

## Goal

Port the design's Router config panel verbatim. Plug real data into the
shapes the design provides — **without** modifying layout, classes,
paddings, proportions, or adding new components.

## Important: design ↔ backend reconciliation

After inspecting the design more carefully, the panel is **rule-based**
(declarative `if when then model` rules with share / cost / authorship)
while what we built in P15.3.1–9 is **UniRoute** (continuous embedding
clusters + Ψ matrix). These are different routing paradigms.

The roadmap originally promised UI elements that **do not exist in the
design**: "Ψ heatmap, cluster distribution chart, λ slider, candidate
queue". The design has none of those. The user's later constraint —
*"deixei mais fiel o possivel ou se tiver deixe na mesma estetica mas só
se for extremamente necessário"* — supersedes the roadmap.

**Reconciliation adopted in this PLAN:**

1. **Port the design's UI verbatim.** Layout, classes, drawer chrome,
   summary cards, filter pills, modal — all unchanged.
2. **UniRoute lives as the "default" rule row.** The design has a
   `isDefault: true` rule that's the catch-all; we map UniRoute to that
   row. Click into it → existing `RuleDrawer` renders UniRoute internals
   in its existing three-tab layout (Overview / Sample matches / History).
3. **No new UI components.** Ψ summary, K, drift, λ, candidate queue
   are rendered as **text/meta rows** inside the existing drawer's
   Overview tab — no inline SVG heatmap, no slider, no separate panel.
4. **Manual routing rules are deferred.** The "+ Add routing rule" button
   is shown but disabled for v1, with an "Coming soon" tooltip. Per the
   no-fabrication rule (memory `feedback_ui_panels.md`), we don't ship a
   non-functional modal that pretends to work.
5. **Top summary cards** derive from `/v1/router/config` + trace data;
   missing fields render as `—`, never synthesized values.
6. **Trace drawer** (separate screen) renders the `routing_decision`
   payload exposed by P15.3.8 — that's where users see "why this model"
   per request.

If the user wants the dropped elements (Ψ heatmap, λ slider) elsewhere,
that's a follow-up phase, not this one.

## Scope

### In scope
- `ui/src/screens/Technical/RouterConfig.tsx` — port `RouterConfig` from `<DESIGN>/screens/Technical.jsx:670-861`. Wire summary cards + rules table + filters + search to real data.
- `ui/src/screens/Technical/RuleDrawer.tsx` — port `RuleDrawer` from `<DESIGN>/screens/Technical.jsx:862-944`. Add a small content adapter: when the rule represents the UniRoute default, the Overview / Samples / History tabs render UniRoute-derived data; for any future non-default rule, render the design's stock fields.
- `ui/src/screens/Technical/NewRuleModal.tsx` — port `NewRuleModal` from `<DESIGN>/screens/Technical.jsx:945-1006`. **Disable** the submit button for v1; show `Coming soon — manual rules deferred` in the modal body (same chrome, different copy).
- `ui/src/screens/Technical/index.tsx` (extend) — wire `RouterConfig` into the existing Technical screen routing; replace whatever mock currently fills `/technical/router`.
- `ui/src/api.ts` (extend) — add typed clients for `getRouterConfig()`, `getRouterRules()`, `getRouterCandidates()`, `getRouterHealth()`, plus the existing `getTraceDetail()` already returns the new `routing_decision` fields.
- `runtime/server.py` (extend) — `GET /router/rules`, `GET /router/candidates`, `GET /router/health`. Three small read endpoints; payloads documented below.
- `backend/channels/router/{rules,candidates,health}.ts` — proxies for the three new endpoints.
- `ui/src/screens/Technical/TraceDrawer.tsx` (extend) — within the existing Trace drawer (already shipped pre-P15.3), render the new `routing_decision` payload as a Routing section. Same chrome (existing `.sheet-section`).
- `ui/src/styles.css` — verify all classes the design uses are present (`router-summary`, `dist-bar`, `seg`, `seg-haiku`, `seg-sonnet`, `rule-head`, `rule-row`, `rule-priority`, `reorder`, `row-action`, `toggle`, `meta-grid`, `meta-row`, `sample`). Add **only** any that are missing — port from design's `styles.css` verbatim.
- Tests: `ui/src/screens/Technical/__tests__/RouterConfig.test.tsx`, `RuleDrawer.test.tsx`. Visual smoke via Playwright (compare to design screenshot in `/tmp/design_unpack/...`).

### Out of scope (deferred)
- Manual routing rules engine (rule evaluator, persistence) — `+ Add routing rule` stays disabled.
- Ψ heatmap rendering — design has no such element; defer until a future phase explicitly requests it.
- λ slider in the panel — the operator-driven λ override path lives at `PUT /v1/router/config` (P15.3.8); UI control deferred. Ops can curl for v1.
- Candidate queue as a separate UI section — candidates already surface in the existing Review queue (`Lesson(status="awaiting_review", kind="router_config")`); link from the default-rule drawer's History tab.
- Reordering rules / drag-and-drop — design has reorder buttons but for v1 we have only one row; the buttons are present in the chrome but hidden on the default row (already true in the design via `!r.isDefault`).
- Mobile / responsive tuning — desktop-first per the existing UI; mobile viewport untested in this phase.

## Reference → target file map

| Design (`screens/Technical.jsx`) | Target (`ui/src/screens/Technical/`) | Port mode |
|---|---|---|
| `RouterConfig` (lines 670–861) | `RouterConfig.tsx` | verbatim layout; wire data |
| `RuleDrawer` (lines 862–944) | `RuleDrawer.tsx` | verbatim layout; conditional content for default rule |
| `NewRuleModal` (lines 945–1006) | `NewRuleModal.tsx` | verbatim layout; disabled submit + "Coming soon" copy |
| `ROUTER_RULES` mock data (lines 649–666) | — | dropped; data comes from `/v1/router/rules` |

## Endpoint surfaces (new, in scope)

### `GET /v1/router/rules`
Synthesized rules list. For v1 returns exactly one row representing UniRoute:

```json
[
  {
    "id": "uniroute_default",
    "name": "UniRoute (default)",
    "when": "default",
    "then": "uniroute",
    "share": 1.0,
    "cost": null,
    "auth": "agent",
    "enabled": true,
    "isDefault": true,
    "rationale": "Trained router. Picks per request based on prompt embedding cluster + per-model expected error.",
    "history": [...],
    "samples": [...]
  }
]
```

`history` derived from `Lesson(kind="router_config")` entries via `/v1/lessons?kind=router_config`. `samples` derived from `/v1/router/health.cluster_distribution` plus a representative prompt per top cluster (best-effort from recent traces).

### `GET /v1/router/candidates`
Candidate proposals awaiting review:

```json
[
  {
    "lesson_id": "L-20260510-...",
    "version_proposed": 3,
    "delta_auroc": 0.04,
    "delta_avg_error": -0.02,
    "created_at": "2026-05-10T14:22:00Z",
    "review_link": "/review/L-20260510-..."
  }
]
```

Read from `/v1/lessons?kind=router_config&status=awaiting_review`. Empty array when no candidates pending.

### `GET /v1/router/health`
Same payload as the MCP tool from P15.3.9, exposed over HTTP for the UI.

## Pre-work

- P15.3.2 endpoints (`GET /router/config`) live and returning cold-start payloads correctly.
- P15.3.7 producing `Lesson(kind="router_config")` entries.
- P15.3.8 stamping `routing_decision` into `StageRecord` and the trace JSON.
- P15.3.9's `compute_router_health()` callable.

## Tasks (atomic, ordered)

### T1 — Port `RouterConfig.tsx` shell
Create `ui/src/screens/Technical/RouterConfig.tsx`. Steps:

1. Copy the JSX of `RouterConfig` from `<DESIGN>/screens/Technical.jsx:670-861` into a new TS/TSX file. Keep `className`s, attribute order, inline styles, conditional rendering identical.
2. Replace the inline `ROUTER_RULES` constant with `useEffect` + `getRouterRules()` from `api.ts`.
3. State: `rules` (from API), `filter`, `search`, `openId`, `creating`, `toast`. Same shape as design.
4. Top summary cards (`router-summary`):
   - Active rules: `rules.filter(r => r.enabled).length`.
   - Avg cost / conv: derive from trace data via `/v1/metrics/overview.avg_cost_per_conv` if available, else `—`.
   - Routed share: always `100%` since there's exactly one default rule for v1; fall back to design's existing math when more rules exist.
   - Distribution bar: render one `seg` for the default rule (full width) using the model's color class (`seg-{model_family}`). When more rules ship later, the math already works.
5. Filter pills (`pill` buttons) and search wired to the same state shape — for v1 they don't filter much (one row), but UI honors the design.
6. Rules table (`card > rule-head + rule-row`): render one row from `rules`. Click → `setOpenId(r.id)`.
7. `+ Add routing rule` button: `disabled` attribute + `title="Coming soon — manual rules deferred to a future phase"`.

### T2 — Port `RuleDrawer.tsx` with UniRoute adapter
Create `ui/src/screens/Technical/RuleDrawer.tsx`:

1. Copy the JSX of `RuleDrawer` from `<DESIGN>:862-944` verbatim. Same `sheet`, `sheet-head`, `sheet-tabs`, `sheet-body`, `sheet-foot`, three tabs (Overview / Samples / History).
2. Add a content adapter:

```tsx
function useDrawerContent(rule: RouterRule): DrawerContent {
  const [config, setConfig] = useState<RouterConfigView | null>(null);
  const [health, setHealth] = useState<RouterHealth | null>(null);
  const [candidates, setCandidates] = useState<RouterCandidate[]>([]);

  useEffect(() => {
    if (rule.id !== "uniroute_default") return;
    void Promise.all([getRouterConfig(), getRouterHealth(), getRouterCandidates()])
      .then(([c, h, q]) => { setConfig(c); setHealth(h); setCandidates(q); });
  }, [rule.id]);

  if (rule.id !== "uniroute_default") {
    return staticContentFromRule(rule);
  }
  // UniRoute-specific:
  return {
    overview: <UniRouteOverview config={config} health={health}/>,
    samples: <UniRouteSamples health={health}/>,
    history: <UniRouteHistory candidates={candidates}/>,
  };
}
```

`UniRouteOverview` renders inside the existing `.sheet-section` + `.meta-grid` chrome — no new classes. Content:

- A `<p>` rationale: "Trained router. Picks per request based on prompt embedding cluster + per-model expected error."
- `meta-grid` with `meta-row` for: K, Model count, λ (cost weight), Last fit, Drift score, Drift baseline, Trace count since last fit, Current avg error, Current win rate. All as text. Cold-start: each row shows `—` and the rationale changes to "Cold-start: no config fitted yet. Claude Code will propose one once enough traffic accumulates."

`UniRouteSamples` reuses `.sample` chrome from the design. Renders top 3 prompts representative of the most active clusters from `health.cluster_distribution`. Empty state uses the design's existing "No matched samples yet" copy.

`UniRouteHistory` reuses the design's history tab grid. Each entry is a `Lesson(kind="router_config")` summary: `{when: relative_time(lesson.promoted_at), what: lesson.voice}`. Click on a row → navigate to `/lesson/${lesson.id}` (existing route). Pending candidates from `/v1/router/candidates` show at the top with a `Tag kind="warn"` "Awaiting review" → link to `/review/${lesson_id}`.

### T3 — Port `NewRuleModal.tsx` (disabled state)
Create `ui/src/screens/Technical/NewRuleModal.tsx`. Copy the JSX verbatim. Then:

- Replace the submit button with a disabled version: `<button className="btn primary" disabled>Coming soon</button>`.
- Add a small `<p className="dim">` under the form: "Manual routing rules are deferred to a future phase. The default UniRoute rule is in effect."
- Inputs stay rendered (so the chrome looks identical) but their values aren't wired to anything.

This satisfies the "preserve visual panels" rule from `feedback_ui_panels.md` while being honest about the gap.

### T4 — Wire into `/technical/router`
Edit `ui/src/screens/Technical/index.tsx` (or the parent Technical layout):
- Replace the existing `/router` route mock content with `<RouterConfig/>`.
- Confirm the existing tab/menu marker for "Router config" still highlights when on this route.

### T5 — Backend endpoints
Add to `runtime/server.py`:

```python
@app.get("/router/rules", response_model=list[RouterRuleView])
async def list_router_rules() -> list[RouterRuleView]:
    """Synthesized rules list. v1: one row representing UniRoute."""
    return [_build_uniroute_default_rule()]


@app.get("/router/candidates", response_model=list[RouterCandidateView])
async def list_router_candidates() -> list[RouterCandidateView]:
    """Pending router_config Lessons (status=awaiting_review)."""
    lessons = read_lessons(kind="router_config", status="awaiting_review")
    return [_lesson_to_candidate_view(l) for l in lessons]


@app.get("/router/health", response_model=RouterHealthView)
async def get_router_health() -> RouterHealthView:
    from router.feedback.health import compute_router_health
    return RouterHealthView(**compute_router_health().to_dict())
```

`_build_uniroute_default_rule` reads `/router/config` payload + last lesson + a few sample prompts (best-effort). Cold-start path returns the row with `enabled: true`, `share: 1.0`, but rationale and samples reflect the cold-start state.

Add Pydantic models (`RouterRuleView`, `RouterCandidateView`, `RouterHealthView`) to `runtime/types.py`.

Update `runtime/server.py` docstring endpoint listing.

### T6 — Backend proxy channels
Add `backend/channels/router/{rules,candidates,health}.ts` mirroring the existing `config.ts` pattern (auth middleware, fetch, passthrough). Register in main backend router.

### T7 — Trace drawer extension
Edit `ui/src/screens/Technical/TraceDrawer.tsx` (the existing component from P15.1):

- After the existing stages section, when any stage has `routing_decision`, add a `<div className="sheet-section"><h3>Routing decision</h3>...` block.
- Inside, use the same `.meta-grid` / `.meta-row` chrome the AgentSheet uses — no new classes.
- Show: selected_model, cluster_id, cluster_probabilities (top-3 only as `<span className="mono">c{i}: {p.toFixed(2)}</span>`), expected_error, cost_adjusted_score, reasoning (multi-line `<pre className="mono">`). Cold-start path shows `fallback_reason` instead.

This satisfies the roadmap's "UI Trace drawer can display the reasoning" DoD.

### T8 — Tests

`ui/src/screens/Technical/__tests__/RouterConfig.test.tsx`:
- `renders cold-start state`: `getRouterRules` returns the synthesized default with `cold_start` markers; component shows the row with `share: 100%` and rationale "Cold-start: ...".
- `renders fitted state`: rules row + summary cards populated; click → drawer opens with UniRoute content.
- `add button is disabled with tooltip`: presence of `disabled` + `title` attribute.
- `filter pills don't break with one row`: switching filters doesn't crash; counts render correctly.

`RuleDrawer.test.tsx`:
- `default rule shows UniRoute overview`: meta-grid contains K, Model count, λ rows.
- `default rule history links to /lesson/<id>`: click on a history entry navigates correctly.
- `cold-start drawer shows em-dashes`: `—` for K, drift, etc., when config is null.

Visual smoke (Playwright, `e2e/router_config.spec.ts`):
- Capture screenshot of `/technical/router` when fitted; pixel-diff against `/tmp/design_unpack/.../scraps/sketch-*.napkin` (or the design HTML rendered in a headless browser).
- Capture screenshot of the drawer open; same comparison.

Tolerance: zero diff on shape/spacing/colors per roadmap DoD. Allow text differences (UniRoute meta vs design's mock text).

### T9 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
# Backend + runtime up
python -m uvicorn runtime.server:app --port 8001 &
cd backend && OPENTRACY_API_KEY=dev npm run start &
cd ../ui && npm run dev   # :5174

# Manual:
# 1. Open http://localhost:5174 → switch to Technical view → Router config.
# 2. Verify: page renders with one rule "UniRoute (default)".
# 3. Click into the rule → drawer opens with three tabs:
#    - Overview: meta-grid showing K, drift, λ, last fit, etc.
#    - Sample matches: top prompts from active clusters (or empty state).
#    - History: list of router_config Lessons.
# 4. Click a history entry → navigate to /lesson/<id>.
# 5. Click "+ Add routing rule" → modal with disabled submit + "Coming soon".
# 6. Open a recent trace → drawer's Routing decision section shows the cluster + reasoning.

# Pixel diff (manual):
# Render the design HTML and the implemented page side-by-side in a browser; eyeball spacing.
# Or run e2e/router_config.spec.ts which does this programmatically.

# Tests
cd ui && npm test
```

## Acceptance criteria (DoD)

1. **Page renders verbatim:** `/technical/router` loads with the design's exact layout — same `router-summary`, `trace-toolbar`, `rule-head`, `rule-row`, `card`, `add-btn` chrome. CSS classes match design 1:1.
2. **Default-rule row reflects UniRoute state:** when `router_config_current` exists, the rule shows model "uniroute" with non-zero share. Cold-start path shows the same row with rationale "Cold-start: no config fitted yet."
3. **Drawer renders UniRoute internals on the default row:** Overview tab has K / Model count / λ / drift / last fit; Samples tab has top prompts; History tab lists router_config Lessons that link to `/lesson/<id>`.
4. **Trace drawer shows `routing_decision`** when present in the trace JSON: selected_model, cluster_id, top-3 cluster_probs, reasoning. Cold-start traces show fallback_reason.
5. **`+ Add routing rule` is disabled** with a "Coming soon" tooltip; the modal opens but submission is blocked, and a "deferred" copy explains why.
6. **No new UI components introduced** outside the design — drawer content adapter uses existing `.sheet-section`, `.meta-grid`, `.meta-row`, `.sample`, `.tag`, `.btn` classes only.
7. **Pixel-diff vs design = zero on shape/spacing/colors.** Text content can differ (UniRoute meta vs design mocks); CSS / layout cannot.
8. **Three new endpoints respond:** `GET /v1/router/rules`, `GET /v1/router/candidates`, `GET /v1/router/health` return their documented shapes; cold-start responses are 200 with empty/`—` payloads, never 404 or 500.
9. **Candidate queue links to Review:** entries from `/v1/router/candidates` route to `/review/<lesson_id>` and the Review screen handles `kind="router_config"` lessons (existing Review screen reads kind generically — verify).
10. No regressions: `cd ui && npm run build` succeeds; existing UI tests pass.

## Risks / open questions

- **Design ↔ backend paradigm mismatch.** The design is rule-based; UniRoute is embedding-based. We bridge by mapping UniRoute to "the default rule". If the user wants a richer router UI (heatmap, λ slider, candidate queue as separate panels), that's a follow-up phase, not P15.3.10. Document in roadmap follow-ups.
- **Manual rules deferral honesty.** `+ Add routing rule` modal renders with disabled submit. Some operators may find this confusing — a non-functional control is a gentler signal than hiding the button entirely. Tradeoff: honesty vs minimalism. Going with disabled-with-tooltip for "preserve visual panels" alignment.
- **Sample prompts for the default-rule drawer.** Best-effort: pull a representative prompt per top cluster from recent traces. If trace count is low / clusters thin, the Samples tab may render fewer than 3 prompts or fall back to empty state.
- **History tab scaling.** As retrains accumulate (months in), the History tab gets long. The design has no pagination. Acceptable for v1 (<100 retrains/year at the proposed cadence); add pagination only if it becomes a real problem.
- **Cold-start visual consistency.** The default-rule row needs to look right when no config exists yet — `cost: —`, `share: 100%` (since it's the only thing in effect), enabled toggle "on" but locked. Ensure cold-start data flows produce a presentable row, not blank fields.
- **Backend `/v1/router/rules` is synthesized.** Future when rules engine ships, this endpoint backs by a real rules store. Today it's a single computed row. Document in the endpoint docstring so future-us knows the v1 shape.
- **Pixel-diff tolerance.** "Zero diff on shape/spacing/colors" is a strong claim. Subpixel rendering, font fallbacks, and sub-pixel shadow positions can produce false diffs. Use Playwright's `toHaveScreenshot` with a reasonable threshold (e.g., 0.5%) on color delta; flag anything bigger as a real diff.
- **Roadmap fabrications acknowledged.** Original roadmap said "Ψ heatmap, cluster distribution chart, λ slider, candidate queue". None of those are in the design. This PLAN drops them. ROADMAP_P15.3.md needs a follow-up edit to reflect reality once this phase lands.
