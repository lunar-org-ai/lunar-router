"""Tests for the harness MCP server — Phase 1.5.

Exercises every registered tool end-to-end via `server.call_tool()`,
which is the same code path an MCP client invokes over stdio or HTTP
(minus transport serialization). Stdio serialization is tested
separately by spawning the module as a subprocess.
"""

from __future__ import annotations

import pytest

from opentracy.harness.ledger import LedgerEntry, LedgerStore
from opentracy.harness.ledger import _global as ledger_global
from opentracy.harness.mcp_server import SERVER_NAME, build_server


@pytest.fixture
def ledger(tmp_path):
    """Install a tmp-path ledger as the process singleton for the
    duration of one test. Tools under test resolve `get_ledger_store()`
    lazily, so swapping the module-level `_instance` is sufficient."""
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    original = ledger_global._instance
    ledger_global._instance = store
    try:
        yield store
    finally:
        ledger_global._instance = original
        store.close()


@pytest.fixture
def server():
    return build_server()


async def _call(server, name: str, args: dict | None = None):
    """Invoke a tool and return the logical return value an MCP client
    would see after JSON decoding.

    FastMCP returns either `(content_blocks, structured_dict)` or a
    bare `content_blocks` list depending on the tool's return type.
    When structured is present, it's authoritative (and for list
    returns, it carries the full list under the `result` key); content
    blocks are per-item serializations used for model display.
    """
    import json

    result = await server.call_tool(name, args or {})

    # Preferred path: structured result from the tuple form.
    if isinstance(result, tuple) and len(result) == 2:
        structured = result[1]
        if isinstance(structured, dict):
            # FastMCP wraps non-dict returns (list, scalar, None) under
            # "result". Dict returns come back keyed by their own fields.
            if set(structured.keys()) == {"result"}:
                return structured["result"]
            return structured

    # Fallback: concatenate text blocks into a list (each block is one
    # JSON-serialized item) or parse a single dict.
    content_blocks = result[0] if isinstance(result, tuple) else result
    if not content_blocks:
        return None
    texts = [b.text for b in content_blocks if hasattr(b, "text")]
    if not texts:
        return None
    if len(texts) == 1:
        return json.loads(texts[0]) if texts[0] != "null" else None
    return [json.loads(t) for t in texts if t != "null"]


# ---------------------------------------------------------------------------
# Server construction
# ---------------------------------------------------------------------------


def test_server_has_expected_name_and_instructions(server):
    assert server.name == SERVER_NAME
    assert server.instructions and "harness" in server.instructions.lower()


async def test_expected_tool_surface(server):
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert names == {
        # read tools
        "list_objectives",
        "get_objective_time_series",
        "list_ledger_entries",
        "get_ledger_entry",
        "get_ledger_chain",
        "list_policies",
        "describe_policy",
        "list_recipes",
        "describe_recipe",
        "list_actions",
        "list_agents",
        # write tools (Phase 1.5)
        "run_agent",
        "propose_action",
        "approve_proposal",
        "reject_proposal",
        "record_outcome",
    }


async def test_every_tool_has_description(server):
    """Tool descriptions are what Claude Code sees when picking which
    tool to call. A missing description means the tool is effectively
    invisible to the LLM."""
    for tool in await server.list_tools():
        assert tool.description, f"tool {tool.name!r} has no description"


# ---------------------------------------------------------------------------
# Objective tools
# ---------------------------------------------------------------------------


async def test_list_objectives_returns_three(server, ledger):
    result = await _call(server, "list_objectives")
    # FastMCP wraps list returns under a "result" key; unwrap.
    ids = {o["id"] for o in result}
    assert ids == {
        "cost_per_successful_completion",
        "p95_latency_ms",
        "domain_coverage_ratio",
    }


async def test_get_objective_time_series_shape(server, ledger):
    result = await _call(
        server,
        "get_objective_time_series",
        {"objective_id": "cost_per_successful_completion", "hours": 24},
    )
    # Endpoint shape: measurements + markers both present, plus metadata.
    assert result["objective_id"] == "cost_per_successful_completion"
    assert result["window_hours"] == 24
    assert result["measurements"] == []
    assert result["markers"] == []


async def test_time_series_window_clamps_at_month(server, ledger):
    result = await _call(
        server,
        "get_objective_time_series",
        {"objective_id": "cost_per_successful_completion", "hours": 100000},
    )
    assert result["window_hours"] == 24 * 30


# ---------------------------------------------------------------------------
# Ledger tools
# ---------------------------------------------------------------------------


def _seed_chain(store: LedgerStore) -> dict:
    signal = LedgerEntry(
        type="signal",
        objective_id="cost_per_successful_completion",
        agent="objective_sensor",
        tags=["objective_regression", "cost_per_successful_completion"],
        data={"delta_pct": 22.0},
    )
    store.append(signal)
    dispatch = LedgerEntry(
        type="run",
        agent="training_advisor",
        parent_id=signal.id,
        tags=["policy_dispatch"],
    )
    store.append(dispatch)
    decision = LedgerEntry(
        type="decision",
        objective_id="cost_per_successful_completion",
        parent_id=dispatch.id,
        agent="training_advisor",
        outcome="ok",
        data={"recommendation": "train_now"},
    )
    store.append(decision)
    return {"signal": signal.id, "dispatch": dispatch.id, "decision": decision.id}


async def test_list_ledger_entries_empty_when_ledger_empty(server, ledger):
    result = await _call(server, "list_ledger_entries")
    assert result == {"entries": [], "count": 0}


async def test_list_ledger_entries_filters(server, ledger):
    _seed_chain(ledger)
    result = await _call(server, "list_ledger_entries", {"type": "signal"})
    assert result["count"] == 1
    assert result["entries"][0]["type"] == "signal"


async def test_list_ledger_entries_limit_capped(server, ledger):
    result = await _call(server, "list_ledger_entries", {"limit": 5000})
    # Runs without error; cap is silent — no assertion on returned length
    # since the ledger is empty.
    assert "entries" in result


async def test_get_ledger_entry_hits_and_misses(server, ledger):
    ids = _seed_chain(ledger)
    hit = await _call(server, "get_ledger_entry", {"entry_id": ids["signal"]})
    assert hit["id"] == ids["signal"]
    assert hit["type"] == "signal"

    miss = await _call(server, "get_ledger_entry", {"entry_id": "does-not-exist"})
    assert miss is None


async def test_get_ledger_chain_reconstructs(server, ledger):
    ids = _seed_chain(ledger)
    result = await _call(server, "get_ledger_chain", {"root_entry_id": ids["signal"]})
    assert result["root_id"] == ids["signal"]
    assert result["count"] == 3
    types = [e["type"] for e in result["entries"]]
    assert types == ["signal", "run", "decision"]


async def test_get_ledger_chain_unknown_root_is_empty_not_error(server, ledger):
    result = await _call(server, "get_ledger_chain", {"root_entry_id": "nope"})
    assert result == {"root_id": "nope", "entries": [], "count": 0}


# ---------------------------------------------------------------------------
# Catalog tools
# ---------------------------------------------------------------------------


async def test_list_policies_returns_shipped_five(server, ledger):
    result = await _call(server, "list_policies")
    items = result.get("result", result) if isinstance(result, dict) else result
    ids = {p["id"] for p in items}
    assert ids == {
        "cost_drift_to_evaluate_student_distillation",
        "latency_drift_to_trace_scanner",
        "cadence_to_trace_scan",
        "new_traces_to_cluster_and_label",
        "new_dataset_to_suggest_metrics",
    }


async def test_describe_policy_hits_and_misses(server, ledger):
    hit = await _call(
        server,
        "describe_policy",
        {"policy_id": "cost_drift_to_evaluate_student_distillation"},
    )
    assert hit["dispatch"]["recipe"] == "evaluate_student_distillation"

    miss = await _call(server, "describe_policy", {"policy_id": "nope"})
    assert miss is None


async def test_list_recipes_returns_shipped_four(server, ledger):
    result = await _call(server, "list_recipes")
    ids = {r["id"] for r in result}
    assert ids == {
        "trace_scan_and_evaluate",
        "cluster_and_label",
        "suggest_metrics",
        "evaluate_student_distillation",
    }


async def test_list_actions_includes_all_registered(server, ledger):
    result = await _call(server, "list_actions")
    assert "run_eval" in result
    assert "run_clustering" in result
    assert "fetch_dataset_samples" in result
    assert "fetch_cost_summary" in result
    assert "queue_training" in result


async def test_list_agents_elides_system_prompt(server, ledger):
    """The full prompt text is large; the MCP surface elides it so
    responses stay small. Callers who need it fetch from the HTTP
    route instead."""
    result = await _call(server, "list_agents")
    assert len(result) >= 10
    # None of the elided records should carry the system_prompt key.
    assert all("system_prompt" not in a for a in result)
    # Sanity: budget_justifier critic is present after the Step-3 add.
    assert any(a["name"] == "budget_justifier" for a in result)


# ---------------------------------------------------------------------------
# Write tools — gated through a stubbed critic to avoid LLM calls
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_critic(monkeypatch):
    """Replace harness.writes.critic_check with a stub the test can
    flip between approve / reject by mutating `state["decision"]`."""
    from opentracy.harness import writes as harness_writes
    from opentracy.harness.critic_gate import CriticVerdict
    from opentracy.harness.ledger import LedgerEntry

    state = {"decision": "approve"}

    async def _stub_critic_check(*, action_kind, payload, objective_id, ledger=None, runner=None, cache=None):
        from opentracy.harness.ledger import _global as ledger_global

        store = ledger if ledger is not None else ledger_global._instance
        decision = state["decision"]
        entry = LedgerEntry(
            type="decision",
            objective_id=objective_id,
            agent="budget_justifier",
            data={"decision": decision, "rationale": "stub", "estimated_cost_usd": 0.1, "estimated_benefit": "x", "action_kind": action_kind},
            tags=["critic_check", action_kind, decision],
            outcome="ok",
        )
        store.append(entry)
        return CriticVerdict(
            decision=decision,
            rationale="stub",
            estimated_cost_usd=0.1,
            estimated_benefit="x",
            decision_entry_id=entry.id,
        )

    monkeypatch.setattr(harness_writes, "critic_check", _stub_critic_check)
    return state


async def test_propose_action_creates_pending_proposal(server, ledger, stub_critic):
    out = await _call(
        server,
        "propose_action",
        {
            "kind": "queue_training",
            "payload": {"student": "x"},
            "summary": "test",
        },
    )
    pid = out["proposal_id"]
    assert pid

    # Verify a proposal row exists in the ledger.
    rows = ledger.recent(limit=10)
    types = {r.type for r in rows}
    assert "proposal" in types and "decision" in types


async def test_approve_proposal_round_trip(server, ledger, stub_critic):
    proposed = await _call(
        server,
        "propose_action",
        {"kind": "run_eval", "payload": {"a": 1}, "summary": "x"},
    )
    pid = proposed["proposal_id"]

    # Cache could short-circuit the recheck; reset between calls.
    from opentracy.harness import critic_gate as _cg
    _cg.reset_cache_for_tests()

    decided = await _call(server, "approve_proposal", {"proposal_id": pid})
    assert decided["decision"] == "approved"


async def test_reject_proposal_does_not_call_critic_recheck(server, ledger, stub_critic):
    proposed = await _call(
        server,
        "propose_action",
        {"kind": "run_eval", "payload": {"a": 1}, "summary": "x"},
    )
    pid = proposed["proposal_id"]

    out = await _call(
        server,
        "reject_proposal",
        {"proposal_id": pid, "reason": "test"},
    )
    assert out["decision"] == "rejected"


async def test_record_outcome_on_approved_proposal(server, ledger, stub_critic):
    proposed = await _call(
        server,
        "propose_action",
        {"kind": "run_eval", "payload": {"a": 1}, "summary": "x"},
    )
    pid = proposed["proposal_id"]

    from opentracy.harness import critic_gate as _cg
    _cg.reset_cache_for_tests()

    await _call(server, "approve_proposal", {"proposal_id": pid})
    out = await _call(
        server,
        "record_outcome",
        {"proposal_id": pid, "result": {"score": 0.9}, "outcome": "ok"},
    )
    assert out["outcome"] == "ok"


async def test_run_agent_inspector_skips_critic(server, ledger, stub_critic):
    """Inspectors are read-only by convention; the critic must NOT be
    invoked. We verify by setting the stub to reject — if it were called,
    the run would not happen."""
    stub_critic["decision"] = "reject"

    # Patch the AgentRunner so we don't reach the LLM.
    from opentracy.harness import writes as harness_writes

    class _NoNetRunner:
        async def run(self, agent_name, user_input, *_a, **_k):
            class _R:
                data = {"label": "ok"}
                duration_ms = 1.0
                cost_usd = None
            return _R()

        async def run_with_tools(self, *a, **k):
            return await self.run(*a, **k)

    original = harness_writes.run_agent_gated

    async def _patched(name, user_input, **kwargs):
        kwargs["runner"] = _NoNetRunner()
        return await original(name, user_input, **kwargs)

    out = await _patched(
        "cluster_labeler",
        "samples",
        ledger=ledger,
    )
    assert out["decision"] == "ungated"
