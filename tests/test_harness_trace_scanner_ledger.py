"""Ledger wiring tests for TraceScanner.

These tests exercise the observability contract added in Phase 1 Step 3:
every scheduled scan must produce a `run` ledger entry plus one
`observation` per detected issue, and the chain must reconstruct from
the scan id.

Network I/O is avoided by monkey-patching the two methods that talk to
the Go engine (`_fetch_traces`, `_generate_eval_cases`). Agent-layer
checks are skipped organically: the synthetic traces use an empty
`input_text`, so `content_traces` is empty and `_run_agent_checks` is
never called.
"""

from __future__ import annotations

import pytest

from opentracy.harness.ledger import LedgerStore
from opentracy.harness.memory_store import MemoryStore
from opentracy.harness.trace_scanner import (
    ISSUE_TYPE_TO_OBJECTIVE,
    TraceScanner,
)


def _make_trace(
    idx: int,
    latency_ms: float,
    cost_usd: float,
    model: str = "gpt-4o-mini",
) -> dict:
    """Build a synthetic trace that will NOT trigger incomplete_response,
    format_violation, or the LLM agent layer."""
    return {
        "request_id": f"req-{idx}",
        "selected_model": model,
        "latency_ms": latency_ms,
        "total_cost_usd": cost_usd,
        "tokens_out": 50,
        "output_text": "A reasonable response containing enough characters.",
        "input_text": "",  # empty → no agent check, no format_violation
        "is_error": 0,
    }


def _make_population_with_spike() -> list[dict]:
    """9 normal traces + 1 latency+cost outlier. Yields exactly one
    latency_spike and one cost_anomaly issue."""
    traces = [_make_trace(i, latency_ms=100, cost_usd=0.001) for i in range(9)]
    traces.append(_make_trace(9, latency_ms=1000, cost_usd=0.10))
    return traces


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


@pytest.fixture
def memory(tmp_path) -> MemoryStore:
    return MemoryStore(memory_dir=tmp_path / "memory")


@pytest.fixture
def scanner(ledger, memory) -> TraceScanner:
    s = TraceScanner(memory_store=memory, ledger=ledger)

    async def _noop_eval(issues, scan_id):
        return []

    s._generate_eval_cases = _noop_eval  # avoid AgentRunner network call
    return s


def _install_fake_traces(scanner: TraceScanner, traces: list[dict]) -> None:
    async def _fake_fetch(days=7, limit=100):
        return traces

    scanner._fetch_traces = _fake_fetch


# ---------------------------------------------------------------------------
# Happy path — run + observations + summary all land in the ledger.
# ---------------------------------------------------------------------------


async def test_scan_emits_run_entry_with_scan_id(scanner, ledger):
    _install_fake_traces(scanner, _make_population_with_spike())

    await scanner.scan(scan_id="scan-001", days=1, limit=10)

    run = ledger.get("scan-001")
    assert run is not None
    assert run.type == "run"
    assert run.agent == "trace_scanner"
    assert run.parameters_in == {"days": 1, "limit": 10}
    assert "scheduler_tick" in run.tags


async def test_scan_emits_one_observation_per_issue(scanner, ledger):
    _install_fake_traces(scanner, _make_population_with_spike())

    issues = await scanner.scan(scan_id="scan-002", days=1, limit=10)

    # Expect one latency_spike + one cost_anomaly → 2 issues
    assert len(issues) == 2

    # Each observation must be parented to the run and carry objective_id.
    chain = ledger.chain("scan-002")
    observations = [e for e in chain if e.type == "observation"]
    issue_obs = [o for o in observations if "scan_summary" not in o.tags]
    assert len(issue_obs) == 2

    objectives = {o.objective_id for o in issue_obs}
    assert objectives == {"p95_latency_ms", "cost_per_successful_completion"}

    for obs in issue_obs:
        assert obs.parent_id == "scan-002"
        assert obs.agent == "trace_scanner"
        assert obs.data["issue_type"] in ISSUE_TYPE_TO_OBJECTIVE


async def test_scan_emits_summary_observation_with_ok_outcome(scanner, ledger):
    _install_fake_traces(scanner, _make_population_with_spike())

    await scanner.scan(scan_id="scan-003", days=1, limit=10)

    chain = ledger.chain("scan-003")
    summaries = [e for e in chain if "scan_summary" in e.tags]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.outcome == "ok"
    assert summary.data["traces_scanned"] == 10
    assert summary.data["issues_found"] == 2
    assert summary.duration_ms is not None and summary.duration_ms >= 0


async def test_chain_reconstructs_full_scan(scanner, ledger):
    _install_fake_traces(scanner, _make_population_with_spike())

    await scanner.scan(scan_id="scan-004", days=1, limit=10)

    chain = ledger.chain("scan-004")
    types = [e.type for e in chain]
    # 1 run + 2 issue observations + 1 summary observation = 4 entries
    assert types.count("run") == 1
    assert types.count("observation") == 3
    assert chain[0].id == "scan-004"  # BFS root first


# ---------------------------------------------------------------------------
# Empty trace population — contract still holds: run + summary written.
# ---------------------------------------------------------------------------


async def test_empty_scan_still_emits_run_and_summary(scanner, ledger):
    _install_fake_traces(scanner, [])

    issues = await scanner.scan(scan_id="scan-empty", days=1, limit=10)

    assert issues == []
    chain = ledger.chain("scan-empty")
    assert len(chain) == 2  # run + summary
    summary = next(e for e in chain if "scan_summary" in e.tags)
    assert summary.outcome == "ok"
    assert summary.data["traces_scanned"] == 0
    assert summary.data["issues_found"] == 0


# ---------------------------------------------------------------------------
# Failure path — exception in _fetch_traces must still leave a trail.
# ---------------------------------------------------------------------------


async def test_scan_failure_emits_failed_observation(scanner, ledger):
    async def _boom(days=7, limit=100):
        raise RuntimeError("clickhouse unavailable")

    scanner._fetch_traces = _boom

    with pytest.raises(RuntimeError, match="clickhouse unavailable"):
        await scanner.scan(scan_id="scan-fail", days=1, limit=10)

    chain = ledger.chain("scan-fail")
    failures = [e for e in chain if e.outcome == "failed"]
    assert len(failures) == 1
    assert "scan_failed" in failures[0].tags
    assert "clickhouse unavailable" in failures[0].data["error"]


# ---------------------------------------------------------------------------
# Backwards compatibility — TraceScanner() with no ledger stays silent.
# ---------------------------------------------------------------------------


async def test_scanner_without_ledger_emits_nothing(memory, ledger):
    """A scanner constructed without `ledger=...` must not touch any
    ledger — existing callsites and the rebrand tests rely on this."""
    s = TraceScanner(memory_store=memory)  # no ledger arg

    async def _noop_eval(issues, scan_id):
        return []

    s._generate_eval_cases = _noop_eval
    _install_fake_traces(s, _make_population_with_spike())

    await s.scan(scan_id="scan-noledger", days=1, limit=10)

    # The fixture's ledger is a fresh separate instance — should be empty.
    assert ledger.get("scan-noledger") is None
    assert ledger.recent(limit=10) == []
