"""Operator tools — concrete action implementations for the OperatorLoop.

Each tool returns a structured dict that the Operator persists as
``outcome_json`` in the ``operator_decisions`` table. Tools must be
defensive: the operator runs unattended, so any failure should degrade
gracefully to a heuristic fallback or a structured error.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Optional
from opentracy._env import env

logger = logging.getLogger(__name__)

_AGENT_DIR = Path(__file__).parent / "agents"


# ---------------------------------------------------------------------------
# cluster_traces
# ---------------------------------------------------------------------------

async def cluster_traces(limit: int = 1000, days: int = 7) -> dict[str, Any]:
    """Run the existing clustering pipeline and return new cluster ids.

    Args:
        limit: Upper bound on traces to consider. (Informational — pipeline
            reads from ClickHouse with a time window.)
        days: Lookback window in days.

    Returns:
        {"run_id": str, "cluster_ids": [int], "num_clusters": int,
         "trace_count": int, "skipped": bool}
    """
    from ..clustering.pipeline import ClusteringPipeline

    engine_url = env("ENGINE_URL", "http://localhost:8080")
    pipeline = ClusteringPipeline(strategy="auto", engine_url=engine_url)
    try:
        result = await pipeline.run(days=days, min_traces=50)
    except Exception as exc:
        logger.exception("clustering pipeline failed")
        return {"skipped": True, "error": str(exc), "cluster_ids": []}

    cluster_ids = [int(ds.cluster_id) for ds in result.datasets]
    return {
        "run_id": result.version.run_id,
        "num_clusters": result.version.num_clusters,
        "trace_count": result.version.trace_count,
        "cluster_ids": cluster_ids,
        "qualified": sum(1 for d in result.datasets if d.status == "qualified"),
        "limit": limit,
        "days": days,
    }


# ---------------------------------------------------------------------------
# propose_dataset_from_cluster
# ---------------------------------------------------------------------------

async def propose_dataset_from_cluster(
    tenant_id: str,
    run_id: str,
    cluster_id: int,
) -> dict[str, Any]:
    """Build a ``pending_curation`` dataset from a cluster.

    Flow: read cluster members + label → call cluster_labeler + dataset_summarizer
    prompts via MistralClient → pick 3 representative + 2 edge samples →
    ``create_dataset(status='pending_curation', source='auto')`` →
    ``add_samples(status='suggested')``.

    Falls back to heuristic naming + random sampling if the LLM path fails.
    """
    from ..datasets import repository as ds_repo
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None or not run_id or cluster_id < 0:
        return {"skipped": True, "reason": "missing clickhouse/run/cluster"}

    # Fetch cluster label/description (best-effort).
    cluster_meta: dict[str, Any] = {}
    try:
        rr = client.query(
            "SELECT domain_label, short_description, sample_prompts, trace_count "
            "FROM cluster_datasets WHERE run_id = {rid:String} AND cluster_id = {cid:Int32} LIMIT 1",
            parameters={"rid": run_id, "cid": int(cluster_id)},
        )
        if rr.result_rows:
            row = rr.result_rows[0]
            cluster_meta = {
                "domain_label": row[0] or "",
                "short_description": row[1] or "",
                "sample_prompts": _safe_json_list(row[2]),
                "trace_count": int(row[3] or 0),
            }
    except Exception as exc:
        logger.debug("cluster_datasets read failed: %s", exc)

    # Fetch cluster member traces for sample selection.
    members: list[dict[str, Any]] = []
    try:
        rr = client.query(
            "SELECT tcm.request_id, t.input_text, t.output_text "
            "FROM trace_cluster_map tcm "
            "LEFT JOIN llm_traces t ON t.request_id = tcm.request_id "
            "WHERE tcm.run_id = {rid:String} AND tcm.cluster_id = {cid:Int32} "
            "LIMIT 200",
            parameters={"rid": run_id, "cid": int(cluster_id)},
        )
        for row in rr.result_rows:
            members.append(
                {
                    "trace_id": str(row[0]),
                    "input": row[1] or "",
                    "output": row[2] or "",
                }
            )
    except Exception as exc:
        logger.debug("trace_cluster_map read failed: %s", exc)

    if not members:
        return {"skipped": True, "reason": "no cluster members found"}

    # Attempt LLM label/summary, fall back to heuristic on failure.
    sample_prompts = cluster_meta.get("sample_prompts") or [
        m["input"] for m in members[:8]
    ]
    label_json, summary_json, llm_used = await _llm_label_and_summary(
        sample_prompts, cluster_meta
    )

    title = (
        (summary_json.get("title") if isinstance(summary_json, dict) else None)
        or (label_json.get("domain_label") if isinstance(label_json, dict) else None)
        or cluster_meta.get("domain_label")
        or _heuristic_title(members)
    )
    description = (
        (summary_json.get("summary") if isinstance(summary_json, dict) else None)
        or cluster_meta.get("short_description")
        or ""
    )

    rationale_payload = {
        "source": "operator",
        "cluster_ref": {"run_id": run_id, "cluster_id": int(cluster_id)},
        "llm_used": llm_used,
        "label": label_json if isinstance(label_json, dict) else {},
        "summary": summary_json if isinstance(summary_json, dict) else {},
        "trace_count": len(members),
    }

    # Pick 3 representative + 2 edge samples.
    picks = _pick_samples(members, representative=3, edge=2)

    try:
        ds = ds_repo.create_dataset(
            tenant_id,
            name=f"[Auto] {title}",
            description=description[:500],
            source="auto",
            status="pending_curation",
            rationale=json.dumps(rationale_payload, ensure_ascii=False),
        )
    except Exception as exc:
        logger.exception("create_dataset failed")
        return {"skipped": True, "reason": f"create_dataset: {exc}"}

    samples = [
        {
            "input": p["input"],
            "expected_output": p["output"],
            "trace_id": p["trace_id"],
            "metadata": {"role": p.get("role", "representative")},
            "status": "suggested",
        }
        for p in picks
    ]
    added = ds_repo.add_samples(tenant_id, ds["dataset_id"], samples)

    return {
        "dataset_id": ds["dataset_id"],
        "title": title,
        "samples_added": added,
        "llm_used": llm_used,
        "cluster_ref": rationale_payload["cluster_ref"],
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_json_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return [str(x) for x in val] if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _heuristic_title(members: list[dict[str, Any]]) -> str:
    if not members:
        return "Uncategorised cluster"
    first = (members[0].get("input") or "").strip().splitlines()[0]
    first = re.sub(r"\s+", " ", first)
    return (first[:60] + "…") if len(first) > 60 else (first or "Uncategorised cluster")


def _pick_samples(
    members: list[dict[str, Any]],
    *,
    representative: int = 3,
    edge: int = 2,
) -> list[dict[str, Any]]:
    """Pick a small, diverse sample set. Representative = shortest inputs,
    edge = longest. Deterministic-ish via length sort + random tiebreak.
    """
    if not members:
        return []
    rng = random.Random(42)
    sorted_by_len = sorted(
        members, key=lambda m: (len(m.get("input") or ""), rng.random())
    )
    picks: list[dict[str, Any]] = []
    for m in sorted_by_len[:representative]:
        picks.append({**m, "role": "representative"})
    for m in sorted_by_len[-edge:]:
        if any(p["trace_id"] == m["trace_id"] for p in picks):
            continue
        picks.append({**m, "role": "edge"})
    return picks


def _load_agent_prompt(name: str) -> str:
    path = _AGENT_DIR / f"{name}.md"
    if not path.exists():
        return ""
    text = path.read_text()
    # Strip YAML frontmatter between --- markers.
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()


async def _llm_label_and_summary(
    sample_prompts: list[str],
    cluster_meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Return (label_json, summary_json, llm_used)."""
    label: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    try:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return label, summary, False
        from ..models.llm_client import AnthropicClient

        client = AnthropicClient(model="claude-haiku-4-5")
        labeler_prompt = _load_agent_prompt("cluster_labeler")
        summarizer_prompt = _load_agent_prompt("dataset_summarizer")

        prompt_block = "\n".join(f"- {p[:500]}" for p in sample_prompts[:8])

        label_input = (
            f"{labeler_prompt}\n\n## Sample prompts\n{prompt_block}\n\n"
            "Respond with JSON only."
        )
        label_resp = client.generate(label_input, max_tokens=400, temperature=0.1)
        label = _extract_json(label_resp.text)

        summary_input = (
            f"{summarizer_prompt}\n\n"
            f"## Cluster info\n"
            f"- label: {label.get('domain_label') or cluster_meta.get('domain_label', '')}\n"
            f"- description: {label.get('short_description') or cluster_meta.get('short_description', '')}\n"
            f"- trace_count: {cluster_meta.get('trace_count', 0)}\n\n"
            f"## Sample prompts\n{prompt_block}\n\n"
            "Respond with JSON only."
        )
        summary_resp = client.generate(
            summary_input, max_tokens=500, temperature=0.2
        )
        summary = _extract_json(summary_resp.text)
        return label, summary, True
    except Exception as exc:
        logger.warning("LLM label/summary failed, falling back to heuristic: %s", exc)
        return label, summary, False


def _extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    # Strip ```json ... ``` fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else {}
    except (json.JSONDecodeError, TypeError):
        # Best-effort: find first {...} block.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                val = json.loads(match.group(0))
                return val if isinstance(val, dict) else {}
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}
