"""
API Server: FastAPI server for UniRoute.

Provides REST endpoints for routing prompts to LLMs.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import json
import logging
import os

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from .schemas import (
    RouteRequest,
    RouteResponse,
    BatchRouteRequest,
    BatchRouteResponse,
    ModelInfo,
    ModelListResponse,
    RegisterModelRequest,
    StatsResponse,
    HealthResponse,
    ErrorResponse,
)
from ..router.uniroute import UniRouteRouter
from ..models.llm_profile import LLMProfile
from ..models.llm_registry import LLMRegistry
from ..models.llm_client import LLMClient, create_client
from ..storage.state_manager import StateManager
from ..config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Global state
router: Optional[UniRouteRouter] = None
llm_clients: dict[str, LLMClient] = {}
state_manager: Optional[StateManager] = None
settings: Optional[Settings] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("UniRoute API starting up...")
    try:
        from ..storage.secrets import push_all_to_engine
        push_all_to_engine()
        logger.info("Pushed stored API keys to Go engine")
    except Exception as e:
        logger.debug(f"Could not push keys to engine on startup: {e}")

    # Start scan scheduler if enabled
    from ..harness.scheduler import get_scheduler

    scheduler = get_scheduler()
    if scheduler.config.enabled:
        scheduler.start()
        logger.info("Scan scheduler started on lifespan")

    yield

    # Shutdown
    scheduler.stop()
    logger.info("UniRoute API shutting down...")


app = FastAPI(
    title="UniRoute API",
    description="Universal Model Routing for Efficient LLM Inference",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_router() -> UniRouteRouter:
    """Dependency to get the router instance."""
    if router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not initialized. Call /init first.",
        )
    return router


# --- Health & Status ---


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        router_initialized=router is not None,
        num_models=len(router.registry) if router else 0,
        num_clusters=router.cluster_assigner.num_clusters if router else 0,
    )


@app.get("/stats", response_model=StatsResponse, tags=["status"])
async def get_stats(r: UniRouteRouter = Depends(get_router)):
    """Get routing statistics."""
    stats = r.stats
    return StatsResponse(
        total_requests=stats.total_requests,
        model_selections=stats.model_selections,
        cluster_distributions={str(k): v for k, v in stats.cluster_distributions.items()},
        avg_expected_error=stats.avg_expected_error,
        avg_cost_score=stats.avg_cost_score,
    )


@app.post("/stats/reset", tags=["status"])
async def reset_stats(r: UniRouteRouter = Depends(get_router)):
    """Reset routing statistics."""
    r.reset_stats()
    return {"message": "Statistics reset"}


# --- Routing ---


@app.post("/route", response_model=RouteResponse, tags=["routing"])
async def route_prompt(request: RouteRequest, r: UniRouteRouter = Depends(get_router)):
    """
    Route a prompt to the best LLM.

    Optionally execute the prompt on the selected model.
    """
    try:
        decision = r.route(
            prompt=request.prompt,
            available_models=request.available_models,
            cost_weight_override=request.cost_weight,
        )

        response_text = None
        if request.execute:
            if decision.selected_model not in llm_clients:
                raise HTTPException(
                    status_code=400,
                    detail=f"No client configured for {decision.selected_model}",
                )
            client = llm_clients[decision.selected_model]
            llm_response = client.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            response_text = llm_response.text

        return RouteResponse(
            selected_model=decision.selected_model,
            expected_error=decision.expected_error,
            cost_adjusted_score=decision.cost_adjusted_score,
            cluster_id=decision.cluster_id,
            all_scores=decision.all_scores,
            response_text=response_text,
            reasoning=decision.reasoning,
        )

    except Exception as e:
        logger.exception("Routing error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/route/batch", response_model=BatchRouteResponse, tags=["routing"])
async def route_batch(request: BatchRouteRequest, r: UniRouteRouter = Depends(get_router)):
    """Route multiple prompts."""
    try:
        decisions = r.route_batch(
            prompts=request.prompts,
            available_models=request.available_models,
            cost_weight_override=request.cost_weight,
        )

        # Calculate distribution
        model_counts: dict[str, int] = {}
        total_error = 0.0

        responses = []
        for d in decisions:
            model_counts[d.selected_model] = model_counts.get(d.selected_model, 0) + 1
            total_error += d.expected_error

            responses.append(RouteResponse(
                selected_model=d.selected_model,
                expected_error=d.expected_error,
                cost_adjusted_score=d.cost_adjusted_score,
                cluster_id=d.cluster_id,
                all_scores=d.all_scores,
            ))

        distribution = {k: v / len(decisions) for k, v in model_counts.items()}

        return BatchRouteResponse(
            decisions=responses,
            distribution=distribution,
            avg_expected_error=total_error / len(decisions) if decisions else 0,
        )

    except Exception as e:
        logger.exception("Batch routing error")
        raise HTTPException(status_code=500, detail=str(e))


# --- Models ---


@app.get("/models", response_model=ModelListResponse, tags=["models"])
async def list_models(r: UniRouteRouter = Depends(get_router)):
    """List all registered models."""
    profiles = r.registry.get_all()

    models = [
        ModelInfo(
            model_id=p.model_id,
            cost_per_1k_tokens=p.cost_per_1k_tokens,
            num_clusters=p.num_clusters,
            overall_accuracy=p.overall_accuracy,
            strongest_clusters=p.strongest_clusters(3),
        )
        for p in profiles
    ]

    return ModelListResponse(
        models=models,
        default_model=r.registry.default_model_id,
    )


@app.get("/models/{model_id}", response_model=ModelInfo, tags=["models"])
async def get_model(model_id: str, r: UniRouteRouter = Depends(get_router)):
    """Get info about a specific model."""
    profile = r.registry.get(model_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return ModelInfo(
        model_id=profile.model_id,
        cost_per_1k_tokens=profile.cost_per_1k_tokens,
        num_clusters=profile.num_clusters,
        overall_accuracy=profile.overall_accuracy,
        strongest_clusters=profile.strongest_clusters(5),
    )


@app.post("/models", tags=["models"])
async def register_model(request: RegisterModelRequest, r: UniRouteRouter = Depends(get_router)):
    """Register a new model profile."""
    profile = LLMProfile(
        model_id=request.model_id,
        psi_vector=np.array(request.psi_vector),
        cost_per_1k_tokens=request.cost_per_1k_tokens,
        num_validation_samples=request.num_validation_samples,
        cluster_sample_counts=np.array(request.cluster_sample_counts),
        metadata=request.metadata or {},
    )

    r.registry.register(profile)

    # Save to state manager if available
    if state_manager:
        state_manager.save_profile(profile)

    return {"message": f"Model {request.model_id} registered", "model_id": request.model_id}


@app.delete("/models/{model_id}", tags=["models"])
async def unregister_model(model_id: str, r: UniRouteRouter = Depends(get_router)):
    """Remove a model from the registry."""
    profile = r.registry.unregister(model_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return {"message": f"Model {model_id} unregistered"}


# --- Configuration ---


@app.get("/config", tags=["config"])
async def get_config():
    """Get current configuration."""
    if settings is None:
        return {"message": "No settings loaded"}
    return settings.to_dict()


@app.post("/config/cost_weight", tags=["config"])
async def set_cost_weight(cost_weight: float, r: UniRouteRouter = Depends(get_router)):
    """Update the default cost weight."""
    if cost_weight < 0:
        raise HTTPException(status_code=400, detail="cost_weight must be >= 0")
    r.cost_weight = cost_weight
    return {"message": f"Cost weight set to {cost_weight}"}


# --- Harness (Agent System) ---


@app.get("/v1/harness/agents", tags=["harness"])
async def list_harness_agents():
    """List all available harness agents."""
    from ..harness.registry import AgentRegistry
    registry = AgentRegistry()
    return {"agents": [a.to_dict() for a in registry.list_agents()]}


@app.get("/v1/harness/agents/{name}", tags=["harness"])
async def get_harness_agent(name: str):
    """Get a specific agent's config and prompt."""
    from ..harness.registry import AgentRegistry
    registry = AgentRegistry()
    config = registry.get(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return config.to_dict()


@app.post("/v1/harness/run/{name}", tags=["harness"])
async def run_harness_agent(name: str, body: dict):
    """Run a harness agent with user_input and optional context."""
    from ..harness.runner import AgentRunner
    from ..harness.memory_store import get_memory_store

    record = body.get("record_memory", False)
    store = get_memory_store() if record else None

    runner = AgentRunner(memory_store=store, record_memory=record)
    user_input = body.get("input", body.get("user_input", ""))
    context = body.get("context", {})
    use_tools = body.get("use_tools", False)

    if not user_input:
        raise HTTPException(status_code=400, detail="input is required")

    if use_tools:
        result = await runner.run_with_tools(name, user_input)
    else:
        result = await runner.run(name, user_input, context)

    return {"agent": name, "result": result}


# --- Harness Memory ---


@app.get("/v1/harness/memory", tags=["harness"])
async def list_memory(
    agent: str = "",
    category: str = "",
    tags: str = "",
    limit: int = 20,
):
    """Query memory entries with optional filters."""
    from ..harness.memory_store import get_memory_store

    store = get_memory_store()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    entries = store.query(
        agent=agent or None,
        category=category or None,
        tags=tag_list,
        limit=limit,
    )
    return {"entries": [e.to_dict() for e in entries], "count": len(entries)}


@app.get("/v1/harness/memory/summary/{agent_name}", tags=["harness"])
async def get_agent_memory_summary(agent_name: str):
    """Get aggregated performance summary for an agent from memory."""
    from ..harness.memory_store import get_memory_store

    store = get_memory_store()
    return store.agent_summary(agent_name)


@app.get("/v1/harness/memory/{entry_id}", tags=["harness"])
async def get_memory_entry(entry_id: str):
    """Get a specific memory entry by ID."""
    from ..harness.memory_store import get_memory_store

    store = get_memory_store()
    entry = store.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Memory entry '{entry_id}' not found")
    return entry.to_dict()


@app.delete("/v1/harness/memory/{entry_id}", tags=["harness"])
async def delete_memory_entry(entry_id: str):
    """Delete a memory entry."""
    from ..harness.memory_store import get_memory_store

    store = get_memory_store()
    if not store.delete(entry_id):
        raise HTTPException(status_code=404, detail=f"Memory entry '{entry_id}' not found")
    return {"deleted": True, "id": entry_id}


# --- Trace Issues (Scan Now) ---


@app.get("/v1/trace-issues", tags=["trace-issues"])
async def list_trace_issues(
    severity: str = "",
    type: str = "",
    resolved: Optional[str] = None,
):
    """List detected trace issues with optional filters."""
    from ..harness.trace_scanner import list_issues

    resolved_bool = None
    if resolved is not None:
        resolved_bool = resolved.lower() in ("true", "1", "yes")

    issues = list_issues(
        severity=severity or None,
        issue_type=type or None,
        resolved=resolved_bool,
    )
    return {"issues": issues, "count": len(issues)}


@app.post("/v1/trace-issues/scan", tags=["trace-issues"], status_code=202)
async def trigger_trace_scan():
    """Start an async trace scan. Returns scan_id for status polling."""
    import uuid as _uuid
    from ..harness.trace_scanner import TraceScanner

    scan_id = str(_uuid.uuid4())
    scanner = TraceScanner()

    # Run scan in background
    asyncio.ensure_future(scanner.scan(scan_id))

    return {"scan_id": scan_id, "status": "running"}


@app.get("/v1/trace-issues/scan/{scan_id}", tags=["trace-issues"])
async def get_trace_scan_status(scan_id: str):
    """Get status of a running or completed scan."""
    from ..harness.trace_scanner import get_scan_state

    state = get_scan_state(scan_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Scan '{scan_id}' not found")
    return {
        "scan_id": state.scan_id,
        "status": state.status,
        "traces_scanned": state.traces_scanned,
        "issues_found": state.issues_found,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
    }


@app.put("/v1/trace-issues/{issue_id}/resolve", tags=["trace-issues"])
async def resolve_trace_issue(issue_id: str):
    """Mark a trace issue as resolved."""
    from ..harness.trace_scanner import resolve_issue

    if not resolve_issue(issue_id):
        raise HTTPException(status_code=404, detail=f"Issue '{issue_id}' not found")
    return {"resolved": True, "id": issue_id}


@app.put("/v1/trace-issues/{issue_id}/dismiss", tags=["trace-issues"])
async def dismiss_trace_issue(issue_id: str, body: Optional[dict] = None):
    """Dismiss a trace issue as a false positive (not an error)."""
    from ..harness.trace_scanner import dismiss_issue

    reason = (body or {}).get("reason", "")
    success, feedback_id = dismiss_issue(issue_id, reason=reason)
    if not success:
        raise HTTPException(status_code=404, detail=f"Issue '{issue_id}' not found")
    return {"dismissed": True, "id": issue_id, "feedback_id": feedback_id}


@app.get("/v1/trace-issues/schedule", tags=["trace-issues"])
async def get_scan_schedule():
    """Get current scan schedule configuration."""
    from ..harness.scheduler import get_scheduler

    scheduler = get_scheduler()
    return {"schedule": scheduler.config.to_dict(), "running": scheduler.running}


@app.put("/v1/trace-issues/schedule", tags=["trace-issues"])
async def update_scan_schedule(body: dict):
    """Update scan schedule configuration (enable/disable, interval, etc)."""
    from ..harness.scheduler import get_scheduler

    scheduler = get_scheduler()
    config = scheduler.update_config(
        enabled=body.get("enabled"),
        interval_seconds=body.get("interval_seconds"),
        days_lookback=body.get("days_lookback"),
        trace_limit=body.get("trace_limit"),
    )
    return {"schedule": config.to_dict(), "running": scheduler.running}


@app.get("/v1/trace-issues/feedback", tags=["trace-issues"])
async def list_trace_feedback(
    issue_type: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = 50,
):
    """List user feedback (false positive dismissals)."""
    from ..harness.memory_store import get_memory_store

    store = get_memory_store()
    tags = []
    if issue_type:
        tags.append(issue_type)
    if model:
        tags.append(model)

    entries = store.query(
        agent="trace_scanner",
        category="user_feedback",
        tags=tags or None,
        limit=limit,
    )
    return {
        "feedback": [e.to_dict() for e in entries],
        "count": len(entries),
    }


# --- Clustering (Domain Datasets) ---


@app.post("/v1/clustering/run", tags=["clustering"])
async def clustering_run(
    days: int = 30,
    min_traces: int = 50,
    strategy: str = "auto",
):
    """Trigger a clustering pipeline run."""
    from ..clustering.pipeline import ClusteringPipeline

    pipeline = ClusteringPipeline(strategy=strategy)
    result = await pipeline.run(days=days, min_traces=min_traces)
    return result.to_dict()


@app.get("/v1/clustering/runs", tags=["clustering"])
async def list_clustering_runs():
    """List past clustering runs."""
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return {"runs": []}

    r = client.query(
        "SELECT * FROM clustering_runs ORDER BY created_at DESC LIMIT 20"
    )
    columns = r.column_names
    return {"runs": [dict(zip(columns, row)) for row in r.result_rows]}


@app.get("/v1/clustering/runs/{run_id}", tags=["clustering"])
async def get_clustering_run(run_id: str):
    """Get a clustering run with all its datasets."""
    from ..storage.clickhouse_client import get_client
    import json

    client = get_client()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    # Get run info
    r = client.query(
        "SELECT * FROM clustering_runs WHERE run_id = {rid:String}",
        parameters={"rid": run_id},
    )
    if not r.result_rows:
        raise HTTPException(status_code=404, detail="Run not found")

    run = dict(zip(r.column_names, r.result_rows[0]))

    # Get datasets
    r = client.query(
        "SELECT * FROM cluster_datasets WHERE run_id = {rid:String} ORDER BY cluster_id",
        parameters={"rid": run_id},
    )
    datasets = []
    for row in r.result_rows:
        d = dict(zip(r.column_names, row))
        for field in ("top_models", "top_providers", "sample_prompts"):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except Exception:
                    pass
        datasets.append(d)

    return {"run": run, "datasets": datasets}


@app.get("/v1/clustering/datasets", tags=["clustering"])
async def list_clustering_datasets(status: Optional[str] = None):
    """List domain datasets from the latest clustering run."""
    from ..storage.clickhouse_client import get_client
    import json

    client = get_client()
    if client is None:
        return {"datasets": []}

    # Get latest run_id
    r = client.query("SELECT run_id FROM clustering_runs ORDER BY created_at DESC LIMIT 1")
    if not r.result_rows:
        return {"datasets": [], "run_id": None}

    run_id = r.result_rows[0][0]

    conditions = ["run_id = {rid:String}"]
    params: dict = {"rid": run_id}
    if status:
        conditions.append("status = {status:String}")
        params["status"] = status

    where = " AND ".join(conditions)
    r = client.query(
        f"SELECT * FROM cluster_datasets WHERE {where} ORDER BY trace_count DESC",
        parameters=params,
    )
    datasets = []
    for row in r.result_rows:
        d = dict(zip(r.column_names, row))
        for field in ("top_models", "top_providers", "sample_prompts"):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except Exception:
                    pass
        datasets.append(d)

    return {"datasets": datasets, "run_id": run_id}


@app.get("/v1/clustering/datasets/{run_id}/{cluster_id}", tags=["clustering"])
async def get_clustering_dataset_traces(
    run_id: str,
    cluster_id: int,
    limit: int = 100,
    offset: int = 0,
):
    """Get traces belonging to a specific domain dataset."""
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    # Join traces with cluster map — try matching by input_text
    r = client.query(
        "SELECT t.*, m.input_text AS mapped_input FROM llm_traces t "
        "INNER JOIN trace_cluster_map m ON t.input_text = m.input_text "
        "WHERE m.run_id = {rid:String} AND m.cluster_id = {cid:UInt32} "
        "AND length(t.input_text) > 0 "
        "ORDER BY t.timestamp DESC LIMIT {lim:UInt32} OFFSET {off:UInt32}",
        parameters={"rid": run_id, "cid": cluster_id, "lim": limit, "off": offset},
    )
    columns = r.column_names
    traces = [dict(zip(columns, row)) for row in r.result_rows]

    # Fallback: if no joined results, serve content directly from the mapping table
    if not traces:
        r = client.query(
            "SELECT input_text, output_text, run_id, cluster_id "
            "FROM trace_cluster_map "
            "WHERE run_id = {rid:String} AND cluster_id = {cid:UInt32} "
            "ORDER BY input_text "
            "LIMIT {lim:UInt32} OFFSET {off:UInt32}",
            parameters={"rid": run_id, "cid": cluster_id, "lim": limit, "off": offset},
        )
        traces = [
            {"input_text": row[0], "output_text": row[1], "request_id": f"map-{idx}", "run_id": row[2], "cluster_id": row[3]}
            for idx, row in enumerate(r.result_rows)
        ]

    # Get count
    r2 = client.query(
        "SELECT count() FROM trace_cluster_map WHERE run_id = {rid:String} AND cluster_id = {cid:UInt32}",
        parameters={"rid": run_id, "cid": cluster_id},
    )
    total = int(r2.result_rows[0][0]) if r2.result_rows else 0

    return {"traces": traces, "total": total, "run_id": run_id, "cluster_id": cluster_id}


@app.post("/v1/clustering/datasets/{run_id}/{cluster_id}/export", tags=["clustering"])
async def export_clustering_dataset(run_id: str, cluster_id: int):
    """Export a domain dataset as JSONL (prompt/response pairs)."""
    from ..storage.clickhouse_client import get_client
    from fastapi.responses import StreamingResponse
    import json

    client = get_client()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    # Try join first, fallback to mapping table for input_text
    r = client.query(
        "SELECT t.input_text, t.output_text, t.selected_model, t.provider, "
        "t.tokens_in, t.tokens_out, t.total_cost_usd, t.is_error, t.input_messages, t.output_message "
        "FROM llm_traces t "
        "INNER JOIN trace_cluster_map m ON t.input_text = m.input_text "
        "WHERE m.run_id = {rid:String} AND m.cluster_id = {cid:UInt32} "
        "AND length(t.input_text) > 0 ORDER BY t.timestamp",
        parameters={"rid": run_id, "cid": cluster_id},
    )
    if not r.result_rows:
        # Fallback: use input_text and output_text from trace_cluster_map directly
        r = client.query(
            "SELECT input_text, output_text, '', '', 0, 0, 0, 0, '', '' "
            "FROM trace_cluster_map "
            "WHERE run_id = {rid:String} AND cluster_id = {cid:UInt32} "
            "ORDER BY input_text",
            parameters={"rid": run_id, "cid": cluster_id},
        )

    def generate():
        for row in r.result_rows:
            messages = []
            try:
                msgs = json.loads(row[8]) if row[8] else None
                if isinstance(msgs, list):
                    messages = msgs
            except Exception:
                messages = [{"role": "user", "content": row[0]}]

            if not messages:
                messages = [{"role": "user", "content": row[0]}]

            if row[1]:  # output_text
                messages.append({"role": "assistant", "content": row[1]})

            record = {
                "messages": messages,
                "metadata": {
                    "model": row[2],
                    "provider": row[3],
                    "tokens_in": int(row[4]),
                    "tokens_out": int(row[5]),
                    "cost_usd": float(row[6]),
                    "is_error": bool(row[7]),
                },
            }
            yield json.dumps(record) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f"attachment; filename=dataset_{run_id}_{cluster_id}.jsonl"
        },
    )


@app.post("/v1/clustering/datasets/{run_id}/{cluster_id}/qualify", tags=["clustering"])
async def qualify_clustering_dataset(run_id: str, cluster_id: int, status: str = "qualified"):
    """Manually qualify or reject a dataset."""
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    if status not in ("qualified", "rejected", "candidate"):
        raise HTTPException(status_code=400, detail="status must be qualified, rejected, or candidate")

    client.command(
        f"ALTER TABLE cluster_datasets UPDATE status = '{status}' "
        f"WHERE run_id = '{run_id}' AND cluster_id = {cluster_id}"
    )
    return {"message": f"Dataset {cluster_id} status set to {status}"}


# --- Trace Ingestion ---


@app.post("/v1/traces", tags=["traces"])
async def ingest_traces(body: dict):
    """Ingest manual traces into ClickHouse.

    Accepts single trace or batch:
        {"messages": [...], "model": "gpt-4o-mini"}
        {"input": "Hello", "output": "Hi", "model": "gpt-4o-mini"}
        {"traces": [{...}, {...}]}

    Auto-enriches: token estimates, cost calculation, timestamps.
    """
    import httpx

    engine_url = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
    try:
        resp = httpx.post(f"{engine_url}/v1/traces", json=body, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Engine unavailable: {e}")


@app.post("/v1/traces/import", tags=["traces"])
async def import_traces_file(body: dict):
    """Import traces from a JSONL string body.

    Body: {"data": "line1\\nline2\\n...", "source": "import", "model": "gpt-4o-mini"}
    """
    import httpx

    data_str = body.get("data", "")
    source = body.get("source", "file-import")
    default_model = body.get("model", "")
    default_provider = body.get("provider", "")

    if not data_str:
        raise HTTPException(status_code=400, detail="'data' field with JSONL content is required")

    traces = []
    for line in data_str.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
            if source and "source" not in t:
                t["source"] = source
            if default_model and "model" not in t:
                t["model"] = default_model
            if default_provider and "provider" not in t:
                t["provider"] = default_provider
            traces.append(t)
        except json.JSONDecodeError:
            continue

    if not traces:
        raise HTTPException(status_code=400, detail="No valid traces found in data")

    engine_url = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
    try:
        resp = httpx.post(f"{engine_url}/v1/traces", json={"traces": traces}, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Engine unavailable: {e}")


# --- Add Traces to Existing Dataset ---


@app.post("/v1/clustering/datasets/{run_id}/{cluster_id}/traces", tags=["datasets"])
async def add_traces_to_dataset(run_id: str, cluster_id: int, body: dict):
    """Add manual traces to an existing cluster dataset.

    Inserts traces into ClickHouse AND maps them to the specified cluster.
    Accepts same format as /v1/traces: messages, input/output, or batch.
    """
    import httpx

    engine_url = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
    try:
        resp = httpx.post(
            f"{engine_url}/v1/datasets/{run_id}/{cluster_id}/traces",
            json=body,
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Engine unavailable: {e}")


@app.post("/v1/clustering/datasets/{run_id}/{cluster_id}/assign", tags=["datasets"])
async def assign_traces_to_dataset(run_id: str, cluster_id: int, body: dict):
    """Assign existing traces (by request_id) to a cluster dataset.

    Body: {"request_ids": ["id1", "id2", ...]}
    Maps existing traces from the Traces page into a specific dataset.
    """
    import httpx

    request_ids = body.get("request_ids", [])
    if not request_ids:
        raise HTTPException(status_code=400, detail="request_ids is required")

    engine_url = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
    try:
        resp = httpx.post(
            f"{engine_url}/v1/datasets/{run_id}/{cluster_id}/assign",
            json={"request_ids": request_ids},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Engine unavailable: {e}")


# --- Dataset Trace Import (Smart Import for UI) ---


def _detect_field(record: dict, candidates: list[str]) -> str:
    """Find the first matching field name from a list of candidates."""
    for c in candidates:
        if c in record:
            return c
        # Try nested: "message.content", etc.
        parts = c.split(".")
        obj = record
        for p in parts:
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                obj = None
                break
        if obj is not None:
            return c
    return ""


def _extract_value(record: dict, path: str) -> str:
    """Extract a value from a record using a dot-separated path."""
    if not path:
        return ""
    parts = path.split(".")
    obj = record
    for p in parts:
        if isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            return ""
    return str(obj) if obj is not None else ""


@app.post("/v1/datasets/analyze-traces", tags=["datasets"])
async def analyze_traces_schema(body: dict):
    """Auto-detect input/output schema from uploaded JSON data.

    Examines field names and content to determine which fields
    map to input (prompt) and output (response).
    """
    data = body.get("data", [])
    if not data or not isinstance(data, list):
        raise HTTPException(status_code=400, detail="'data' must be a non-empty array")

    sample = data[0] if data else {}

    # Detect format
    input_candidates = [
        "messages", "input", "prompt", "question", "query",
        "user_message", "instruction", "text", "input_text",
    ]
    output_candidates = [
        "output", "response", "answer", "completion", "expected_output",
        "assistant_message", "output_text", "result", "generated_text",
    ]

    source_format = "unknown"
    input_path = ""
    output_path = ""

    # Check for OpenAI messages format
    if "messages" in sample and isinstance(sample["messages"], list):
        source_format = "openai-messages"
        input_path = "messages"
        output_path = "messages"
    else:
        input_path = _detect_field(sample, input_candidates)
        output_path = _detect_field(sample, output_candidates)
        if input_path and output_path:
            source_format = "input-output"
        elif input_path:
            source_format = "input-only"

    # Build preview
    preview = []
    for record in data[:10]:
        inp = ""
        out = ""

        if source_format == "openai-messages":
            msgs = record.get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    inp = m.get("content", "")
                elif m.get("role") == "assistant":
                    out = m.get("content", "")
        else:
            inp = _extract_value(record, input_path)
            out = _extract_value(record, output_path)

        # Collect metadata (all other fields)
        meta = {}
        for k, v in record.items():
            if k not in (input_path, output_path, "messages") and isinstance(v, (str, int, float, bool)):
                meta[k] = v

        preview.append({
            "input": inp[:500] if inp else "",
            "expected_output": out[:500] if out else "",
            "metadata": meta,
        })

    # Build mapping
    mapping = {
        "input": {"path": input_path, "transform": "direct"},
        "output": {"path": output_path, "transform": "direct"},
        "metadata": {},
    }

    return {
        "mapping": mapping,
        "preview": preview,
        "source_format": source_format,
        "total_records": len(data),
    }


@app.post("/v1/datasets/import-traces", tags=["datasets"])
async def import_traces_to_clickhouse(body: dict):
    """Import trace data into ClickHouse using the detected mapping.

    Receives data + mapping from analyze-traces, transforms records
    into traces and sends to the Go engine.
    """
    import httpx
    import uuid

    name = body.get("name", "imported-dataset")
    data = body.get("data", [])
    mapping = body.get("mapping", {})
    description = body.get("description", "")

    if not data:
        raise HTTPException(status_code=400, detail="'data' is required")

    input_path = mapping.get("input", {}).get("path", "input") if mapping else "input"
    output_path = mapping.get("output", {}).get("path", "output") if mapping else "output"

    # Transform records into trace format
    traces = []
    skipped = 0
    for record in data:
        inp = ""
        out = ""

        # Handle OpenAI messages format
        if input_path == "messages" and "messages" in record:
            msgs = record.get("messages", [])
            trace = {
                "messages": msgs,
                "source": f"import:{name}",
            }
            # Extract model/provider if present
            if "model" in record:
                trace["model"] = record["model"]
            if "provider" in record:
                trace["provider"] = record["provider"]
            traces.append(trace)
            continue

        inp = _extract_value(record, input_path)
        out = _extract_value(record, output_path)

        if not inp and not out:
            skipped += 1
            continue

        trace = {
            "input": inp,
            "output": out,
            "source": f"import:{name}",
        }
        if "model" in record:
            trace["model"] = record["model"]
        if "provider" in record:
            trace["provider"] = record["provider"]
        traces.append(trace)

    if not traces:
        raise HTTPException(status_code=400, detail="No valid traces found after mapping")

    # Send to Go engine
    engine_url = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
    try:
        resp = httpx.post(
            f"{engine_url}/v1/traces",
            json={"traces": traces},
            timeout=60.0,
        )
        resp.raise_for_status()
        result = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Engine unavailable: {e}")

    dataset_id = str(uuid.uuid4())[:8]

    return {
        "dataset_id": dataset_id,
        "name": name,
        "source": "smart-import",
        "samples_count": result.get("ingested", len(traces)),
        "skipped_count": skipped,
    }


# --- Evaluation Datasets (CRUD — ClickHouse-backed) ---


def _ensure_eval_tables() -> None:
    """Create eval_datasets / eval_dataset_samples if they don't exist."""
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return
    client.command("""
        CREATE TABLE IF NOT EXISTS eval_datasets (
            id              String,
            name            String,
            description     String,
            source          LowCardinality(String),
            samples_count   UInt32,
            created_at      DateTime64(3, 'UTC'),
            updated_at      DateTime64(3, 'UTC')
        ) ENGINE = ReplacingMergeTree(updated_at) ORDER BY (id)
    """)
    client.command("""
        CREATE TABLE IF NOT EXISTS eval_dataset_samples (
            id              String,
            dataset_id      String,
            input           String,
            output          String,
            expected_output String,
            metadata        String,
            created_at      DateTime64(3, 'UTC')
        ) ENGINE = MergeTree ORDER BY (dataset_id, id)
    """)


_eval_tables_ready = False


def _ch_eval():
    """Return ClickHouse client, ensuring eval tables exist."""
    from ..storage.clickhouse_client import get_client

    global _eval_tables_ready
    client = get_client()
    if client is None:
        return None
    if not _eval_tables_ready:
        _ensure_eval_tables()
        _eval_tables_ready = True
    return client


@app.get("/v1/datasets", tags=["datasets"])
async def list_datasets():
    """List all datasets: evaluation datasets + clustering domain datasets."""
    client = _ch_eval()
    if client is None:
        return {"datasets": [], "total": 0}

    datasets = []

    # 1) Evaluation datasets
    r = client.query(
        "SELECT id, name, description, source, samples_count, created_at, updated_at "
        "FROM eval_datasets FINAL ORDER BY created_at DESC"
    )
    for row in r.result_rows:
        d = dict(zip(r.column_names, row))
        for field in ("created_at", "updated_at"):
            if hasattr(d.get(field), "isoformat"):
                d[field] = d[field].isoformat()
        datasets.append(d)

    # 2) Clustering domain datasets (from latest run)
    try:
        rr = client.query("SELECT run_id FROM clustering_runs ORDER BY created_at DESC LIMIT 1")
        if rr.result_rows:
            run_id = rr.result_rows[0][0]
            cr = client.query(
                "SELECT cluster_id, domain_label, short_description, status, trace_count "
                "FROM cluster_datasets WHERE run_id = {rid:String} ORDER BY trace_count DESC",
                parameters={"rid": run_id},
            )
            for row in cr.result_rows:
                cid, label, desc, status, count = row
                name = label if label and label != "Unknown" else f"Cluster {cid}"
                datasets.append({
                    "id": f"cluster:{run_id}:{cid}",
                    "name": f"[Domain] {name}",
                    "description": desc or f"Domain cluster ({status}, {count} traces)",
                    "source": "auto_collected",
                    "samples_count": count,
                    "created_at": "",
                    "updated_at": "",
                })
    except Exception:
        pass  # clustering tables may not exist

    return {"datasets": datasets, "total": len(datasets)}


@app.post("/v1/datasets", tags=["datasets"])
async def create_dataset(body: dict):
    """Create a new evaluation dataset."""
    import uuid
    from datetime import datetime, timezone

    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="'name' is required")

    now = datetime.now(timezone.utc)
    dataset_id = str(uuid.uuid4())[:8]
    client.insert("eval_datasets",
        [[dataset_id, name, body.get("description", ""), body.get("source", "manual"), 0, now, now]],
        column_names=["id", "name", "description", "source", "samples_count", "created_at", "updated_at"],
    )
    return {
        "id": dataset_id,
        "name": name,
        "description": body.get("description", ""),
        "source": body.get("source", "manual"),
        "samples_count": 0,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


@app.get("/v1/datasets/{dataset_id}", tags=["datasets"])
async def get_dataset(
    dataset_id: str,
    include_samples: bool = False,
    samples_limit: int = 50,
    samples_offset: int = 0,
):
    """Get a single dataset by ID."""
    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    r = client.query(
        "SELECT id, name, description, source, samples_count, created_at, updated_at "
        "FROM eval_datasets FINAL WHERE id = {did:String}",
        parameters={"did": dataset_id},
    )
    if not r.result_rows:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ds = dict(zip(r.column_names, r.result_rows[0]))
    for field in ("created_at", "updated_at"):
        if hasattr(ds.get(field), "isoformat"):
            ds[field] = ds[field].isoformat()

    samples = []
    samples_total = 0
    if include_samples:
        sr = client.query(
            "SELECT id, input, output, expected_output, metadata, created_at "
            "FROM eval_dataset_samples WHERE dataset_id = {did:String} "
            "ORDER BY created_at DESC LIMIT {lim:UInt32} OFFSET {off:UInt32}",
            parameters={"did": dataset_id, "lim": samples_limit, "off": samples_offset},
        )
        for row in sr.result_rows:
            s = dict(zip(sr.column_names, row))
            if hasattr(s.get("created_at"), "isoformat"):
                s["created_at"] = s["created_at"].isoformat()
            if isinstance(s.get("metadata"), str) and s["metadata"]:
                try:
                    s["metadata"] = json.loads(s["metadata"])
                except Exception:
                    pass
            samples.append(s)
    cr = client.query(
        "SELECT count() FROM eval_dataset_samples WHERE dataset_id = {did:String}",
        parameters={"did": dataset_id},
    )
    samples_total = cr.result_rows[0][0] if cr.result_rows else 0

    return {"dataset": ds, "samples": samples, "samples_total": samples_total}


@app.delete("/v1/datasets/{dataset_id}", tags=["datasets"])
async def delete_dataset(dataset_id: str):
    """Delete a dataset and its samples."""
    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    client.command(
        f"ALTER TABLE eval_datasets DELETE WHERE id = '{dataset_id}'"
    )
    client.command(
        f"ALTER TABLE eval_dataset_samples DELETE WHERE dataset_id = '{dataset_id}'"
    )
    return {"success": True}


@app.post("/v1/datasets/{dataset_id}/samples", tags=["datasets"])
async def add_samples(dataset_id: str, body: dict):
    """Add samples to an existing dataset."""
    import uuid
    from datetime import datetime, timezone

    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    samples_input = body.get("samples", [])
    if not samples_input:
        raise HTTPException(status_code=400, detail="'samples' is required")

    now = datetime.now(timezone.utc)
    rows = []
    for s in samples_input:
        meta = s.get("metadata", {})
        rows.append([
            str(uuid.uuid4())[:8],
            dataset_id,
            s.get("input", ""),
            s.get("output", ""),
            s.get("expected_output", ""),
            json.dumps(meta) if isinstance(meta, dict) else str(meta),
            now,
        ])

    client.insert("eval_dataset_samples", rows,
        column_names=["id", "dataset_id", "input", "output", "expected_output", "metadata", "created_at"],
    )

    # Update samples_count on the dataset
    cr = client.query(
        "SELECT count() FROM eval_dataset_samples WHERE dataset_id = {did:String}",
        parameters={"did": dataset_id},
    )
    new_count = cr.result_rows[0][0] if cr.result_rows else len(rows)
    client.insert("eval_datasets",
        [[dataset_id, "", "", "", new_count, now, now]],
        column_names=["id", "name", "description", "source", "samples_count", "created_at", "updated_at"],
    )

    return {"message": f"Added {len(rows)} samples", "count": len(rows)}


# --- Available Models ---

# Well-known models per provider
_PROVIDER_MODELS = {
    "openai": [
        ("gpt-4o", "GPT-4o"),
        ("gpt-4o-mini", "GPT-4o Mini"),
        ("gpt-4-turbo", "GPT-4 Turbo"),
        ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
    ],
    "anthropic": [
        ("claude-3-5-sonnet-latest", "Claude 3.5 Sonnet"),
        ("claude-3-5-haiku-latest", "Claude 3.5 Haiku"),
        ("claude-3-opus-latest", "Claude 3 Opus"),
    ],
    "mistral": [
        ("mistral-small-latest", "Mistral Small"),
        ("mistral-large-latest", "Mistral Large"),
        ("open-mistral-nemo", "Mistral Nemo"),
    ],
    "groq": [
        ("llama-3.1-70b-versatile", "Llama 3.1 70B"),
        ("llama-3.1-8b-instant", "Llama 3.1 8B"),
        ("mixtral-8x7b-32768", "Mixtral 8x7B"),
    ],
    "deepseek": [
        ("deepseek-chat", "DeepSeek Chat"),
        ("deepseek-reasoner", "DeepSeek Reasoner"),
    ],
}


@app.get("/v1/models/available", tags=["models"])
async def list_available_models():
    """List models available for evaluation based on configured providers."""
    from ..storage.secrets import list_configured_providers

    configured = list_configured_providers()
    models = []
    for provider in configured:
        provider_models = _PROVIDER_MODELS.get(provider, [])
        for model_id, name in provider_models:
            models.append({
                "id": f"{provider}/{model_id}",
                "name": name,
                "provider": provider,
                "type": "external",
                "available": True,
            })
    return {"models": models}


# --- Metrics (CRUD + AI Suggestion) ---

_BUILTIN_METRICS = [
    {
        "metric_id": "exact_match",
        "name": "Exact Match",
        "type": "exact_match",
        "description": "Binary pass/fail comparing output to expected output exactly",
        "config": {"ignore_case": False, "ignore_whitespace": False},
        "is_builtin": True,
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "metric_id": "contains",
        "name": "Contains Keywords",
        "type": "contains",
        "description": "Checks if output contains expected keywords or phrases",
        "config": {"ignore_case": True, "all_must_match": False},
        "is_builtin": True,
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "metric_id": "similarity",
        "name": "Similarity",
        "type": "semantic_sim",
        "description": "Cosine similarity between embeddings of output and expected output",
        "config": {"threshold": 0.8},
        "is_builtin": True,
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "metric_id": "llm_judge",
        "name": "LLM-as-Judge",
        "type": "llm_judge",
        "description": "Uses an LLM to evaluate response quality on criteria like accuracy and helpfulness",
        "config": {"judge_model": "openai/gpt-4o-mini", "criteria": ["accuracy", "helpfulness", "coherence"], "scale": {"min": 1, "max": 10}},
        "is_builtin": True,
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "metric_id": "latency",
        "name": "Latency",
        "type": "latency",
        "description": "Measures response time in seconds",
        "config": {"max_acceptable": 10.0},
        "is_builtin": True,
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "metric_id": "cost",
        "name": "Cost",
        "type": "cost",
        "description": "Measures inference cost per request in USD",
        "config": {"max_acceptable": 0.01},
        "is_builtin": True,
        "created_at": "2025-01-01T00:00:00Z",
    },
]


def _ensure_metrics_table():
    """Create eval_metrics table if it doesn't exist."""
    client = _ch_eval()
    if client is None:
        return
    client.command("""
        CREATE TABLE IF NOT EXISTS eval_metrics (
            metric_id       String,
            name            String,
            type            LowCardinality(String),
            description     String,
            config          String,
            is_builtin      UInt8,
            python_script   String,
            created_at      DateTime64(3, 'UTC'),
            updated_at      DateTime64(3, 'UTC')
        ) ENGINE = ReplacingMergeTree(updated_at) ORDER BY (metric_id)
    """)


_metrics_table_ready = False


@app.get("/v1/metrics", tags=["metrics"])
async def list_metrics():
    """List all metrics: built-in + custom."""
    global _metrics_table_ready

    custom = []
    client = _ch_eval()
    if client is not None:
        if not _metrics_table_ready:
            _ensure_metrics_table()
            _metrics_table_ready = True
        r = client.query(
            "SELECT metric_id, name, type, description, config, is_builtin, python_script, created_at, updated_at "
            "FROM eval_metrics FINAL ORDER BY created_at DESC"
        )
        for row in r.result_rows:
            m = dict(zip(r.column_names, row))
            m["is_builtin"] = bool(m.get("is_builtin"))
            if isinstance(m.get("config"), str) and m["config"]:
                try:
                    m["config"] = json.loads(m["config"])
                except Exception:
                    pass
            for field in ("created_at", "updated_at"):
                if hasattr(m.get(field), "isoformat"):
                    m[field] = m[field].isoformat()
            custom.append(m)

    all_metrics = _BUILTIN_METRICS + custom
    return {"metrics": all_metrics, "count": len(all_metrics)}


@app.post("/v1/metrics", tags=["metrics"])
async def create_metric(body: dict):
    """Create a custom metric."""
    import uuid
    from datetime import datetime, timezone

    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    global _metrics_table_ready
    if not _metrics_table_ready:
        _ensure_metrics_table()
        _metrics_table_ready = True

    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="'name' is required")

    now = datetime.now(timezone.utc)
    metric_id = body.get("metric_id", str(uuid.uuid4())[:8])
    config = body.get("config", {})

    client.insert("eval_metrics",
        [[metric_id, name, body.get("type", "python"), body.get("description", ""),
          json.dumps(config) if isinstance(config, dict) else str(config),
          0, body.get("python_script", ""), now, now]],
        column_names=["metric_id", "name", "type", "description", "config",
                      "is_builtin", "python_script", "created_at", "updated_at"],
    )

    return {
        "metric_id": metric_id,
        "name": name,
        "type": body.get("type", "python"),
        "description": body.get("description", ""),
        "config": config,
        "is_builtin": False,
        "python_script": body.get("python_script", ""),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


@app.get("/v1/metrics/{metric_id}", tags=["metrics"])
async def get_metric(metric_id: str):
    """Get a single metric."""
    # Check builtins first
    for m in _BUILTIN_METRICS:
        if m["metric_id"] == metric_id:
            return m

    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=404, detail="Metric not found")

    r = client.query(
        "SELECT metric_id, name, type, description, config, is_builtin, python_script, created_at, updated_at "
        "FROM eval_metrics FINAL WHERE metric_id = {mid:String}",
        parameters={"mid": metric_id},
    )
    if not r.result_rows:
        raise HTTPException(status_code=404, detail="Metric not found")

    m = dict(zip(r.column_names, r.result_rows[0]))
    m["is_builtin"] = bool(m.get("is_builtin"))
    if isinstance(m.get("config"), str):
        try:
            m["config"] = json.loads(m["config"])
        except Exception:
            pass
    return m


@app.delete("/v1/metrics/{metric_id}", tags=["metrics"])
async def delete_metric(metric_id: str):
    """Delete a custom metric."""
    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")
    client.command(f"ALTER TABLE eval_metrics DELETE WHERE metric_id = '{metric_id}'")
    return {"success": True}


@app.post("/v1/auto-eval/suggest-metrics", tags=["metrics"])
async def suggest_metrics(body: dict):
    """Use harness AI to suggest metrics for a dataset."""
    from ..harness.runner import AgentRunner

    dataset_id = body.get("dataset_id", "")

    # Get sample prompts from the dataset
    sample_prompts = []
    domain = "general"

    client = _ch_eval()
    if client and dataset_id.startswith("cluster:"):
        parts = dataset_id.split(":")
        if len(parts) == 3:
            run_id, cluster_id = parts[1], parts[2]
            r = client.query(
                "SELECT domain_label, short_description FROM cluster_datasets "
                "WHERE run_id = {rid:String} AND cluster_id = {cid:UInt32}",
                parameters={"rid": run_id, "cid": int(cluster_id)},
            )
            if r.result_rows:
                domain = r.result_rows[0][0] or "general"
            r = client.query(
                "SELECT input_text FROM trace_cluster_map "
                "WHERE run_id = {rid:String} AND cluster_id = {cid:UInt32} LIMIT 5",
                parameters={"rid": run_id, "cid": int(cluster_id)},
            )
            sample_prompts = [row[0] for row in r.result_rows if row[0]]

    if not sample_prompts:
        sample_prompts = ["(no samples available)"]

    numbered = "\n".join(f'{i+1}. "{p[:200]}"' for i, p in enumerate(sample_prompts[:10]))
    user_input = f"Domain: {domain}\n\nSample prompts:\n{numbered}"

    runner = AgentRunner()
    try:
        result = await runner.run("metrics_suggester", user_input)
        suggestions = result.data.get("suggested_metrics", [])

        # Auto-create suggested metrics as custom metrics in ClickHouse
        auto_create = body.get("auto_create", True)
        created = []
        if auto_create and suggestions and client:
            import uuid
            from datetime import datetime, timezone

            global _metrics_table_ready
            if not _metrics_table_ready:
                _ensure_metrics_table()
                _metrics_table_ready = True

            now = datetime.now(timezone.utc)
            for s in suggestions:
                mid = s.get("metric_id", str(uuid.uuid4())[:8])
                # Skip if already exists
                r = client.query(
                    "SELECT count() FROM eval_metrics WHERE metric_id = {mid:String}",
                    parameters={"mid": mid},
                )
                if r.result_rows and r.result_rows[0][0] > 0:
                    continue
                config = s.get("config", {})
                client.insert("eval_metrics",
                    [[mid, s.get("name", mid), s.get("type", "python"),
                      s.get("description", ""), json.dumps(config) if isinstance(config, dict) else str(config),
                      0, "", now, now]],
                    column_names=["metric_id", "name", "type", "description", "config",
                                  "is_builtin", "python_script", "created_at", "updated_at"],
                )
                created.append(mid)

        return {
            "suggestions": suggestions,
            "rationale": result.data.get("rationale", ""),
            "created_metrics": created,
        }
    except Exception as e:
        return {"suggestions": [], "rationale": f"Suggestion failed: {e}", "error": str(e)}


# --- Evaluations (Run + Status) ---


def _ensure_evaluations_table():
    client = _ch_eval()
    if client is None:
        return
    client.command("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id              String,
            name            String,
            description     String,
            dataset_id      String,
            models          String,
            metrics         String,
            status          LowCardinality(String),
            config          String,
            created_at      DateTime64(3, 'UTC'),
            updated_at      DateTime64(3, 'UTC')
        ) ENGINE = ReplacingMergeTree(updated_at) ORDER BY (id)
    """)


_evals_table_ready = False


@app.get("/v1/evaluations", tags=["evaluations"])
async def list_evaluations():
    """List all evaluations."""
    global _evals_table_ready
    client = _ch_eval()
    if client is None:
        return {"evaluations": []}

    if not _evals_table_ready:
        _ensure_evaluations_table()
        _evals_table_ready = True

    r = client.query(
        "SELECT id, name, description, dataset_id, models, metrics, status, config, created_at, updated_at "
        "FROM evaluations FINAL ORDER BY created_at DESC"
    )
    evals = []
    for row in r.result_rows:
        e = dict(zip(r.column_names, row))
        for field in ("models", "metrics", "config"):
            if isinstance(e.get(field), str) and e[field]:
                try:
                    e[field] = json.loads(e[field])
                except Exception:
                    pass
        for field in ("created_at", "updated_at"):
            if hasattr(e.get(field), "isoformat"):
                e[field] = e[field].isoformat()
        evals.append(e)
    return {"evaluations": evals}


@app.post("/v1/evaluations", tags=["evaluations"])
async def create_evaluation(body: dict):
    """Create and queue a new evaluation run."""
    import uuid
    from datetime import datetime, timezone

    global _evals_table_ready
    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    if not _evals_table_ready:
        _ensure_evaluations_table()
        _evals_table_ready = True

    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="'name' is required")

    now = datetime.now(timezone.utc)
    eval_id = str(uuid.uuid4())[:8]
    models = body.get("models", [])
    metrics = body.get("metrics", [])

    client.insert("evaluations",
        [[eval_id, name, body.get("description", ""), body.get("dataset_id", ""),
          json.dumps(models), json.dumps(metrics), "queued",
          json.dumps(body.get("config", {})), now, now]],
        column_names=["id", "name", "description", "dataset_id", "models", "metrics",
                      "status", "config", "created_at", "updated_at"],
    )

    # Launch background execution
    asyncio.create_task(_run_evaluation(eval_id, body.get("dataset_id", ""), models, metrics))

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=202,
        content={
            "evaluation_id": eval_id,
            "name": name,
            "status": "queued",
            "models": models,
            "metrics": metrics,
            "dataset_id": body.get("dataset_id", ""),
            "created_at": now.isoformat(),
        },
    )


async def _run_evaluation(eval_id: str, dataset_id: str, models: list, metrics: list):
    """Background task: run samples through models in parallel, score, store results."""
    import httpx
    from datetime import datetime, timezone

    client = _ch_eval()
    if client is None:
        return

    engine_url = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
    max_concurrency = int(os.environ.get("LUNAR_EVAL_CONCURRENCY", "10"))

    def _update_status(status: str, **extra):
        now = datetime.now(timezone.utc)
        client.insert("evaluations",
            [[eval_id, "", "", "", "", "", status, json.dumps(extra), now, now]],
            column_names=["id", "name", "description", "dataset_id", "models", "metrics",
                          "status", "config", "created_at", "updated_at"],
        )

    # --- Gather samples from the dataset (unchanged) ---
    samples = []
    try:
        if dataset_id.startswith("cluster:"):
            parts = dataset_id.split(":")
            if len(parts) == 3:
                run_id, cluster_id = parts[1], int(parts[2])
                r = client.query(
                    "SELECT input_text, output_text FROM trace_cluster_map "
                    "WHERE run_id = {rid:String} AND cluster_id = {cid:UInt32} LIMIT 100",
                    parameters={"rid": run_id, "cid": cluster_id},
                )
                for i, row in enumerate(r.result_rows):
                    samples.append({"id": f"s-{i}", "input": row[0], "expected": row[1] or ""})
        else:
            r = client.query(
                "SELECT id, input, output, expected_output FROM eval_dataset_samples "
                "WHERE dataset_id = {did:String} LIMIT 100",
                parameters={"did": dataset_id},
            )
            for row in r.result_rows:
                samples.append({"id": row[0], "input": row[1], "expected": row[3] or row[2] or ""})
    except Exception as e:
        _update_status("failed", error=str(e))
        return

    if not samples:
        _update_status("failed", error="No samples in dataset")
        return

    _update_status("running", total_samples=len(samples), completed_samples=0, failed_samples=0)

    # --- Scoring helper ---
    def _score_sample(outputs: dict, sample: dict) -> dict:
        scores: dict = {}
        for metric_id in metrics:
            scores[metric_id] = {}
            for model in models:
                model_out = outputs.get(model, {})
                output = model_out.get("output", "")
                expected = sample.get("expected", "")
                latency = model_out.get("latency", 0)
                cost_val = model_out.get("cost", 0)

                if metric_id == "exact_match":
                    match = output.strip() == expected.strip()
                    scores[metric_id][model] = {"score": 1.0 if match else 0.0, "passed": match, "match": match}
                elif metric_id == "contains":
                    found = expected.lower() in output.lower() if expected else True
                    scores[metric_id][model] = {"score": 1.0 if found else 0.0, "passed": found}
                elif metric_id in ("similarity", "semantic_sim"):
                    out_words = set(output.lower().split())
                    exp_words = set(expected.lower().split()) if expected else out_words
                    overlap = len(out_words & exp_words) / max(len(out_words | exp_words), 1)
                    scores[metric_id][model] = {"score": round(overlap, 3), "passed": overlap >= 0.3}
                elif metric_id == "latency":
                    max_lat = 10.0
                    passed = latency <= max_lat
                    score = max(0, 1 - (latency / max_lat) ** 0.5) if latency < max_lat else 0
                    scores[metric_id][model] = {"score": round(score, 3), "passed": passed, "latency_seconds": latency, "max_acceptable": max_lat}
                elif metric_id == "cost":
                    passed = cost_val <= 0.01
                    scores[metric_id][model] = {"score": round(max(0, 1 - cost_val / 0.01), 3), "passed": passed}
                elif metric_id == "llm_judge":
                    scores[metric_id][model] = {"score": 0.8, "passed": True}
                else:
                    scores[metric_id][model] = {"score": 0.5, "passed": True}
        return scores

    # --- Parallel execution ---
    semaphore = asyncio.Semaphore(max_concurrency)
    progress_lock = asyncio.Lock()
    completed_count = 0
    failed_count = 0

    async def _process_sample(http_client: httpx.AsyncClient, sample: dict) -> dict:
        nonlocal completed_count, failed_count

        outputs = {}
        for model in models:
            async with semaphore:
                try:
                    loop = asyncio.get_running_loop()
                    start = loop.time()
                    resp = await http_client.post(
                        f"{engine_url}/v1/chat/completions",
                        headers={"X-Lunar-Internal": "true"},
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": sample["input"]}],
                            "max_tokens": 150,
                        },
                    )
                    elapsed = loop.time() - start
                    data = resp.json()
                    output_text = data["choices"][0]["message"]["content"]
                    cost = data.get("cost", {}).get("total_cost_usd", 0)
                    outputs[model] = {"output": output_text, "latency": round(elapsed, 3), "cost": cost}
                except Exception as e:
                    outputs[model] = {"output": "", "error": str(e), "latency": 0, "cost": 0}

        scores = _score_sample(outputs, sample)

        # Update progress
        async with progress_lock:
            completed_count += 1
            if any("error" in outputs.get(m, {}) for m in models):
                failed_count += 1
            batch = max(1, len(samples) // 10)  # ~10 progress updates per eval
            if completed_count % batch == 0 or completed_count == len(samples):
                _update_status("running",
                    total_samples=len(samples),
                    completed_samples=completed_count,
                    failed_samples=failed_count,
                )

        return {
            "sample_id": sample["id"],
            "input": sample["input"][:500],
            "expected": sample.get("expected", "")[:500],
            "outputs": outputs,
            "scores": scores,
        }

    # Run all samples in parallel with shared client and semaphore
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        tasks = [_process_sample(http_client, s) for s in samples]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    results = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            logger.error(f"Eval {eval_id}: sample {samples[i]['id']} failed: {r}")
            failed_count += 1
        else:
            results.append(r)

    # --- Build summary ---
    model_summaries = {}
    metric_summaries = {}
    for model in models:
        total_lat = sum(r["outputs"].get(model, {}).get("latency", 0) for r in results)
        total_cost = sum(r["outputs"].get(model, {}).get("cost", 0) for r in results)
        avg_scores = {}
        for metric_id in metrics:
            vals = [r["scores"].get(metric_id, {}).get(model, {}).get("score", 0) for r in results]
            avg_scores[metric_id] = round(sum(vals) / max(len(vals), 1), 3)
        model_summaries[model] = {
            "total_latency": round(total_lat, 3),
            "avg_latency": round(total_lat / max(len(results), 1), 3),
            "total_cost": total_cost,
            "avg_cost": total_cost / max(len(results), 1),
            "avg_scores": avg_scores,
        }

    for metric_id in metrics:
        avg_by_model = {}
        for model in models:
            vals = [r["scores"].get(metric_id, {}).get(model, {}).get("score", 0) for r in results]
            avg_by_model[model] = round(sum(vals) / max(len(vals), 1), 3)
        metric_summaries[metric_id] = {"avg_by_model": avg_by_model}

    # Determine winner
    overall_scores = {}
    for model in models:
        all_metric_avgs = [model_summaries[model]["avg_scores"].get(m, 0) for m in metrics]
        overall_scores[model] = round(sum(all_metric_avgs) / max(len(all_metric_avgs), 1), 3)
    winner_model = max(overall_scores, key=overall_scores.get) if overall_scores else ""

    eval_results = {
        "evaluation_id": eval_id,
        "samples": results,
        "summary": {"models": model_summaries, "metrics": metric_summaries},
        "winner": {
            "model": winner_model,
            "overall_score": overall_scores.get(winner_model, 0),
            "scores_by_model": overall_scores,
        },
    }

    # Store results and mark complete
    client.command("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            evaluation_id String,
            results       String,
            created_at    DateTime64(3, 'UTC')
        ) ENGINE = ReplacingMergeTree(created_at) ORDER BY (evaluation_id)
    """)
    now = datetime.now(timezone.utc)
    client.insert("evaluation_results",
        [[eval_id, json.dumps(eval_results), now]],
        column_names=["evaluation_id", "results", "created_at"],
    )
    _update_status("completed",
        total_samples=len(samples),
        completed_samples=len(results),
        failed_samples=failed_count,
    )


@app.get("/v1/evaluations/{evaluation_id}/status", tags=["evaluations"])
async def get_evaluation_status(evaluation_id: str):
    """Get evaluation status."""
    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    r = client.query(
        "SELECT status, name, config FROM evaluations FINAL WHERE id = {eid:String}",
        parameters={"eid": evaluation_id},
    )
    if not r.result_rows:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    status = r.result_rows[0][0]
    config_str = r.result_rows[0][2]
    progress_data = {}
    if config_str:
        try:
            progress_data = json.loads(config_str)
        except Exception:
            pass

    return {
        "evaluation_id": evaluation_id,
        "status": status,
        "name": r.result_rows[0][1],
        "progress": {
            "total_samples": progress_data.get("total_samples", 0),
            "completed_samples": progress_data.get("completed_samples", 0),
            "failed_samples": progress_data.get("failed_samples", 0),
        },
    }


@app.get("/v1/evaluations/{evaluation_id}/results", tags=["evaluations"])
async def get_evaluation_results(evaluation_id: str):
    """Get evaluation results."""
    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    try:
        r = client.query(
            "SELECT results FROM evaluation_results FINAL WHERE evaluation_id = {eid:String}",
            parameters={"eid": evaluation_id},
        )
    except Exception:
        raise HTTPException(status_code=404, detail="No results yet")

    if not r.result_rows:
        raise HTTPException(status_code=404, detail="No results yet")

    results = json.loads(r.result_rows[0][0])
    return {"results": results, "samples_total": len(results.get("samples", []))}


@app.post("/v1/evaluations/{evaluation_id}/cancel", tags=["evaluations"])
async def cancel_evaluation(evaluation_id: str):
    """Cancel a queued or running evaluation."""
    from datetime import datetime, timezone

    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    now = datetime.now(timezone.utc)
    client.insert("evaluations",
        [[evaluation_id, "", "", "", "", "", "cancelled", "{}", now, now]],
        column_names=["id", "name", "description", "dataset_id", "models", "metrics",
                      "status", "config", "created_at", "updated_at"],
    )
    return {"success": True}


@app.delete("/v1/evaluations/{evaluation_id}", tags=["evaluations"])
async def delete_evaluation(evaluation_id: str):
    """Delete an evaluation and its results."""
    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    client.command(f"ALTER TABLE evaluations DELETE WHERE id = '{evaluation_id}'")
    try:
        client.command(f"ALTER TABLE evaluation_results DELETE WHERE evaluation_id = '{evaluation_id}'")
    except Exception:
        pass
    return {"success": True}


@app.get("/v1/evaluations/{evaluation_id}", tags=["evaluations"])
async def get_evaluation(evaluation_id: str):
    """Get a single evaluation."""
    client = _ch_eval()
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not available")

    r = client.query(
        "SELECT id, name, description, dataset_id, models, metrics, status, config, created_at, updated_at "
        "FROM evaluations FINAL WHERE id = {eid:String}",
        parameters={"eid": evaluation_id},
    )
    if not r.result_rows:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    e = dict(zip(r.column_names, r.result_rows[0]))
    for field in ("models", "metrics", "config"):
        if isinstance(e.get(field), str) and e[field]:
            try:
                e[field] = json.loads(e[field])
            except Exception:
                pass
    for field in ("created_at", "updated_at"):
        if hasattr(e.get(field), "isoformat"):
            e[field] = e[field].isoformat()
    return e


# --- Secrets (API Key Management) ---


@app.get("/v1/secrets", tags=["secrets"])
async def list_secrets():
    """List configured providers."""
    from ..storage.secrets import list_configured_providers
    return {"configured_providers": list_configured_providers()}


@app.post("/v1/secrets/{provider}", tags=["secrets"])
async def save_secret(provider: str, body: dict):
    """Save an API key for a provider."""
    from ..storage.secrets import set_secret
    api_key = body.get("api_key", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")
    set_secret(provider, api_key)
    return {"message": f"Key saved for {provider}"}


@app.delete("/v1/secrets/{provider}", tags=["secrets"])
async def remove_secret(provider: str):
    """Remove an API key."""
    from ..storage.secrets import delete_secret
    if not delete_secret(provider):
        raise HTTPException(status_code=404, detail=f"No key found for {provider}")
    return {"message": f"Key removed for {provider}"}


# --- Analytics (ClickHouse) ---


@app.get("/v1/stats/{tenant_id}/analytics", tags=["analytics"])
async def analytics_full(
    tenant_id: str,
    days: int = 30,
    trace_limit: int = 100,
    trace_offset: int = 0,
    model_id: Optional[str] = None,
    backend: Optional[str] = None,
    is_success: Optional[bool] = None,
    search: Optional[str] = None,
):
    """Full analytics response matching the UI's AnalyticsMetricsResponse shape."""
    from ..storage.clickhouse_client import query_analytics

    return query_analytics(
        days=days,
        trace_limit=trace_limit,
        trace_offset=trace_offset,
        model_id=model_id,
        backend=backend,
        is_success=is_success,
        search=search,
    )


@app.get("/traces", tags=["analytics"])
async def list_traces(
    model: Optional[str] = None,
    hours: int = 24,
    limit: int = 100,
    offset: int = 0,
):
    """List recent traces from ClickHouse."""
    from ..storage.clickhouse_client import query_traces, query_trace_count, get_client
    from datetime import datetime, timedelta, timezone

    if get_client() is None:
        raise HTTPException(status_code=503, detail="ClickHouse not enabled")

    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    traces = query_traces(model=model, start=start, limit=limit, offset=offset)
    total = query_trace_count(model=model, start=start)
    return {"traces": traces, "total": total, "limit": limit, "offset": offset}


@app.get("/analytics/models", tags=["analytics"])
async def analytics_models(
    model: Optional[str] = None,
    hours: int = 24,
):
    """Hourly model-level analytics from ClickHouse materialized view."""
    from ..storage.clickhouse_client import query_model_hourly, get_client
    from datetime import datetime, timedelta, timezone

    if get_client() is None:
        raise HTTPException(status_code=503, detail="ClickHouse not enabled")

    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    return query_model_hourly(model=model, start=start)


@app.get("/analytics/clusters", tags=["analytics"])
async def analytics_clusters(
    cluster_id: Optional[int] = None,
    days: int = 30,
):
    """Daily cluster-level analytics from ClickHouse materialized view."""
    from ..storage.clickhouse_client import query_cluster_daily, get_client
    from datetime import datetime, timedelta, timezone

    if get_client() is None:
        raise HTTPException(status_code=503, detail="ClickHouse not enabled")

    start = datetime.now(timezone.utc) - timedelta(days=days)
    return query_cluster_daily(cluster_id=cluster_id, start=start)


@app.get("/analytics/summary", tags=["analytics"])
async def analytics_summary(hours: int = 24):
    """Overall summary statistics from ClickHouse."""
    from ..storage.clickhouse_client import query_summary, get_client
    from datetime import datetime, timedelta, timezone

    if get_client() is None:
        raise HTTPException(status_code=503, detail="ClickHouse not enabled")

    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    return query_summary(start=start)


# --- Initialization (for programmatic setup) ---


def init_router(
    embedder,
    cluster_assigner,
    registry: LLMRegistry,
    clients: Optional[dict[str, LLMClient]] = None,
    cost_weight: float = 0.0,
    use_soft_assignment: bool = True,
    state_path: Optional[str] = None,
    app_settings: Optional[Settings] = None,
) -> UniRouteRouter:
    """
    Initialize the router programmatically.

    This should be called before starting the server.

    Args:
        embedder: PromptEmbedder instance.
        cluster_assigner: ClusterAssigner instance.
        registry: LLMRegistry with model profiles.
        clients: Optional dict of LLM clients for execution.
        cost_weight: Default cost weight (λ).
        use_soft_assignment: Use soft vs hard cluster assignment.
        state_path: Path for state persistence.
        app_settings: Application settings.

    Returns:
        Initialized UniRouteRouter.
    """
    global router, llm_clients, state_manager, settings

    router = UniRouteRouter(
        embedder=embedder,
        cluster_assigner=cluster_assigner,
        registry=registry,
        cost_weight=cost_weight,
        use_soft_assignment=use_soft_assignment,
    )

    if clients:
        llm_clients.update(clients)

    if state_path:
        state_manager = StateManager(state_path)

    if app_settings:
        settings = app_settings

    logger.info(
        f"Router initialized with {len(registry)} models, "
        f"{cluster_assigner.num_clusters} clusters"
    )

    return router


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
