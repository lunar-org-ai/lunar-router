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

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    KpiValue,
    EfficiencyResponse,
    ModelPerformanceResponse,
    TrainingActivityResponse,
    RoutingIntelligenceResponse,
    RoutingDecisionItem,
    WinRatePoint,
    ConfidenceBucket,
    EfficiencyTrendPoint,
    ModelUsageItem,
    DailyVolumePoint,
    LatencyPercentilesItem,
    ErrorBreakdownItem,
    AdvisorConfigResponse,
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

    # Run datasets ClickHouse migration
    try:
        _run_datasets_migration()
    except Exception as e:
        logger.warning(f"Datasets migration skipped: {e}")

    # Run evaluation tables migration
    try:
        _run_eval_migration()
    except Exception as e:
        logger.warning(f"Eval tables migration skipped: {e}")

    # Run distillation tables migration
    try:
        _run_distillation_migration()
    except Exception as e:
        logger.warning(f"Distillation migration skipped: {e}")

    # Seed built-in evaluation metrics
    try:
        from ..metrics.repository import seed_builtin_metrics
        seed_builtin_metrics()
        logger.info("Built-in evaluation metrics seeded")
    except Exception as e:
        logger.warning(f"Builtin metrics seeding skipped: {e}")

    # Run operator metadata migration (idempotent).
    try:
        _run_operator_migration()
    except Exception as e:
        logger.warning(f"Operator migration skipped: {e}")

    # Run eval_datasets tenant-alignment migration (idempotent).
    try:
        _run_sql_migration("011_eval_datasets_tenant.sql")
    except Exception as e:
        logger.warning(f"eval_datasets tenant migration skipped: {e}")

    # Start scan scheduler if enabled
    from ..harness.scheduler import get_scheduler

    scheduler = get_scheduler()
    if scheduler.config.enabled:
        scheduler.start()
        logger.info("Scan scheduler started on lifespan")

    # Start autonomous operator loop if enabled via env flag.
    operator = None
    if os.getenv("AUTO_OPERATOR", "false").lower() == "true":
        try:
            from ..harness.operator import get_operator

            operator = get_operator()
            operator.start()
            logger.info("OperatorLoop started on lifespan (AUTO_OPERATOR=true)")
        except Exception as e:
            logger.warning(f"OperatorLoop start failed: {e}")

    # Start auto-trainer background loop if enabled and weights are ready.
    training_manager = None
    try:
        from ..training.runtime import get_training_manager

        training_manager = get_training_manager()
        if os.getenv("AUTO_TRAINER", "false").lower() == "true":
            if training_manager.weights_ready:
                training_manager.start_scheduled(interval_seconds=3600)
                logger.info("AUTO_TRAINER=true: ScheduledTrainer started")
            else:
                logger.warning(
                    "AUTO_TRAINER=true but weights dir %s is empty; training disabled. "
                    "Run `python -m opentracy.scripts.bootstrap_weights` first.",
                    training_manager.weights_path,
                )
    except Exception as e:
        logger.warning(f"TrainingManager init failed: {e}")

    # Cleanup stale vLLM deployments from previous runs
    try:
        from ..deployment.manager import cleanup_stale_deployments
        await cleanup_stale_deployments()
        logger.info("Checked for stale deployments")
    except Exception as e:
        logger.debug(f"Deployment cleanup skipped: {e}")

    yield

    # Shutdown
    scheduler.stop()
    if operator is not None:
        try:
            operator.stop()
        except Exception as e:
            logger.debug(f"OperatorLoop stop error: {e}")
    if training_manager is not None:
        try:
            training_manager.stop_scheduled()
        except Exception as e:
            logger.debug(f"TrainingManager stop error: {e}")
    logger.info("UniRoute API shutting down...")


def _run_eval_migration():
    """Execute the eval tables ClickHouse migration if not yet applied."""
    import pathlib
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return
    migration_file = pathlib.Path(__file__).resolve().parents[2] / "clickhouse" / "008_create_eval_tables.sql"
    if not migration_file.exists():
        logger.debug("Eval migration file not found: %s", migration_file)
        return
    sql = migration_file.read_text()
    for statement in sql.split(";"):
        stmt = statement.strip()
        if stmt:
            try:
                client.command(stmt)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("Eval migration statement failed: %s — %s", stmt[:80], e)


def _run_distillation_migration():
    """Execute the distillation tables ClickHouse migration if not yet applied."""
    import pathlib
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return
    migration_file = pathlib.Path(__file__).resolve().parents[2] / "clickhouse" / "009_create_distillation_tables.sql"
    if not migration_file.exists():
        logger.debug("Distillation migration file not found: %s", migration_file)
        return
    sql = migration_file.read_text()
    for statement in sql.split(";"):
        stmt = statement.strip()
        if stmt:
            try:
                client.command(stmt)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("Distillation migration statement failed: %s — %s", stmt[:80], e)


def _run_datasets_migration():
    """Execute the datasets ClickHouse migration if not yet applied."""
    import pathlib
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return
    migration_file = pathlib.Path(__file__).resolve().parents[2] / "clickhouse" / "007_create_eval_datasets.sql"
    if not migration_file.exists():
        logger.debug("Datasets migration file not found: %s", migration_file)
        return
    sql = migration_file.read_text()
    for statement in sql.split(";"):
        stmt = statement.strip()
        if stmt:
            try:
                client.command(stmt)
            except Exception as e:
                # Ignore "already exists" errors
                if "already exists" not in str(e).lower():
                    logger.warning("Migration statement failed: %s — %s", stmt[:80], e)


def _run_sql_migration(filename: str):
    """Execute a ClickHouse SQL migration file; ignore idempotent-failure errors."""
    import pathlib
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return
    migration_file = pathlib.Path(__file__).resolve().parents[2] / "clickhouse" / filename
    if not migration_file.exists():
        logger.debug("Migration file not found: %s", migration_file)
        return
    sql = migration_file.read_text()
    # Strip SQL line comments so a stray `;` in a comment doesn't split badly.
    cleaned = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    for statement in cleaned.split(";"):
        stmt = statement.strip()
        if not stmt:
            continue
        try:
            client.command(stmt)
        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ("already exists", "already has column", "cannot add column", "no such column")):
                continue
            logger.warning("Migration %s statement failed: %s — %s", filename, stmt[:80], e)


def _run_operator_migration():
    """Apply operator metadata columns + operator_decisions table (idempotent)."""
    import pathlib
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return
    migration_file = pathlib.Path(__file__).resolve().parents[2] / "clickhouse" / "010_operator_metadata.sql"
    if not migration_file.exists():
        logger.debug("Operator migration file not found: %s", migration_file)
        return
    sql = migration_file.read_text()
    for statement in sql.split(";"):
        stmt = statement.strip()
        if stmt:
            try:
                client.command(stmt)
            except Exception as e:
                msg = str(e).lower()
                if "already exists" in msg or "already has column" in msg or "cannot add column" in msg:
                    continue
                logger.warning("Operator migration statement failed: %s — %s", stmt[:80], e)


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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler so that 500s still carry CORS headers."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# --- Mount vLLM deployment router ---
from ..deployment.routes import deployment_router
from ..deployment.proxy import proxy_router as deployment_proxy_router

app.include_router(deployment_router, prefix="/v1/deployments", tags=["deployments"])
app.include_router(deployment_proxy_router, prefix="/v1/deployments", tags=["deployments"])


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


# ---------------------------------------------------------------------------
# Operator loop endpoints (Phase 1 of autonomous curation)
# ---------------------------------------------------------------------------

@app.get("/v1/operator/status", tags=["operator"])
async def operator_status():
    """Return operator loop running state + next tick info."""
    from ..harness.operator import get_operator

    op = get_operator()
    return op.get_status()


@app.get("/v1/operator/decisions", tags=["operator"])
async def operator_decisions(limit: int = 50):
    """Return recent operator tick decisions (most recent first)."""
    from ..harness.operator import get_operator

    op = get_operator()
    return {"decisions": op.list_decisions(limit=limit)}


@app.post("/v1/operator/pause", tags=["operator"])
async def operator_pause():
    from ..harness.operator import get_operator

    op = get_operator()
    op.pause()
    return op.get_status()


@app.post("/v1/operator/resume", tags=["operator"])
async def operator_resume():
    from ..harness.operator import get_operator

    op = get_operator()
    op.resume()
    return op.get_status()


# ---------------------------------------------------------------------------
# Training endpoints (continuous auto-training)
# ---------------------------------------------------------------------------

@app.get("/v1/training/status", tags=["training"])
async def training_status():
    from ..training.runtime import get_training_manager

    return get_training_manager().status()


@app.get("/v1/training/runs", tags=["training"])
async def training_runs(limit: int = 20):
    from ..training.runtime import get_training_manager

    return {"runs": get_training_manager().history(limit=limit)}


@app.post("/v1/training/run_now", tags=["training"])
async def training_run_now(request: Request):
    if request.headers.get("x-admin", "").lower() != "true":
        raise HTTPException(status_code=403, detail="X-Admin: true header required")
    from ..training.runtime import get_training_manager

    tenant = request.headers.get("x-tenant-id", "default")
    tm = get_training_manager()
    result = await tm.run_once(trigger="manual", tenant_id=tenant)
    return result


@app.post("/v1/training/pause", tags=["training"])
async def training_pause():
    from ..training.runtime import get_training_manager

    tm = get_training_manager()
    tm.pause()
    return tm.status()


@app.post("/v1/training/resume", tags=["training"])
async def training_resume():
    from ..training.runtime import get_training_manager

    tm = get_training_manager()
    tm.resume()
    return tm.status()


# ---------------------------------------------------------------------------
# Dataset curation endpoints (human approval of operator proposals)
# ---------------------------------------------------------------------------

@app.get("/v1/datasets/pending", tags=["datasets"])
async def list_pending_datasets(request: Request):
    """Return datasets awaiting human curation review."""
    from ..datasets import repository as ds_repo

    tenant = request.headers.get("x-tenant-id", "default")
    try:
        rows = ds_repo.list_pending_datasets(tenant)
    except Exception as e:
        # eval_datasets schema may not match (pre-existing Go vs Python table
        # mismatch — TODO(operator-phase-2): unify schemas).
        logger.warning("list_pending_datasets failed: %s", e)
        rows = []
    for d in rows:
        d["id"] = d.get("dataset_id") or d.get("id", "")
    return {"datasets": rows, "total": len(rows)}


@app.post("/v1/datasets/{dataset_id}/approve", tags=["datasets"])
async def approve_dataset(dataset_id: str, request: Request):
    from ..datasets import repository as ds_repo

    tenant = request.headers.get("x-tenant-id", "default")
    ds = ds_repo.approve_dataset(tenant, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ds["id"] = ds.get("dataset_id") or dataset_id
    return ds


@app.post("/v1/datasets/{dataset_id}/reject", tags=["datasets"])
async def reject_dataset(dataset_id: str, request: Request):
    from ..datasets import repository as ds_repo

    tenant = request.headers.get("x-tenant-id", "default")
    ds = ds_repo.reject_dataset(tenant, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ds["id"] = ds.get("dataset_id") or dataset_id
    return ds


@app.post("/v1/datasets/{dataset_id}/samples/{sample_id}/approve", tags=["datasets"])
async def approve_sample(dataset_id: str, sample_id: str, request: Request):
    from ..datasets import repository as ds_repo

    tenant = request.headers.get("x-tenant-id", "default")
    s = ds_repo.approve_sample(tenant, dataset_id, sample_id)
    if not s:
        raise HTTPException(status_code=404, detail="Sample not found")
    return s


@app.post("/v1/datasets/{dataset_id}/samples/{sample_id}/reject", tags=["datasets"])
async def reject_sample(dataset_id: str, sample_id: str, request: Request):
    from ..datasets import repository as ds_repo

    tenant = request.headers.get("x-tenant-id", "default")
    s = ds_repo.reject_sample(tenant, dataset_id, sample_id)
    if not s:
        raise HTTPException(status_code=404, detail="Sample not found")
    return s


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
    import json as _json
    import numpy as _np
    from ..clustering.pipeline import ClusteringPipeline

    class _NumpyEncoder(_json.JSONEncoder):
        """Safety-net encoder that converts any remaining numpy scalars."""
        def default(self, obj):
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            return super().default(obj)

    pipeline = ClusteringPipeline(strategy=strategy)
    result = await pipeline.run(days=days, min_traces=min_traces)
    payload = result.to_dict()
    # Re-serialize through NumpyEncoder to catch any stray numpy types
    safe = _json.loads(_json.dumps(payload, cls=_NumpyEncoder))
    return safe


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

    engine_url = env("ENGINE_URL", "http://localhost:8080")
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

    engine_url = env("ENGINE_URL", "http://localhost:8080")
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

    engine_url = env("ENGINE_URL", "http://localhost:8080")
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

    engine_url = env("ENGINE_URL", "http://localhost:8080")
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
    engine_url = env("ENGINE_URL", "http://localhost:8080")
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
    """Create eval_datasets / eval_samples if they don't exist (matches migration 007)."""
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return
    client.command("""
        CREATE TABLE IF NOT EXISTS eval_datasets (
            dataset_id      String,
            tenant_id       String       DEFAULT 'default',
            name            String,
            description     String       DEFAULT '',
            source          LowCardinality(String)  DEFAULT 'manual',
            samples_count   UInt32       DEFAULT 0,
            created_at      DateTime64(3, 'UTC') DEFAULT now64(3),
            updated_at      DateTime64(3, 'UTC') DEFAULT now64(3)
        ) ENGINE = ReplacingMergeTree(updated_at) ORDER BY (tenant_id, dataset_id)
    """)
    client.command("""
        CREATE TABLE IF NOT EXISTS eval_samples (
            sample_id       String,
            dataset_id      String,
            tenant_id       String       DEFAULT 'default',
            input           String       DEFAULT '',
            expected_output String       DEFAULT '',
            metadata        String       DEFAULT '{}',
            trace_id        String       DEFAULT '',
            created_at      DateTime64(3, 'UTC') DEFAULT now64(3)
        ) ENGINE = ReplacingMergeTree(created_at) ORDER BY (tenant_id, dataset_id, sample_id)
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
async def list_datasets(request: Request):
    """List all datasets: evaluation datasets + clustering domain datasets."""
    from ..datasets import repository as ds_repo
    tenant = request.headers.get("x-tenant-id", "default")

    datasets = []

    # 1) Evaluation datasets
    try:
        rows = ds_repo.list_datasets(tenant)
        for d in rows:
            # Normalise id field for the frontend
            d["id"] = d.get("dataset_id") or d.get("id", "")
            datasets.append(d)
    except Exception as e:
        logger.warning("Failed to query eval_datasets: %s", e)

    # 2) Clustering domain datasets (from latest run)
    try:
        client = _ch_eval()
        if client:
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
async def create_dataset(body: dict, request: Request):
    """Create a new evaluation dataset."""
    from ..datasets import repository as ds_repo
    tenant = request.headers.get("x-tenant-id", "default")

    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="'name' is required")

    ds = ds_repo.create_dataset(
        tenant,
        name=name,
        description=body.get("description", ""),
        source=body.get("source", "manual"),
    )
    ds["id"] = ds["dataset_id"]
    return ds


@app.get("/v1/datasets/{dataset_id}", tags=["datasets"])
async def get_dataset(
    dataset_id: str,
    request: Request,
    include_samples: bool = True,
    samples_limit: int = 1000,
    samples_offset: int = 0,
):
    """Get a single dataset by ID.

    Defaults: ``include_samples=True`` and ``samples_limit=1000`` so that any
    caller hitting this endpoint without options gets a usable response.
    Pass ``include_samples=false`` explicitly when only metadata is needed
    (e.g. list-page auxiliary lookups). Prior defaults (``False``/``50``)
    silently returned empty ``samples`` arrays and caused the dataset detail
    page to render "No samples yet" for populated datasets.
    """
    from ..datasets import repository as ds_repo
    tenant = request.headers.get("x-tenant-id", "default")

    ds = ds_repo.get_dataset(tenant, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ds["id"] = ds.get("dataset_id") or dataset_id

    samples = []
    samples_total = 0
    if include_samples:
        sample_rows = ds_repo.get_samples(tenant, dataset_id, limit=samples_limit, offset=samples_offset)
        for s in sample_rows:
            s["id"] = s.get("sample_id") or s.get("id", "")
            samples.append(s)

    samples_total = ds_repo.get_samples_count(tenant, dataset_id)
    return {"dataset": ds, "samples": samples, "samples_total": samples_total}


@app.delete("/v1/datasets/{dataset_id}", tags=["datasets"])
async def delete_dataset(dataset_id: str, request: Request):
    """Delete a dataset and its samples."""
    from ..datasets import repository as ds_repo
    tenant = request.headers.get("x-tenant-id", "default")
    ds_repo.delete_dataset(tenant, dataset_id)
    return {"success": True}


@app.post("/v1/datasets/{dataset_id}/samples", tags=["datasets"])
async def add_samples(dataset_id: str, body: dict, request: Request):
    """Add samples to an existing dataset."""
    from ..datasets import repository as ds_repo
    tenant = request.headers.get("x-tenant-id", "default")

    samples_input = body.get("samples", [])
    if not samples_input:
        raise HTTPException(status_code=400, detail="'samples' is required")

    count = ds_repo.add_samples(tenant, dataset_id, samples_input)
    return {"message": f"Added {count} samples", "count": count}


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

    engine_url = env("ENGINE_URL", "http://localhost:8080")
    max_concurrency = int(env("EVAL_CONCURRENCY", "10"))

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
                "SELECT sample_id, input, expected_output FROM eval_samples "
                "WHERE dataset_id = {did:String} LIMIT 100",
                parameters={"did": dataset_id},
            )
            for row in r.result_rows:
                samples.append({"id": row[0], "input": row[1], "expected": row[2] or ""})
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
                        headers={"X-OpenTracy-Internal": "true"},
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


@app.get("/v1/traces", tags=["traces"])
async def list_traces_v1(
    limit: int = 100,
    offset: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: Optional[str] = None,
    model_id: Optional[str] = None,
):
    """List traces for evaluations UI (v1 API)."""
    from ..storage.clickhouse_client import query_traces, query_trace_count, get_client
    from datetime import datetime, timedelta, timezone

    if get_client() is None:
        raise HTTPException(status_code=503, detail="ClickHouse not enabled")

    start = None
    end = None
    if start_date:
        try:
            start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            start = datetime.now(timezone.utc) - timedelta(hours=24)
    else:
        start = datetime.now(timezone.utc) - timedelta(hours=24)
    if end_date:
        try:
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            pass

    traces = query_traces(model=model_id, start=start, end=end, limit=limit, offset=offset)
    total = query_trace_count(model=model_id, start=start, end=end)
    return {"traces": traces, "total": total, "has_more": (offset + limit) < total}


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


# --- Router Intelligence ---


@app.get("/v1/intelligence/efficiency", response_model=EfficiencyResponse, tags=["intelligence"])
async def intelligence_efficiency(days: int = 30):
    """Router efficiency: cost savings, quality, model distribution."""
    from datetime import datetime, timedelta, timezone

    empty = EfficiencyResponse(
        kpis={
            "cost_saved": KpiValue(value=0),
            "quality_score": KpiValue(value=0),
            "avg_cost_per_request": KpiValue(value=0),
            "requests_routed": KpiValue(value=0),
        },
    )

    # Try to get data from the in-memory router stats first
    has_router = False
    try:
        r = get_router()
        stats = r.stats
        profiles = r.registry.get_all()
        has_router = bool(profiles)
    except Exception:
        pass

    # Safely get ClickHouse client
    from ..storage.clickhouse_client import get_client
    try:
        ch = get_client()
    except Exception:
        ch = None

    # Compute from in-memory router stats if available
    model_breakdown = []

    if has_router and stats.total_requests > 0:
        profile_map = {p.model_id: p for p in profiles}
        max_cost = max(p.cost_per_1k_tokens for p in profiles)
        total_reqs = stats.total_requests
        actual_cost = 0.0
        for model_id, count in stats.model_selections.items():
            p = profile_map.get(model_id)
            cost = p.cost_per_1k_tokens if p else 0.0
            accuracy = (1.0 - p.overall_error_rate) if p else 0.0
            actual_cost += cost * count
            traffic_pct = (count / total_reqs * 100) if total_reqs > 0 else 0.0
            model_breakdown.append({
                "model": model_id, "requests": count,
                "accuracy": round(accuracy, 4), "avg_cost": round(cost, 8),
                "traffic_pct": round(traffic_pct, 1),
            })
        baseline_cost = max_cost * total_reqs
        cost_saved = baseline_cost - actual_cost
        avg_cost = actual_cost / total_reqs
        quality = 1.0 - stats.avg_expected_error
        kpis = {
            "cost_saved": KpiValue(value=round(cost_saved, 6)),
            "quality_score": KpiValue(value=round(quality, 4)),
            "avg_cost_per_request": KpiValue(value=round(avg_cost, 8)),
            "requests_routed": KpiValue(value=total_reqs),
        }

    elif ch is not None:
        # Fallback: compute from ClickHouse traces directly
        try:
            start = datetime.now(timezone.utc) - timedelta(days=days)
            r_summary = ch.query(
                "SELECT selected_model, count() AS cnt, sum(total_cost_usd) AS total_cost, "
                "avg(total_cost_usd) AS avg_cost "
                "FROM llm_traces WHERE timestamp >= {start:DateTime64(3)} "
                "GROUP BY selected_model ORDER BY cnt DESC",
                parameters={"start": start},
            )
            total_reqs = 0
            total_cost = 0.0
            max_cost_per_req = 0.0
            for row in r_summary.result_rows:
                model, cnt, model_total, model_avg = str(row[0]), int(row[1]), float(row[2]), float(row[3])
                total_reqs += cnt
                total_cost += model_total
                if model_avg > max_cost_per_req:
                    max_cost_per_req = model_avg
                model_breakdown.append({
                    "model": model, "requests": cnt,
                    "accuracy": 0, "avg_cost": round(model_avg, 8),
                    "traffic_pct": 0,
                })
            # Compute traffic percentages
            for row in model_breakdown:
                row["traffic_pct"] = round(row["requests"] / total_reqs * 100, 1) if total_reqs > 0 else 0
            avg_cost = total_cost / total_reqs if total_reqs > 0 else 0
            baseline_cost = max_cost_per_req * total_reqs
            cost_saved = baseline_cost - total_cost
            kpis = {
                "cost_saved": KpiValue(value=round(cost_saved, 6)),
                "quality_score": KpiValue(value=0),
                "avg_cost_per_request": KpiValue(value=round(avg_cost, 8)),
                "requests_routed": KpiValue(value=total_reqs),
            }
        except Exception:
            return empty
    else:
        return empty

    # ClickHouse time-series data (optional)
    model_distribution = []
    cost_savings_trend = []
    if ch is not None:
        try:
            start = datetime.now(timezone.utc) - timedelta(days=days)
            r_dist = ch.query(
                "SELECT toDate(timestamp) AS day, selected_model, count() AS cnt "
                "FROM llm_traces WHERE timestamp >= {start:DateTime64(3)} "
                "GROUP BY day, selected_model ORDER BY day",
                parameters={"start": start},
            )
            # Pivot into [{date, model_a: N, model_b: N}, ...]
            day_map: dict[str, dict] = {}
            for row in r_dist.result_rows:
                day_str = str(row[0])
                model = str(row[1])
                cnt = int(row[2])
                if day_str not in day_map:
                    day_map[day_str] = {"date": day_str}
                day_map[day_str][model] = cnt
            model_distribution = list(day_map.values())

            # Get the most expensive model's avg cost for baseline calculation
            r_max = ch.query(
                "SELECT max(avg_cost) FROM ("
                "  SELECT avg(total_cost_usd / greatest(total_tokens, 1) * 1000) AS avg_cost "
                "  FROM llm_traces WHERE timestamp >= {start:DateTime64(3)} "
                "  GROUP BY selected_model"
                ")",
                parameters={"start": start},
            )
            max_model_avg_cost = float(r_max.result_rows[0][0]) if r_max.result_rows else 0

            r_cost = ch.query(
                "SELECT toDate(timestamp) AS day, "
                "  sum(total_cost_usd) AS actual, "
                "  count() AS cnt, "
                "  sum(total_tokens) AS tokens "
                "FROM llm_traces WHERE timestamp >= {start:DateTime64(3)} "
                "GROUP BY day ORDER BY day",
                parameters={"start": start},
            )
            for row in r_cost.result_rows:
                day_str = str(row[0])
                actual_day = float(row[1])
                day_cnt = int(row[2])
                day_tokens = int(row[3])
                # Baseline = what it would cost if all tokens used most expensive model
                baseline_day = max_model_avg_cost * day_tokens / 1000 if max_model_avg_cost > 0 else actual_day * 2
                cost_savings_trend.append({
                    "date": day_str,
                    "actual": round(actual_day, 6),
                    "baseline": round(baseline_day, 6),
                    "saved": round(baseline_day - actual_day, 6),
                })
        except Exception:
            pass

    from .schemas import CostBreakdown, DistillationJobSummary
    import json as _json

    cost_breakdown = None
    distillation_job_summaries: list[DistillationJobSummary] = []

    # Compute accumulated provider baseline and router actual from the data we already have
    provider_baseline = 0.0
    routing_actual = 0.0
    if cost_savings_trend:
        provider_baseline = sum(d.get("baseline", 0) for d in cost_savings_trend)
        routing_actual = sum(d.get("actual", 0) for d in cost_savings_trend)
    elif kpis:
        provider_baseline = kpis["cost_saved"].value + (kpis["avg_cost_per_request"].value * kpis["requests_routed"].value)
        routing_actual = kpis["avg_cost_per_request"].value * kpis["requests_routed"].value

    routing_savings = provider_baseline - routing_actual

    # Fetch distillation jobs for training cost
    training_investment = 0.0
    if ch is not None:
        try:
            r_jobs = ch.query(
                "SELECT job_id, name, status, config, cost_accrued, "
                "toString(created_at) AS created_at, "
                "toString(completed_at) AS completed_at "
                "FROM distillation_jobs FINAL "
                "ORDER BY created_at DESC LIMIT 20"
            )
            for row in r_jobs.result_rows:
                job_id_val = str(row[0])
                job_name = str(row[1])
                job_status = str(row[2])
                job_config_raw = row[3]
                job_cost = float(row[4])
                job_created = str(row[5])
                job_completed = str(row[6]) if row[6] else None

                training_investment += job_cost

                # Parse config to get teacher/student models
                teacher_m = ""
                student_m = ""
                try:
                    cfg = _json.loads(job_config_raw) if isinstance(job_config_raw, str) else job_config_raw
                    teacher_m = cfg.get("teacher_model", "")
                    student_m = cfg.get("student_model", "")
                except Exception:
                    pass

                distillation_job_summaries.append(DistillationJobSummary(
                    job_id=job_id_val,
                    name=job_name,
                    status=job_status,
                    teacher_model=teacher_m,
                    student_model=student_m,
                    cost_accrued=job_cost,
                    created_at=job_created,
                    completed_at=job_completed,
                ))
        except Exception:
            pass

    net_savings = routing_savings - training_investment
    roi_pct = (net_savings / training_investment * 100) if training_investment > 0 else 0.0
    # Monthly projection: extrapolate from the period
    monthly_projection = (routing_savings / days * 30) if days > 0 else 0.0

    cost_breakdown = CostBreakdown(
        provider_baseline=round(provider_baseline, 6),
        routing_actual=round(routing_actual, 6),
        routing_savings=round(routing_savings, 6),
        training_investment=round(training_investment, 6),
        net_savings=round(net_savings, 6),
        roi_pct=round(roi_pct, 1),
        monthly_projection=round(monthly_projection, 6),
    )

    return EfficiencyResponse(
        kpis=kpis,
        model_distribution=model_distribution,
        cost_savings_trend=cost_savings_trend,
        model_breakdown=sorted(model_breakdown, key=lambda x: -x["requests"]),
        cost_breakdown=cost_breakdown,
        distillation_jobs=distillation_job_summaries,
    )


@app.get("/v1/intelligence/models", response_model=ModelPerformanceResponse, tags=["intelligence"])
async def intelligence_models():
    """Model performance: profiles, cluster accuracy, leaderboard."""
    try:
        r = get_router()
    except Exception:
        r = None

    profiles = r.registry.get_all() if r else []
    if not profiles:
        return _build_model_perf_from_ch()

    # KPIs
    best = min(profiles, key=lambda p: p.overall_error_rate)
    cheapest = min(profiles, key=lambda p: p.cost_per_1k_tokens)
    best_value = max(
        profiles,
        key=lambda p: (1 - p.overall_error_rate) / max(p.cost_per_1k_tokens, 1e-10),
    )

    kpis = {
        "models_profiled": len(profiles),
        "best_model": {"model": best.model_id, "accuracy": round(best.overall_accuracy, 4)},
        "cheapest_model": {"model": cheapest.model_id, "cost": cheapest.cost_per_1k_tokens},
        "best_value": {
            "model": best_value.model_id,
            "ratio": round((1 - best_value.overall_error_rate) / max(best_value.cost_per_1k_tokens, 1e-10), 1),
        },
    }

    # Cluster accuracy from Psi vectors
    cluster_accuracy = []
    for p in profiles:
        clusters = {}
        for i, err in enumerate(p.psi_vector.tolist()):
            clusters[str(i)] = round(1.0 - err, 4)
        cluster_accuracy.append({"model": p.model_id, "clusters": clusters})

    # Leaderboard
    leaderboard = []
    for p in profiles:
        strong = [c for c, _ in p.strongest_clusters(3)]
        weak = [c for c, _ in p.weakest_clusters(3)]
        leaderboard.append({
            "model": p.model_id,
            "accuracy": round(p.overall_accuracy, 4),
            "cost": p.cost_per_1k_tokens,
            "strongest_clusters": strong,
            "weakest_clusters": weak,
        })
    leaderboard.sort(key=lambda x: -x["accuracy"])

    # Teacher-student comparison (from distillation jobs if available)
    teacher_student = None
    from ..storage.clickhouse_client import get_client
    ch = get_client()
    if ch is not None:
        try:
            r_dist = ch.query(
                "SELECT config, status FROM distillation_jobs "
                "WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1"
            )
            if r_dist.result_rows:
                import json as _json
                cfg = _json.loads(r_dist.result_rows[0][0])
                teacher = cfg.get("teacher_model", "")
                student = cfg.get("student_model", "")
                # Try exact match first, then strip provider prefix (e.g. openai/gpt-4o-mini → gpt-4o-mini)
                teacher_profile = r.registry.get(teacher)
                if not teacher_profile and "/" in teacher:
                    teacher_profile = r.registry.get(teacher.split("/", 1)[1])
                if teacher_profile:
                    teacher_student = {
                        "teacher": teacher,
                        "student": student,
                        "teacher_accuracy": round(teacher_profile.overall_accuracy, 4),
                        "teacher_cost": teacher_profile.cost_per_1k_tokens,
                    }
        except Exception:
            pass

    return ModelPerformanceResponse(
        kpis=kpis,
        cluster_accuracy=cluster_accuracy,
        leaderboard=leaderboard,
        teacher_student=teacher_student,
    )


def _build_model_perf_from_ch() -> "ModelPerformanceResponse":
    """Fallback: build model performance data from ClickHouse when router registry is empty."""
    from ..storage.clickhouse_client import get_client as _get_ch
    ch = _get_ch()
    if ch is None:
        return ModelPerformanceResponse()

    try:
        # Per-model stats from llm_traces
        r_models = ch.query(
            "SELECT selected_model, "
            "count() AS reqs, "
            "countIf(is_error = 1) AS errs, "
            "avg(total_cost_usd) AS avg_cost, "
            "sum(total_cost_usd) AS total_cost "
            "FROM llm_traces "
            "GROUP BY selected_model ORDER BY reqs DESC"
        )
        if not r_models.result_rows:
            return ModelPerformanceResponse()

        models_data = []
        for row in r_models.result_rows:
            model = str(row[0])
            reqs = int(row[1])
            errs = int(row[2])
            avg_cost = float(row[3])
            total_cost = float(row[4])
            accuracy = round(1.0 - (errs / max(reqs, 1)), 4)
            models_data.append({
                "model": model, "reqs": reqs, "errs": errs,
                "avg_cost": avg_cost, "total_cost": total_cost,
                "accuracy": accuracy,
            })

        best = max(models_data, key=lambda m: m["accuracy"])
        cheapest = min(models_data, key=lambda m: m["avg_cost"])
        best_value = max(
            models_data,
            key=lambda m: m["accuracy"] / max(m["avg_cost"], 1e-10),
        )

        kpis = {
            "models_profiled": len(models_data),
            "best_model": {"model": best["model"], "accuracy": best["accuracy"]},
            "cheapest_model": {"model": cheapest["model"], "cost": round(cheapest["avg_cost"], 6)},
            "best_value": {
                "model": best_value["model"],
                "ratio": round(best_value["accuracy"] / max(best_value["avg_cost"], 1e-10), 1),
            },
        }

        # Per-model per-cluster accuracy from llm_traces
        cluster_accuracy = []
        try:
            r_clust = ch.query(
                "SELECT selected_model, cluster_id, "
                "count() AS reqs, countIf(is_error = 1) AS errs "
                "FROM llm_traces GROUP BY selected_model, cluster_id "
                "ORDER BY selected_model, cluster_id"
            )
            model_clusters: dict[str, dict[str, float]] = {}
            for row in r_clust.result_rows:
                m = str(row[0])
                c = str(int(row[1]))
                reqs = int(row[2])
                errs = int(row[3])
                acc = round(1.0 - (errs / max(reqs, 1)), 4)
                if m not in model_clusters:
                    model_clusters[m] = {}
                model_clusters[m][c] = acc
            for m, clusters in model_clusters.items():
                cluster_accuracy.append({"model": m, "clusters": clusters})
        except Exception:
            pass

        # Leaderboard
        leaderboard = []
        for m in models_data:
            model_name = m["model"]
            # Find best/worst clusters for this model
            ca = next((c for c in cluster_accuracy if c["model"] == model_name), None)
            strong: list[int] = []
            weak: list[int] = []
            if ca and ca["clusters"]:
                sorted_c = sorted(ca["clusters"].items(), key=lambda x: -x[1])
                strong = [int(c) for c, _ in sorted_c[:3]]
                weak = [int(c) for c, _ in sorted_c[-3:]]
            leaderboard.append({
                "model": model_name,
                "accuracy": m["accuracy"],
                "cost": round(m["avg_cost"], 6),
                "strongest_clusters": strong,
                "weakest_clusters": weak,
            })
        leaderboard.sort(key=lambda x: -x["accuracy"])

        # Teacher-student from latest distillation
        teacher_student = None
        try:
            import json as _json
            r_dist = ch.query(
                "SELECT config FROM distillation_jobs "
                "WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1"
            )
            if r_dist.result_rows:
                cfg = _json.loads(r_dist.result_rows[0][0]) if isinstance(r_dist.result_rows[0][0], str) else {}
                teacher = cfg.get("teacher_model", "")
                student = cfg.get("student_model", "")
                if teacher:
                    # Try exact match, then strip provider prefix (openai/gpt-4o-mini → gpt-4o-mini)
                    t_data = next((m for m in models_data if m["model"] == teacher), None)
                    if not t_data and "/" in teacher:
                        bare = teacher.split("/", 1)[1]
                        t_data = next((m for m in models_data if m["model"] == bare), None)
                    teacher_student = {
                        "teacher": teacher,
                        "student": student,
                        "teacher_accuracy": t_data["accuracy"] if t_data else 0,
                        "teacher_cost": round(t_data["avg_cost"], 6) if t_data else 0,
                    }
        except Exception:
            pass

        return ModelPerformanceResponse(
            kpis=kpis,
            cluster_accuracy=cluster_accuracy,
            leaderboard=leaderboard,
            teacher_student=teacher_student,
        )
    except Exception:
        return ModelPerformanceResponse()


def _build_training_runs_detail(ch, training_history: list[dict]) -> list[dict]:
    """Build detailed training runs from distillation_jobs with real duration/confidence."""
    runs: list[dict] = []
    if ch is None:
        for i, h in enumerate(training_history):
            runs.append({
                "run_id": f"run_{i+1:03d}",
                "name": "Training run",
                "date": h.get("date", ""),
                "outcome": "promoted" if h.get("promoted") else "rejected",
                "confidence": 0,
                "cost": 0,
                "duration": "—",
                "reason": h.get("reason", ""),
                "teacher_model": "",
                "student_model": "",
                "quality_score": 0,
                "status": "completed" if h.get("promoted") else "failed",
            })
        return runs

    try:
        import json as _json
        from datetime import datetime

        r_jobs = ch.query(
            "SELECT job_id, name, status, config, cost_accrued, results, "
            "toString(created_at) AS created_at, "
            "toString(completed_at) AS completed_at "
            "FROM distillation_jobs FINAL "
            "ORDER BY created_at DESC LIMIT 30"
        )

        # Batch-fetch metrics for all jobs at once
        all_job_ids = [str(row[0]) for row in r_jobs.result_rows]
        metrics_map: dict[str, float] = {}
        if all_job_ids:
            try:
                r_met = ch.query(
                    "SELECT job_id, argMax(reward_improvement, step) AS best_reward "
                    "FROM distillation_metrics "
                    "WHERE job_id IN ({jids:Array(String)}) "
                    "GROUP BY job_id",
                    parameters={"jids": all_job_ids},
                )
                for mrow in r_met.result_rows:
                    metrics_map[str(mrow[0])] = float(mrow[1])
            except Exception:
                pass

        for row in r_jobs.result_rows:
            job_id = str(row[0])
            name = str(row[1]) or "Distillation run"
            status = str(row[2])
            config_raw = row[3]
            cost = float(row[4])
            results_raw = row[5]
            created_at = str(row[6])
            completed_at = str(row[7]) if row[7] else None

            # Compute real duration with second-level precision
            duration = "—"
            if completed_at and created_at and completed_at != "" and created_at != "":
                try:
                    dt_start = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    dt_end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                    total_secs = round((dt_end - dt_start).total_seconds())
                    if total_secs <= 0:
                        duration = "—"
                    elif total_secs < 60:
                        duration = f"{total_secs}s"
                    elif total_secs < 3600:
                        mins = total_secs // 60
                        secs = total_secs % 60
                        duration = f"{mins}m {secs}s" if secs > 0 else f"{mins}m"
                    else:
                        hrs = total_secs // 3600
                        mins = (total_secs % 3600) // 60
                        duration = f"{hrs}h {mins}m" if mins > 0 else f"{hrs}h"
                except Exception:
                    pass

            # Get confidence: results → metrics → config eval_score
            confidence = 0.0
            try:
                results = _json.loads(results_raw) if isinstance(results_raw, str) and results_raw else {}
                confidence = results.get("final_accuracy", results.get("eval_score", 0))
            except Exception:
                pass

            if confidence == 0:
                confidence = min(1.0, max(0, metrics_map.get(job_id, 0)))

            if confidence == 0:
                try:
                    config = _json.loads(config_raw) if isinstance(config_raw, str) and config_raw else {}
                    confidence = config.get("eval_score", config.get("quality_score", 0))
                except Exception:
                    pass

            outcome = "promoted" if status == "completed" else "rejected"

            # Extract teacher/student from config + quality_score from results
            teacher = ""
            student = ""
            quality_score = 0.0
            try:
                config = _json.loads(config_raw) if isinstance(config_raw, str) and config_raw else {}
                teacher = config.get("teacher_model", "")
                student = config.get("student_model", "")
            except Exception:
                pass
            try:
                results = _json.loads(results_raw) if isinstance(results_raw, str) and results_raw else {}
                quality_score = results.get("quality_score", 0) or 0
            except Exception:
                pass

            # Build reason string
            reason = ""
            if teacher and student:
                reason = f"{teacher} → {student}"
            elif teacher:
                reason = f"Teacher: {teacher}"
            if not reason:
                reason = "Distillation run" if status == "completed" else f"Status: {status}"

            runs.append({
                "run_id": job_id,
                "name": name,
                "date": created_at,
                "outcome": outcome,
                "confidence": round(confidence, 4),
                "cost": round(cost, 4),
                "duration": duration,
                "reason": reason,
                "teacher_model": teacher,
                "student_model": student,
                "quality_score": round(quality_score, 4),
                "status": status,
            })
    except Exception:
        pass

    # Also merge training_history entries not already covered by distillation jobs
    job_dates = {r["date"][:10] for r in runs if r.get("date")}
    for i, h in enumerate(training_history):
        h_date = str(h.get("date", ""))[:10]
        if h_date and h_date not in job_dates:
            runs.append({
                "run_id": f"run_{i+1:03d}",
                "name": "Training run",
                "date": h.get("date", ""),
                "outcome": "promoted" if h.get("promoted") else "rejected",
                "confidence": 0,
                "cost": 0,
                "duration": "—",
                "reason": h.get("reason", ""),
                "teacher_model": "",
                "student_model": "",
                "quality_score": 0,
                "status": "completed" if h.get("promoted") else "failed",
            })

    runs.sort(key=lambda x: x.get("date", ""), reverse=True)
    return runs


@app.get("/v1/intelligence/training", response_model=TrainingActivityResponse, tags=["intelligence"])
async def intelligence_training(days: int = 30):
    """Training activity: advisor decisions, training history, signal trends."""
    from ..harness.memory_store import get_memory_store

    store = get_memory_store()

    # Query advisor decisions
    decisions_raw = store.query(
        agent="training_advisor",
        category="training_decision",
        limit=50,
    )

    advisor_decisions = []
    signal_trends = []
    latest_rec = {"recommendation": "none", "confidence": 0}

    for entry in decisions_raw:
        ev = getattr(entry, "evaluation", {}) or {}
        advisor_decisions.append({
            "id": getattr(entry, "id", ""),
            "timestamp": getattr(entry, "created_at", ""),
            "recommendation": ev.get("recommendation", "unknown"),
            "confidence": ev.get("confidence", 0),
            "reason": ev.get("reason", ""),
            "source": ev.get("source", "heuristic"),
            "signals": ev.get("signals", []),
        })

        # Extract signal data for trends
        for sig in ev.get("signals", []):
            if sig.get("name") in ("error_rate_increase", "drift_ratio", "high_severity_issues"):
                signal_trends.append({
                    "date": getattr(entry, "created_at", ""),
                    "signal": sig["name"],
                    "value": sig.get("value", 0),
                    "triggered": sig.get("triggered", False),
                })

    if advisor_decisions:
        latest_rec = {
            "recommendation": advisor_decisions[0].get("recommendation", "none"),
            "confidence": advisor_decisions[0].get("confidence", 0),
        }

    # Query training cycle results
    training_cycles_raw = store.query(
        agent="auto_trainer",
        category="run_result",
        limit=20,
    )

    training_history = []
    training_cycles = []
    models_updated = 0

    for entry in training_cycles_raw:
        ev = getattr(entry, "evaluation", {}) or {}
        promoted = ev.get("promoted", False)
        training_history.append({
            "date": getattr(entry, "created_at", ""),
            "promoted": promoted,
            "reason": ev.get("reason", ""),
        })
        training_cycles.append({
            "id": getattr(entry, "id", ""),
            "timestamp": getattr(entry, "created_at", ""),
            "promoted": promoted,
            "reason": ev.get("reason", ""),
            "baseline": ev.get("baseline_metrics", {}),
            "new_metrics": ev.get("new_metrics", {}),
        })
        if promoted:
            models_updated += ev.get("models_updated", 0)

    kpis = {
        "training_runs": len(training_history),
        "last_training": training_history[0]["date"] if training_history else None,
        "advisor_status": latest_rec,
        "models_updated": models_updated,
    }

    # Fetch distillation summary for training cost visibility
    distillation_summary = None
    from ..storage.clickhouse_client import get_client as _get_ch
    try:
        _ch = _get_ch()
    except Exception:
        _ch = None

    if _ch is not None:
        try:
            import json as _json
            r_dist_summary = _ch.query(
                "SELECT count() AS total_jobs, "
                "sum(cost_accrued) AS total_cost, "
                "countIf(status = 'completed') AS completed_jobs, "
                "countIf(status = 'running') AS running_jobs, "
                "countIf(status = 'failed') AS failed_jobs "
                "FROM distillation_jobs FINAL"
            )
            if r_dist_summary.result_rows:
                row = r_dist_summary.result_rows[0]
                total_jobs = int(row[0])
                total_cost = float(row[1])
                completed = int(row[2])
                running = int(row[3])
                failed = int(row[4])

                # Get latest completed job details
                latest_job = None
                r_latest = _ch.query(
                    "SELECT job_id, name, config, cost_accrued, "
                    "toString(completed_at) AS completed_at "
                    "FROM distillation_jobs FINAL "
                    "WHERE status = 'completed' "
                    "ORDER BY completed_at DESC LIMIT 1"
                )
                if r_latest.result_rows:
                    lr = r_latest.result_rows[0]
                    cfg = {}
                    try:
                        cfg = _json.loads(lr[2]) if isinstance(lr[2], str) else lr[2]
                    except Exception:
                        pass
                    latest_job = {
                        "job_id": str(lr[0]),
                        "name": str(lr[1]),
                        "teacher_model": cfg.get("teacher_model", ""),
                        "student_model": cfg.get("student_model", ""),
                        "cost": float(lr[3]),
                        "completed_at": str(lr[4]),
                    }

                distillation_summary = {
                    "total_jobs": total_jobs,
                    "completed_jobs": completed,
                    "running_jobs": running,
                    "failed_jobs": failed,
                    "total_training_cost": round(total_cost, 4),
                    "latest_completed_job": latest_job,
                }
        except Exception:
            pass

    # If training_history from memory store is empty, build from distillation_jobs
    if not training_history and _ch is not None:
        try:
            import json as _json2
            r_hist = _ch.query(
                "SELECT job_id, name, status, config, "
                "toString(created_at) AS created_at "
                "FROM distillation_jobs FINAL "
                "ORDER BY created_at DESC LIMIT 30"
            )
            for row in r_hist.result_rows:
                status = str(row[2])
                promoted = status == "completed"
                name = str(row[1]) or "Distillation run"
                reason = name
                try:
                    cfg = _json2.loads(row[3]) if isinstance(row[3], str) and row[3] else {}
                    teacher = cfg.get("teacher_model", "")
                    student = cfg.get("student_model", "")
                    if teacher and student:
                        reason = f"{teacher} → {student}"
                except Exception:
                    pass
                training_history.append({
                    "date": str(row[4]),
                    "promoted": promoted,
                    "reason": reason,
                })
            kpis["training_runs"] = len(training_history)
            if training_history:
                kpis["last_training"] = training_history[0]["date"]
        except Exception:
            pass

    # If signal_trends is empty, build from llm_traces error/latency data
    if not signal_trends and _ch is not None:
        try:
            r_signals = _ch.query(
                "SELECT toDate(timestamp) AS dt, "
                "count() AS total, "
                "countIf(is_error = 1) AS errors, "
                "avg(latency_ms) AS avg_lat "
                "FROM llm_traces "
                "GROUP BY dt ORDER BY dt"
            )
            for row in r_signals.result_rows:
                dt_str = str(row[0])
                total = int(row[1])
                errors = int(row[2])
                avg_lat = float(row[3])
                error_rate = round(errors / max(total, 1) * 100, 2)
                # Drift ratio: normalized latency deviation from 500ms baseline
                drift = round(min(abs(avg_lat - 500) / 500, 5.0), 2)
                signal_trends.append({
                    "date": dt_str,
                    "signal": "error_rate_increase",
                    "value": error_rate,
                    "triggered": error_rate > 10,
                })
                signal_trends.append({
                    "date": dt_str,
                    "signal": "drift_ratio",
                    "value": drift,
                    "triggered": drift > 2.0,
                })
                signal_trends.append({
                    "date": dt_str,
                    "signal": "high_severity_issues",
                    "value": errors,
                    "triggered": errors > 5,
                })
        except Exception:
            pass

    return TrainingActivityResponse(
        kpis=kpis,
        training_history=training_history,
        signal_trends=signal_trends,
        advisor_decisions=advisor_decisions,
        training_cycles=training_cycles,
        distillation_summary=distillation_summary,
        training_runs_detail=_build_training_runs_detail(_ch, training_history),
    )


@app.get("/v1/intelligence/routing", response_model=RoutingIntelligenceResponse, tags=["intelligence"])
async def intelligence_routing(days: int = 30, limit: int = 50):
    """Real routing intelligence: decisions, win rate, confidence distribution, efficiency trend,
    model usage, daily volume, latency percentiles, error breakdown.

    All data derived from llm_traces — no mock data.
    """
    from datetime import datetime, timedelta, timezone
    import json as _j

    empty = RoutingIntelligenceResponse()

    from ..storage.clickhouse_client import get_client
    try:
        ch = get_client()
    except Exception:
        ch = None

    if ch is None:
        return empty

    start = datetime.now(timezone.utc) - timedelta(days=days)

    def _derive_provider(model: str) -> str:
        m = model.lower()
        if "gpt" in m or "o1" in m or "o3" in m or "o4" in m:
            return "OpenAI"
        if "claude" in m:
            return "Anthropic"
        if "llama" in m or "meta" in m:
            return "Meta"
        if "gemma" in m or "gemini" in m:
            return "Google"
        if "mixtral" in m or "mistral" in m:
            return "Mistral"
        if "deepseek" in m:
            return "DeepSeek"
        if "qwen" in m:
            return "Qwen"
        if model.startswith("opentracy/"):
            return "OpenTracy"
        return "Other"

    # ── 1. Recent routing decisions from llm_traces ──────────────────────
    decisions: list[RoutingDecisionItem] = []
    try:
        r_decisions = ch.query(
            "SELECT request_id, selected_model, provider, total_cost_usd, "
            "latency_ms, is_error, toString(timestamp) AS ts, "
            "all_scores, tokens_in, tokens_out "
            "FROM llm_traces "
            "WHERE timestamp >= {start:DateTime64(3)} "
            "ORDER BY timestamp DESC "
            "LIMIT {limit:UInt32}",
            parameters={"start": start, "limit": limit},
        )
        for row in r_decisions.result_rows:
            req_id = str(row[0])
            model = str(row[1])
            provider = str(row[2]) if row[2] else _derive_provider(model)
            cost = float(row[3])
            latency = float(row[4])
            is_err = int(row[5])
            ts = str(row[6])
            scores_raw = str(row[7]) if row[7] else "{}"
            tok_in = int(row[8]) if row[8] else 0
            tok_out = int(row[9]) if row[9] else 0

            # Derive reason from all_scores
            reason = "Router selection"
            try:
                scores = _j.loads(scores_raw)
                if scores:
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
                    best_model = sorted_scores[0][0] if sorted_scores else ""
                    if best_model == model:
                        reason = "Lowest expected error"
                    elif len(sorted_scores) > 1:
                        reason = f"Cost-optimized (vs {sorted_scores[0][0]})"
            except Exception:
                pass

            decisions.append(RoutingDecisionItem(
                request_id=req_id,
                model_chosen=model,
                provider=provider or _derive_provider(model),
                reason=reason,
                cost=round(cost, 6),
                latency=round(latency, 1),
                tokens_in=tok_in,
                tokens_out=tok_out,
                outcome="error" if is_err else "success",
                timestamp=ts,
            ))
    except Exception:
        pass

    # ── 2. Win rate over time ────────────────────────────────────────────
    win_rate: list[WinRatePoint] = []
    try:
        r_wr = ch.query(
            "SELECT toDate(timestamp) AS day, "
            "  selected_model, "
            "  count() AS total, "
            "  countIf(is_error = 0) AS successes, "
            "  avg(total_cost_usd) AS avg_cost "
            "FROM llm_traces "
            "WHERE timestamp >= {start:DateTime64(3)} "
            "GROUP BY day, selected_model "
            "ORDER BY day",
            parameters={"start": start},
        )
        day_data: dict[str, dict] = {}
        for row in r_wr.result_rows:
            day_str = str(row[0])
            model = str(row[1])
            total = int(row[2])
            successes = int(row[3])
            avg_cost = float(row[4])
            if day_str not in day_data:
                day_data[day_str] = {"total": 0, "successes": 0, "models": {}}
            day_data[day_str]["total"] += total
            day_data[day_str]["successes"] += successes
            day_data[day_str]["models"][model] = {
                "total": total, "successes": successes, "avg_cost": avg_cost
            }
        for day_str in sorted(day_data.keys()):
            d = day_data[day_str]
            router_rate = d["successes"] / d["total"] if d["total"] > 0 else 0
            baseline_rate = 0.0
            max_cost = 0.0
            for _m, stats in d["models"].items():
                if stats["avg_cost"] > max_cost:
                    max_cost = stats["avg_cost"]
                    baseline_rate = stats["successes"] / stats["total"] if stats["total"] > 0 else 0
            win_rate.append(WinRatePoint(
                date=day_str,
                router=round(router_rate, 4),
                baseline=round(baseline_rate, 4),
            ))
    except Exception:
        pass

    # ── 3. Confidence distribution (from cost_adjusted_score) ────────────
    confidence_dist: list[ConfidenceBucket] = []
    try:
        r_conf = ch.query(
            "SELECT "
            "  countIf(cost_adjusted_score >= 0 AND cost_adjusted_score < 0.2) AS b0, "
            "  countIf(cost_adjusted_score >= 0.2 AND cost_adjusted_score < 0.4) AS b1, "
            "  countIf(cost_adjusted_score >= 0.4 AND cost_adjusted_score < 0.6) AS b2, "
            "  countIf(cost_adjusted_score >= 0.6 AND cost_adjusted_score < 0.8) AS b3, "
            "  countIf(cost_adjusted_score >= 0.8 AND cost_adjusted_score <= 1.0) AS b4 "
            "FROM llm_traces "
            "WHERE timestamp >= {start:DateTime64(3)} "
            "AND cost_adjusted_score >= 0",
            parameters={"start": start},
        )
        if r_conf.result_rows:
            row = r_conf.result_rows[0]
            buckets = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
            for i, label in enumerate(buckets):
                confidence_dist.append(ConfidenceBucket(bucket=label, count=int(row[i])))
    except Exception:
        pass

    # ── 4. Efficiency trend (daily savings ratio) ────────────────────────
    efficiency_trend: list[EfficiencyTrendPoint] = []
    try:
        r_max = ch.query(
            "SELECT max(avg_cost) FROM ("
            "  SELECT avg(total_cost_usd) AS avg_cost "
            "  FROM llm_traces WHERE timestamp >= {start:DateTime64(3)} "
            "  GROUP BY selected_model"
            ")",
            parameters={"start": start},
        )
        max_model_cost = float(r_max.result_rows[0][0]) if r_max.result_rows else 0
        if max_model_cost > 0:
            r_eff = ch.query(
                "SELECT toDate(timestamp) AS day, "
                "  avg(total_cost_usd) AS actual_avg, "
                "  count() AS cnt "
                "FROM llm_traces "
                "WHERE timestamp >= {start:DateTime64(3)} "
                "GROUP BY day ORDER BY day",
                parameters={"start": start},
            )
            for row in r_eff.result_rows:
                day_str = str(row[0])
                actual_avg = float(row[1])
                score = 1.0 - (actual_avg / max_model_cost) if max_model_cost > 0 else 0
                efficiency_trend.append(EfficiencyTrendPoint(
                    date=day_str,
                    score=round(max(0, min(1, score)), 4),
                ))
    except Exception:
        pass

    # ── 5. Model usage distribution ──────────────────────────────────────
    model_usage: list[ModelUsageItem] = []
    try:
        r_mu = ch.query(
            "SELECT selected_model, provider, "
            "  count() AS cnt, "
            "  avg(total_cost_usd) AS avg_cost, "
            "  avg(latency_ms) AS avg_lat, "
            "  countIf(is_error = 1) AS err_cnt "
            "FROM llm_traces "
            "WHERE timestamp >= {start:DateTime64(3)} "
            "GROUP BY selected_model, provider "
            "ORDER BY cnt DESC",
            parameters={"start": start},
        )
        total_requests = sum(int(r[2]) for r in r_mu.result_rows) if r_mu.result_rows else 0
        for row in r_mu.result_rows:
            model = str(row[0])
            prov = str(row[1]) if row[1] else _derive_provider(model)
            cnt = int(row[2])
            avg_c = float(row[3])
            avg_l = float(row[4])
            err_c = int(row[5])
            model_usage.append(ModelUsageItem(
                model=model,
                provider=prov or _derive_provider(model),
                count=cnt,
                percentage=round((cnt / total_requests * 100) if total_requests > 0 else 0, 2),
                avg_cost=round(avg_c, 6),
                avg_latency=round(avg_l, 1),
                error_rate=round((err_c / cnt) if cnt > 0 else 0, 4),
            ))
    except Exception:
        pass

    # ── 6. Daily volume with latency + cost aggregates ───────────────────
    daily_volume: list[DailyVolumePoint] = []
    try:
        r_dv = ch.query(
            "SELECT toDate(timestamp) AS day, "
            "  count() AS cnt, "
            "  avg(latency_ms) AS avg_lat, "
            "  quantile(0.95)(latency_ms) AS p95_lat, "
            "  countIf(is_error = 1) AS err_cnt, "
            "  sum(total_cost_usd) AS tot_cost "
            "FROM llm_traces "
            "WHERE timestamp >= {start:DateTime64(3)} "
            "GROUP BY day ORDER BY day",
            parameters={"start": start},
        )
        for row in r_dv.result_rows:
            daily_volume.append(DailyVolumePoint(
                date=str(row[0]),
                count=int(row[1]),
                avg_latency=round(float(row[2]), 1),
                p95_latency=round(float(row[3]), 1),
                error_count=int(row[4]),
                total_cost=round(float(row[5]), 6),
            ))
    except Exception:
        pass

    # ── 7. Latency percentiles per model ─────────────────────────────────
    latency_percentiles: list[LatencyPercentilesItem] = []
    try:
        r_lp = ch.query(
            "SELECT selected_model, "
            "  quantile(0.50)(latency_ms) AS p50, "
            "  quantile(0.75)(latency_ms) AS p75, "
            "  quantile(0.95)(latency_ms) AS p95, "
            "  quantile(0.99)(latency_ms) AS p99 "
            "FROM llm_traces "
            "WHERE timestamp >= {start:DateTime64(3)} "
            "GROUP BY selected_model "
            "ORDER BY p50 ASC",
            parameters={"start": start},
        )
        for row in r_lp.result_rows:
            latency_percentiles.append(LatencyPercentilesItem(
                model=str(row[0]),
                p50=round(float(row[1]), 1),
                p75=round(float(row[2]), 1),
                p95=round(float(row[3]), 1),
                p99=round(float(row[4]), 1),
            ))
    except Exception:
        pass

    # ── 8. Error breakdown by category ───────────────────────────────────
    error_breakdown: list[ErrorBreakdownItem] = []
    try:
        r_eb = ch.query(
            "SELECT if(error_category = '', 'Unknown', error_category) AS cat, "
            "  count() AS cnt "
            "FROM llm_traces "
            "WHERE timestamp >= {start:DateTime64(3)} AND is_error = 1 "
            "GROUP BY cat ORDER BY cnt DESC",
            parameters={"start": start},
        )
        for row in r_eb.result_rows:
            error_breakdown.append(ErrorBreakdownItem(
                category=str(row[0]),
                count=int(row[1]),
            ))
    except Exception:
        pass

    # ── 9. Aggregate KPIs ────────────────────────────────────────────────
    p95_latency = 0.0
    cache_hit_rate = 0.0
    total_tokens = 0
    avg_tokens_per_s = 0.0
    try:
        r_agg = ch.query(
            "SELECT "
            "  quantile(0.95)(latency_ms) AS p95, "
            "  countIf(cache_hit = 1) / count() AS cache_rate, "
            "  sum(total_tokens) AS tot_tokens, "
            "  avg(tokens_per_s) AS avg_tps "
            "FROM llm_traces "
            "WHERE timestamp >= {start:DateTime64(3)}",
            parameters={"start": start},
        )
        if r_agg.result_rows:
            row = r_agg.result_rows[0]
            p95_latency = round(float(row[0]), 1) if row[0] else 0.0
            cache_hit_rate = round(float(row[1]), 4) if row[1] else 0.0
            total_tokens = int(row[2]) if row[2] else 0
            avg_tokens_per_s = round(float(row[3]), 1) if row[3] else 0.0
    except Exception:
        pass

    return RoutingIntelligenceResponse(
        decisions=decisions,
        win_rate=win_rate,
        confidence_distribution=confidence_dist,
        efficiency_trend=efficiency_trend,
        model_usage=model_usage,
        daily_volume=daily_volume,
        latency_percentiles=latency_percentiles,
        error_breakdown=error_breakdown,
        p95_latency=p95_latency,
        cache_hit_rate=cache_hit_rate,
        total_tokens=total_tokens,
        avg_tokens_per_s=avg_tokens_per_s,
    )


@app.get("/v1/intelligence/advisor", response_model=AdvisorConfigResponse, tags=["intelligence"])
async def intelligence_advisor():
    """Training advisor configuration and next training trigger estimate.

    Reads advisor state from memory store and computes next trigger estimate
    based on data accumulation rate.
    """
    from ..harness.memory_store import get_memory_store
    from ..storage.clickhouse_client import get_client
    from datetime import datetime, timedelta, timezone

    store = get_memory_store()

    # Get advisor config from settings or defaults
    threshold = 0.75
    strategy = "Quality-first with cost optimization"
    model_targets: list[str] = []

    try:
        r = get_router()
        profiles = r.registry.get_all()
        model_targets = [p.model_id for p in profiles[:5]]

        # Read threshold from settings if available
        _settings = get_settings()
        if hasattr(_settings, "training_threshold"):
            threshold = _settings.training_threshold
    except Exception:
        pass

    # Read latest advisor decision for strategy info
    try:
        decisions_raw = store.query(
            agent="training_advisor",
            category="training_decision",
            limit=1,
        )
        if decisions_raw:
            ev = getattr(decisions_raw[0], "evaluation", {}) or {}
            if ev.get("source"):
                strategy = f"{ev.get('source', 'heuristic').replace('_', ' ').title()} strategy"
            # Extract threshold from signals if available
            for sig in ev.get("signals", []):
                if sig.get("threshold"):
                    threshold = sig["threshold"]
                    break
    except Exception:
        pass

    # Compute next trigger estimate based on data accumulation
    traces_since_last = 0
    data_rate = 0.0
    next_trigger: str | None = None

    try:
        ch = get_client()
    except Exception:
        ch = None

    if ch is not None:
        try:
            # Find last training date
            last_training_date = None
            training_cycles_raw = store.query(
                agent="auto_trainer",
                category="run_result",
                limit=1,
            )
            if training_cycles_raw:
                last_dt = getattr(training_cycles_raw[0], "created_at", "")
                if last_dt:
                    try:
                        last_training_date = datetime.fromisoformat(
                            str(last_dt).replace("Z", "+00:00")
                        )
                    except Exception:
                        pass

            # Also check distillation jobs
            if last_training_date is None:
                r_last = ch.query(
                    "SELECT max(completed_at) FROM distillation_jobs FINAL "
                    "WHERE status = 'completed'"
                )
                if r_last.result_rows and r_last.result_rows[0][0]:
                    try:
                        last_training_date = datetime.fromisoformat(
                            str(r_last.result_rows[0][0]).replace("Z", "+00:00")
                        )
                    except Exception:
                        pass

            # Count traces since last training
            since = last_training_date or (datetime.now(timezone.utc) - timedelta(days=30))
            r_count = ch.query(
                "SELECT count() FROM llm_traces WHERE timestamp >= {since:DateTime64(3)}",
                parameters={"since": since},
            )
            traces_since_last = int(r_count.result_rows[0][0]) if r_count.result_rows else 0

            # Compute data accumulation rate (traces per day)
            r_rate = ch.query(
                "SELECT count() / greatest(dateDiff('day', min(timestamp), max(timestamp)), 1) "
                "FROM llm_traces WHERE timestamp >= {since:DateTime64(3)}",
                parameters={"since": since},
            )
            data_rate = float(r_rate.result_rows[0][0]) if r_rate.result_rows else 0

            # Estimate: next trigger when we accumulate ~500 more traces (configurable)
            target_traces = 500
            remaining = max(0, target_traces - traces_since_last)
            if data_rate > 0 and remaining > 0:
                days_until = remaining / data_rate
                trigger_date = datetime.now(timezone.utc) + timedelta(days=days_until)
                next_trigger = trigger_date.strftime("%Y-%m-%d")
            elif traces_since_last >= target_traces:
                next_trigger = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            pass

    return AdvisorConfigResponse(
        threshold=threshold,
        strategy=strategy,
        model_targets=model_targets,
        next_trigger_estimate=next_trigger,
        data_accumulation_rate=round(data_rate, 1),
        traces_since_last_training=traces_since_last,
    )
from fastapi import Query, Request as _Req
from opentracy._env import env


def _get_tenant(request: _Req) -> str:
    return request.headers.get("x-tenant-id", "default")


def _get_auth(request: _Req) -> str | None:
    return request.headers.get("authorization") or request.headers.get("x-api-key")


@app.delete("/v1/datasets/{dataset_id}/samples/{sample_id}", tags=["datasets"])
async def delete_sample(dataset_id: str, sample_id: str, request: _Req):
    from ..datasets import repository as ds_repo
    tenant = _get_tenant(request)
    ds_repo.delete_sample(tenant, dataset_id, sample_id)
    return {"success": True}


@app.post("/v1/datasets/from-traces", tags=["datasets"])
async def create_from_traces(request: _Req):
    from ..datasets import repository as ds_repo
    from ..datasets.schemas import CreateFromTracesRequest

    tenant = _get_tenant(request)
    body = await request.json()
    b = CreateFromTracesRequest(**body)

    if b.trace_ids:
        traces = []
        for tid in b.trace_ids:
            t = ds_repo.get_trace(tid)
            if t:
                traces.append(t)
    else:
        traces, _ = ds_repo.list_traces(model_id=b.model_id, limit=b.limit or 50)

    if not traces:
        raise HTTPException(status_code=400, detail="No traces found")

    ds = ds_repo.create_dataset(tenant, name=b.name, description=b.description or "", source="traces")

    samples = [
        {
            "input": t.get("input", ""),
            "expected_output": t.get("output", ""),
            "trace_id": t.get("trace_id", t.get("id", "")),
            "metadata": {
                "model_id": t.get("model_id", ""),
                "provider": t.get("provider", ""),
                "latency_ms": t.get("latency_ms", 0),
                "cost_usd": t.get("cost_usd", 0),
            },
        }
        for t in traces
    ]
    ds_repo.add_samples(tenant, ds["dataset_id"], samples)
    ds = ds_repo.get_dataset(tenant, ds["dataset_id"]) or ds
    return {"dataset": ds}


@app.post("/v1/datasets/from-instruction", tags=["datasets"])
async def create_from_instruction(request: _Req):
    from ..datasets import repository as ds_repo
    from ..datasets.schemas import CreateFromInstructionRequest

    tenant = _get_tenant(request)
    body = await request.json()
    b = CreateFromInstructionRequest(**body)

    traces, _ = ds_repo.list_traces(model_id=b.model_id, limit=b.limit or 200)
    if not traces:
        raise HTTPException(status_code=400, detail="No traces found matching criteria")

    ds = ds_repo.create_dataset(tenant, name=b.name, description=b.description or b.instruction, source="instruction")

    samples = [
        {
            "input": t.get("input", ""),
            "expected_output": t.get("output", ""),
            "trace_id": t.get("trace_id", t.get("id", "")),
            "metadata": {"model_id": t.get("model_id", "")},
        }
        for t in traces[: b.max_samples or 100]
    ]
    ds_repo.add_samples(tenant, ds["dataset_id"], samples)
    ds = ds_repo.get_dataset(tenant, ds["dataset_id"]) or ds
    return {"dataset": ds, "samples_added": len(samples), "traces_analyzed": len(traces)}


@app.post("/v1/datasets/generate", tags=["datasets"])
async def generate_dataset_endpoint():
    raise HTTPException(status_code=410, detail="Synthetic generation has been removed")


@app.get("/v1/datasets/{dataset_id}/auto-collect", tags=["datasets"])
async def get_auto_collect(dataset_id: str, request: _Req):
    from ..datasets import repository as ds_repo
    tenant = _get_tenant(request)
    config = ds_repo.get_auto_collect_config(tenant, dataset_id)
    if not config:
        raise HTTPException(status_code=404, detail="No auto-collect config")
    return config


@app.put("/v1/datasets/{dataset_id}/auto-collect", tags=["datasets"])
async def put_auto_collect(dataset_id: str, request: _Req):
    from ..datasets import repository as ds_repo
    from ..datasets.schemas import AutoCollectConfigIn

    tenant = _get_tenant(request)
    ds = ds_repo.get_dataset(tenant, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    body = await request.json()
    b = AutoCollectConfigIn(**body)
    config = ds_repo.put_auto_collect_config(tenant, dataset_id, b.model_dump())
    return config


@app.delete("/v1/datasets/{dataset_id}/auto-collect", tags=["datasets"])
async def delete_auto_collect(dataset_id: str, request: _Req):
    from ..datasets import repository as ds_repo
    tenant = _get_tenant(request)
    ds_repo.delete_auto_collect_config(tenant, dataset_id)
    return {"success": True}


@app.get("/v1/datasets/{dataset_id}/auto-collect/history", tags=["datasets"])
async def auto_collect_history(dataset_id: str, request: _Req, limit: int = Query(20)):
    from ..datasets import repository as ds_repo
    tenant = _get_tenant(request)
    runs = ds_repo.list_collect_runs(tenant, dataset_id, limit=limit)
    return {"runs": runs}


@app.post("/v1/datasets/{dataset_id}/auto-collect/run", tags=["datasets"])
async def trigger_auto_collect(dataset_id: str, request: _Req):
    from ..datasets import repository as ds_repo
    tenant = _get_tenant(request)
    config = ds_repo.get_auto_collect_config(tenant, dataset_id)
    if not config:
        raise HTTPException(status_code=404, detail="No auto-collect config for this dataset")

    source_model = config.get("source_model", "")
    max_samples = config.get("max_samples", 100)

    collected_ids = ds_repo.get_collected_trace_ids(tenant, dataset_id)
    traces, _ = ds_repo.list_traces(model_id=source_model or None, limit=max_samples * 2)
    new_traces = [t for t in traces if t.get("trace_id") not in collected_ids]
    to_add = new_traces[:max_samples]

    samples = [
        {
            "input": t.get("input", ""),
            "expected_output": t.get("output", ""),
            "trace_id": t.get("trace_id", ""),
            "metadata": {
                "model_id": t.get("model_id", ""),
                "provider": t.get("provider", ""),
                "auto_collected": True,
            },
        }
        for t in to_add
    ]

    added = 0
    if samples:
        added = ds_repo.add_samples(tenant, dataset_id, samples)

    run = ds_repo.save_collect_run(tenant, dataset_id, {
        "traces_scanned": len(traces),
        "traces_new": len(new_traces),
        "samples_added": added,
        "status": "completed",
    })

    existing = ds_repo.get_auto_collect_config(tenant, dataset_id)
    if existing:
        from datetime import datetime, timezone
        existing["last_collected_at"] = datetime.now(timezone.utc)
        existing["total_collected"] = (existing.get("total_collected", 0) or 0) + added
        ds_repo.put_auto_collect_config(tenant, dataset_id, existing)

    return {"run_id": run.get("run_id", ""), "samples_added": added}


@app.get("/v1/evaluations/settings", tags=["evaluations"])
async def get_eval_settings(request: _Req):
    from ..settings import repository as settings_repo
    tenant = _get_tenant(request)
    return settings_repo.get(tenant)


@app.put("/v1/evaluations/settings", tags=["evaluations"])
async def put_eval_settings(request: _Req):
    from ..settings import repository as settings_repo
    tenant = _get_tenant(request)
    body = await request.json()

    allowed_fields = ["default_judge_model", "default_temperature", "max_parallel_requests", "python_script_timeout", "config"]
    updates = {}
    for field in allowed_fields:
        if field not in body:
            continue
        value = body[field]
        if field == "default_temperature":
            if not isinstance(value, (int, float)) or value < 0 or value > 2:
                raise HTTPException(status_code=400, detail="default_temperature must be between 0 and 2")
        if field == "max_parallel_requests":
            if not isinstance(value, int) or value < 1 or value > 20:
                raise HTTPException(status_code=400, detail="max_parallel_requests must be between 1 and 20")
        if field == "python_script_timeout":
            if not isinstance(value, int) or value < 1 or value > 300:
                raise HTTPException(status_code=400, detail="python_script_timeout must be between 1 and 300 seconds")
        updates[field] = value

    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    return settings_repo.update(tenant, updates)


@app.get("/v1/evaluations/{evaluation_id}/export", tags=["evaluations"])
async def export_eval_results(evaluation_id: str, request: _Req, format: str = Query("json")):
    import csv as _csv
    from io import StringIO
    from typing import Any as _Any
    from ..evaluations import repository as eval_repo

    tenant = _get_tenant(request)
    evaluation = eval_repo.get(tenant, evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    result = eval_repo.get_result(tenant, evaluation_id)
    if not result:
        raise HTTPException(status_code=400, detail="No results to export")

    if format == "json":
        return {"evaluation": evaluation, "results": result}

    samples = result.get("samples", [])
    if not samples:
        return {"csv": "", "message": "No samples"}

    models = evaluation.get("models", [])
    metrics = evaluation.get("metrics", [])

    columns = ["sample_id", "input"]
    for model in models:
        columns.extend([f"{model}_output", f"{model}_latency", f"{model}_cost"])
    for metric in metrics:
        mid = metric.get("metric_id", metric) if isinstance(metric, dict) else metric
        for model in models:
            columns.append(f"{model}_{mid}")

    output = StringIO()
    writer = _csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()

    for sample in samples:
        row: dict[str, _Any] = {
            "sample_id": sample.get("sample_id", ""),
            "input": sample.get("input", ""),
        }
        outputs = sample.get("outputs", {})
        for model in models:
            mo = outputs.get(model, {})
            row[f"{model}_output"] = mo.get("output", "")
            row[f"{model}_latency"] = mo.get("latency", "")
            row[f"{model}_cost"] = mo.get("cost", "")
        scores = sample.get("scores", {})
        for metric in metrics:
            mid = metric.get("metric_id", metric) if isinstance(metric, dict) else metric
            ms = scores.get(mid, {})
            for model in models:
                val = ms.get(model, "")
                if isinstance(val, dict):
                    val = val.get("score", "")
                row[f"{model}_{mid}"] = val
        writer.writerow(row)

    return {"csv": output.getvalue(), "filename": f"evaluation_{evaluation_id}.csv"}


@app.post("/v1/evaluations/{evaluation_id}/rerun", tags=["evaluations"])
async def rerun_evaluation(evaluation_id: str, request: _Req):
    from ..evaluations import repository as eval_repo
    from ..evaluations.runner import EvaluationRunner

    tenant = _get_tenant(request)
    evaluation = eval_repo.get(tenant, evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    st = evaluation.get("status")
    if st not in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Cannot rerun evaluation with status: {st}")

    new_data = {
        "name": f"{evaluation.get('name')} (rerun)",
        "description": evaluation.get("description", ""),
        "dataset_id": evaluation.get("dataset_id"),
        "models": evaluation.get("models"),
        "metrics": evaluation.get("metrics"),
        "config": evaluation.get("config", {}),
        "total_samples": evaluation.get("total_samples", 0),
        "original_evaluation_id": evaluation_id,
    }
    new_eval = eval_repo.create(tenant, new_data)

    try:
        authorization = _get_auth(request)
        runner = EvaluationRunner()
        result = runner.run(tenant, new_eval["evaluation_id"], authorization=authorization)
        if result.get("success"):
            new_eval = eval_repo.get(tenant, new_eval["evaluation_id"])
    except Exception as e:
        logger.exception("Error rerunning evaluation: %s", e)
        eval_repo.update_status(tenant, new_eval["evaluation_id"], status="failed", error_message=str(e))
        new_eval = eval_repo.get(tenant, new_eval["evaluation_id"])

    return {"message": "Evaluation rerun started", "evaluation": new_eval}


@app.post("/v1/evaluations/log", tags=["evaluations"])
async def log_evaluation(request: _Req):
    from ..evaluations import repository as eval_repo

    tenant = _get_tenant(request)
    body = await request.json()

    required = ["id", "name", "status"]
    missing = [f for f in required if f not in body]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

    metrics = body.get("metrics", body.get("scorer_names", []))

    evaluation_data = {
        "name": body["name"],
        "description": body.get("description", "SDK local execution"),
        "status": body["status"],
        "execution_type": "local",
        "metrics": metrics,
        "models": body.get("models", []),
        "config": body.get("config", {}),
        "progress": body.get("progress", {}).get("completed", 0),
        "total_samples": body.get("progress", {}).get("total", 0),
        "created_at": body.get("created_at"),
    }

    sdk_eval_id = body["id"]
    evaluation = eval_repo.create_with_id(tenant, sdk_eval_id, evaluation_data)

    results = body.get("results", {})
    if results:
        eval_repo.save_result(tenant, evaluation["evaluation_id"], results)
    else:
        rows = body.get("rows", [])
        if rows:
            result_data = {
                "samples": [
                    {
                        "sample_id": row.get("datapoint_id", ""),
                        "input": row.get("input", ""),
                        "output": row.get("output", ""),
                        "expected": row.get("expected", ""),
                        "scores": {
                            score.get("name"): {
                                "score": score.get("score", 0),
                                "raw_value": score.get("raw_value"),
                                "explanation": score.get("explanation"),
                            }
                            for score in row.get("scores", [])
                        },
                    }
                    for row in rows
                ],
                "summary": body.get("summary", {}),
            }
            eval_repo.save_result(tenant, evaluation["evaluation_id"], result_data)

    return {
        "message": "Evaluation logged successfully",
        "evaluation_id": evaluation["evaluation_id"],
        "sdk_evaluation_id": sdk_eval_id,
    }


@app.post("/v1/evaluations/cleanup", tags=["evaluations"])
async def cleanup_stale_evaluations(request: _Req):
    from ..evaluations import repository as eval_repo

    tenant = _get_tenant(request)
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    threshold = float(body.get("stale_threshold_hours", 1.0))

    stale = eval_repo.find_stale_evaluations(tenant, threshold)
    if not stale:
        return {"message": "No stale evaluations found", "cleaned_up": 0, "evaluations": []}

    marked = eval_repo.mark_stale_as_failed(tenant, threshold)
    return {
        "message": f"Cleaned up {len(marked)} stale evaluation(s)",
        "cleaned_up": len(marked),
        "evaluations": [
            {
                "evaluation_id": e.get("evaluation_id"),
                "name": e.get("name"),
                "error_message": e.get("error_message"),
            }
            for e in marked
        ],
    }


@app.put("/v1/metrics/{metric_id}", tags=["metrics"])
async def update_metric(metric_id: str, request: _Req):
    from ..metrics import repository as metrics_repo

    tenant = _get_tenant(request)
    existing = metrics_repo.get(tenant, metric_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Metric not found")
    if existing.get("is_builtin"):
        raise HTTPException(status_code=403, detail="Cannot modify builtin metrics")

    body = await request.json()
    allowed = ["name", "description", "config", "python_script", "requirements"]
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    metric = metrics_repo.update(tenant, metric_id, updates)
    return metric


@app.post("/v1/metrics/validate-script", tags=["metrics"])
async def validate_metric_script(request: _Req):
    body = await request.json()
    if "python_script" not in body:
        raise HTTPException(status_code=400, detail="python_script is required")

    script = body["python_script"]
    errors = []
    warnings = []

    if "def evaluate(" not in script:
        errors.append("Script must define an 'evaluate' function")

    dangerous = ["os", "subprocess", "sys", "shutil", "socket"]
    for imp in dangerous:
        if f"import {imp}" in script or f"from {imp}" in script:
            errors.append(f"Import of '{imp}' is not allowed")

    try:
        compile(script, "<string>", "exec")
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")

    test_result = None
    if not errors and all(k in body for k in ["test_input", "test_output"]):
        try:
            sandbox = {"__builtins__": {
                "len": len, "str": str, "int": int, "float": float,
                "bool": bool, "list": list, "dict": dict, "min": min,
                "max": max, "sum": sum, "abs": abs, "round": round,
                "range": range, "enumerate": enumerate, "zip": zip,
                "True": True, "False": False, "None": None,
            }}
            exec(script, sandbox)
            if "evaluate" in sandbox:
                result = sandbox["evaluate"](
                    output=body["test_output"],
                    expected=body.get("test_expected", ""),
                    input_text=body["test_input"],
                )
                test_result = {"success": True, "result": result}
        except Exception as e:
            test_result = {"success": False, "error": str(e)}

    response = {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    if test_result:
        response["test_result"] = test_result
    return response

@app.get("/v1/experiments", tags=["experiments"])
async def list_experiments(request: _Req, status: str | None = Query(None)):
    from ..experiments import repository as exp_repo
    tenant = _get_tenant(request)
    experiments = exp_repo.list_all(tenant, status=status)
    return {"experiments": experiments, "count": len(experiments)}


@app.post("/v1/experiments", tags=["experiments"], status_code=201)
async def create_experiment(request: _Req):
    from ..experiments import repository as exp_repo
    from ..evaluations import repository as eval_repo

    tenant = _get_tenant(request)
    body = await request.json()

    required = ["name", "dataset_id", "evaluation_ids"]
    missing = [f for f in required if f not in body]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

    evaluation_ids = body["evaluation_ids"]
    if not isinstance(evaluation_ids, list) or len(evaluation_ids) < 2:
        raise HTTPException(status_code=400, detail="evaluation_ids must be a list with at least 2 evaluations")

    for eid in evaluation_ids:
        if not eval_repo.get(tenant, eid):
            raise HTTPException(status_code=404, detail=f"Evaluation not found: {eid}")

    experiment = exp_repo.create(tenant, {
        "name": body["name"],
        "description": body.get("description", ""),
        "dataset_id": body["dataset_id"],
        "evaluation_ids": evaluation_ids,
        "tags": body.get("tags", []),
    })
    return experiment


@app.get("/v1/experiments/{experiment_id}", tags=["experiments"])
async def get_experiment(experiment_id: str, request: _Req):
    from ..experiments import repository as exp_repo
    tenant = _get_tenant(request)
    experiment = exp_repo.get(tenant, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@app.delete("/v1/experiments/{experiment_id}", tags=["experiments"])
async def delete_experiment(experiment_id: str, request: _Req):
    from ..experiments import repository as exp_repo
    tenant = _get_tenant(request)
    experiment = exp_repo.get(tenant, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.get("status") == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running experiment")
    exp_repo.delete(tenant, experiment_id)
    return {"message": "Experiment deleted"}


@app.get("/v1/experiments/{experiment_id}/comparison", tags=["experiments"])
async def get_experiment_comparison(experiment_id: str, request: _Req, force: str | None = Query(None)):
    from ..experiments import repository as exp_repo
    from ..experiments.runner import ExperimentRunner

    tenant = _get_tenant(request)
    experiment = exp_repo.get(tenant, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if force != "true":
        comparison = exp_repo.get_comparison(tenant, experiment_id)
        if comparison:
            return comparison

    if experiment.get("status") in ("draft", "completed"):
        try:
            runner = ExperimentRunner()
            return runner.build_comparison(tenant, experiment_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            logger.exception("Failed to build comparison")
            raise HTTPException(status_code=500, detail="Failed to build comparison")

    if experiment.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Experiment is not completed. Status: {experiment.get('status')}")
    raise HTTPException(status_code=404, detail="Comparison not found")


@app.get("/v1/annotations/queues", tags=["annotations"])
async def list_annotation_queues(request: _Req):
    from ..annotations import repository as ann_repo
    tenant = _get_tenant(request)
    queues = ann_repo.list_queues(tenant)
    return {"queues": queues, "count": len(queues)}


@app.post("/v1/annotations/queues", tags=["annotations"], status_code=201)
async def create_annotation_queue(request: _Req):
    from ..annotations import repository as ann_repo
    from ..datasets.repository import get_samples

    tenant = _get_tenant(request)
    body = await request.json()
    if "name" not in body:
        raise HTTPException(status_code=400, detail="name is required")
    if "dataset_id" not in body:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    queue = ann_repo.create_queue(tenant, {
        "name": body["name"],
        "dataset_id": body["dataset_id"],
        "rubric": body.get("rubric", []),
    })

    samples = get_samples(tenant, body["dataset_id"])
    if samples:
        ann_repo.create_items_from_samples(tenant, queue["queue_id"], samples)
        queue["total_items"] = len(samples)

    return queue


@app.delete("/v1/annotations/queues/{queue_id}", tags=["annotations"])
async def delete_annotation_queue(queue_id: str, request: _Req):
    from ..annotations import repository as ann_repo
    tenant = _get_tenant(request)
    if not ann_repo.get_queue(tenant, queue_id):
        raise HTTPException(status_code=404, detail="Queue not found")
    ann_repo.delete_queue(tenant, queue_id)
    return {"message": "Queue deleted"}


@app.get("/v1/annotations/queues/{queue_id}/items", tags=["annotations"])
async def list_annotation_items(queue_id: str, request: _Req, status: str | None = Query(None)):
    from ..annotations import repository as ann_repo
    tenant = _get_tenant(request)
    items = ann_repo.list_items(tenant, queue_id, status=status)
    return {"items": items, "count": len(items)}


@app.get("/v1/annotations/queues/{queue_id}/next", tags=["annotations"])
async def get_next_annotation_item(queue_id: str, request: _Req):
    from ..annotations import repository as ann_repo
    tenant = _get_tenant(request)
    item = ann_repo.get_next_pending(tenant, queue_id)
    if not item:
        raise HTTPException(status_code=404, detail="No pending items")
    return item


@app.post("/v1/annotations/queues/{queue_id}/items/{item_id}/submit", tags=["annotations"])
async def submit_annotation_item(queue_id: str, item_id: str, request: _Req):
    from ..annotations import repository as ann_repo
    tenant = _get_tenant(request)
    body = await request.json()
    if "scores" not in body:
        raise HTTPException(status_code=400, detail="scores is required")
    result = ann_repo.submit_item(tenant, queue_id, item_id, scores=body["scores"], notes=body.get("notes", ""))
    if not result:
        raise HTTPException(status_code=404, detail="Pending item not found")
    return result


@app.post("/v1/annotations/queues/{queue_id}/items/{item_id}/skip", tags=["annotations"])
async def skip_annotation_item(queue_id: str, item_id: str, request: _Req):
    from ..annotations import repository as ann_repo
    tenant = _get_tenant(request)
    result = ann_repo.skip_item(tenant, queue_id, item_id)
    if not result:
        raise HTTPException(status_code=404, detail="Pending item not found")
    return result


@app.get("/v1/annotations/queues/{queue_id}/stats", tags=["annotations"])
async def get_annotation_stats(queue_id: str, request: _Req):
    from ..annotations import repository as ann_repo
    tenant = _get_tenant(request)
    stats = ann_repo.get_stats(tenant, queue_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Queue not found")
    return stats


@app.get("/v1/annotations/queues/{queue_id}/export", tags=["annotations"])
async def export_annotations(queue_id: str, request: _Req, format: str = Query("json")):
    import csv as _csv
    from io import StringIO
    from typing import Any as _Any
    from ..annotations import repository as ann_repo

    tenant = _get_tenant(request)
    if format not in ("json", "csv"):
        raise HTTPException(status_code=400, detail="format must be 'json' or 'csv'")

    queue = ann_repo.get_queue(tenant, queue_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Queue not found")

    items = ann_repo.get_completed_items(tenant, queue_id)

    if format == "json":
        return {"queue": queue, "annotations": items, "count": len(items)}

    rubric = queue.get("rubric", [])
    criteria_names = [c["name"] for c in rubric]

    if not items:
        return {"csv": "", "filename": f"annotations_{queue_id}.csv"}

    output = StringIO()
    columns = ["sample_id", "input", "expected_output"] + criteria_names + ["notes", "annotated_at"]
    writer = _csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()

    for item in items:
        scores = item.get("scores", {})
        row: dict[str, _Any] = {
            "sample_id": item.get("sample_id", ""),
            "input": item.get("input", ""),
            "expected_output": item.get("expected_output", ""),
            "notes": item.get("notes", ""),
            "annotated_at": item.get("annotated_at", ""),
        }
        for crit in criteria_names:
            row[crit] = scores.get(crit, "")
        writer.writerow(row)

    return {"csv": output.getvalue(), "filename": f"annotations_{queue_id}.csv"}


@app.get("/v1/annotations/queues/{queue_id}/analytics", tags=["annotations"])
async def get_annotation_analytics(queue_id: str, request: _Req):
    import math
    from collections import Counter
    from typing import Any as _Any
    from ..annotations import repository as ann_repo

    tenant = _get_tenant(request)
    queue = ann_repo.get_queue(tenant, queue_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Queue not found")

    items = ann_repo.get_completed_items(tenant, queue_id)
    rubric = queue.get("rubric", [])

    criteria_stats: dict[str, _Any] = {}
    for criterion in rubric:
        crit_name = criterion["name"]
        scale_min = criterion.get("scale_min", 1)
        scale_max = criterion.get("scale_max", 5)

        values = [item["scores"][crit_name] for item in items if crit_name in item.get("scores", {})]
        if not values:
            criteria_stats[crit_name] = {"mean": None, "median": None, "std_dev": None, "min": None, "max": None, "distribution": {}}
            continue

        n = len(values)
        mean = sum(values) / n
        sorted_vals = sorted(values)
        median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        variance = sum((v - mean) ** 2 for v in values) / n
        std_dev = math.sqrt(variance)
        distribution = {str(s): values.count(s) for s in range(scale_min, scale_max + 1)}

        criteria_stats[crit_name] = {
            "mean": round(mean, 2),
            "median": round(median, 2),
            "std_dev": round(std_dev, 2),
            "min": min(values),
            "max": max(values),
            "distribution": distribution,
        }

    agreement = None
    dataset_id = queue.get("dataset_id")
    if dataset_id:
        all_queue_items = ann_repo.get_completed_items_by_dataset(tenant, dataset_id)
        other_queue_ids = [qid for qid in all_queue_items if qid != queue_id]

        if other_queue_ids:
            current_by_sample = {item["sample_id"]: item.get("scores", {}) for item in items if item.get("sample_id")}
            other_by_sample: dict[str, dict[str, _Any]] = {}
            for qid in other_queue_ids:
                for item in all_queue_items[qid]:
                    sid = item.get("sample_id")
                    if sid and sid not in other_by_sample:
                        other_by_sample[sid] = item.get("scores", {})

            overlapping = set(current_by_sample.keys()) & set(other_by_sample.keys())
            if overlapping:
                cohens_kappa: dict[str, float] = {}
                percent_agreement: dict[str, float] = {}

                for criterion in rubric:
                    crit_name = criterion["name"]
                    scale_min = criterion.get("scale_min", 1)
                    scale_max = criterion.get("scale_max", 5)
                    ratings_a, ratings_b = [], []

                    for sid in overlapping:
                        a = current_by_sample[sid].get(crit_name)
                        b = other_by_sample[sid].get(crit_name)
                        if a is not None and b is not None:
                            ratings_a.append(a)
                            ratings_b.append(b)

                    if ratings_a:
                        # Cohen's kappa
                        n_items = len(ratings_a)
                        categories = list(range(scale_min, scale_max + 1))
                        agree = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
                        p_observed = agree / n_items
                        count_a = Counter(ratings_a)
                        count_b = Counter(ratings_b)
                        p_expected = sum((count_a[c] / n_items) * (count_b[c] / n_items) for c in categories)
                        if p_expected == 1.0:
                            kappa = 1.0
                        else:
                            kappa = (p_observed - p_expected) / (1 - p_expected)
                        cohens_kappa[crit_name] = round(kappa, 2)
                        pct = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / len(ratings_a)
                        percent_agreement[crit_name] = round(pct, 2)

                agreement = {
                    "compared_queues": other_queue_ids,
                    "overlapping_samples": len(overlapping),
                    "cohens_kappa": cohens_kappa,
                    "percent_agreement": percent_agreement,
                }

    return {
        "queue_id": queue_id,
        "total_annotated": len(items),
        "criteria": criteria_stats,
        "agreement": agreement,
    }


# ===================================================================
# Auto-eval — config/run endpoints (from auto_eval router)
# ===================================================================

@app.get("/v1/auto-eval/configs", tags=["auto-eval"])
async def list_auto_eval_configs(request: _Req):
    from ..auto_eval import repository as ae_repo
    tenant = _get_tenant(request)
    configs = ae_repo.list_configs(tenant)
    return {"configs": configs, "count": len(configs)}


@app.post("/v1/auto-eval/configs", tags=["auto-eval"], status_code=201)
async def create_auto_eval_config(request: _Req):
    from ..auto_eval import repository as ae_repo
    tenant = _get_tenant(request)
    body = await request.json()

    required = ["name", "dataset_id", "models", "metrics"]
    missing = [f for f in required if f not in body]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

    if not isinstance(body["models"], list) or len(body["models"]) == 0:
        raise HTTPException(status_code=400, detail="models must be a non-empty array")
    if not isinstance(body["metrics"], list) or len(body["metrics"]) == 0:
        raise HTTPException(status_code=400, detail="metrics must be a non-empty array")

    config = ae_repo.create_config(tenant, {
        "name": body["name"],
        "dataset_id": body["dataset_id"],
        "dataset_name": body.get("dataset_name", ""),
        "models": body["models"],
        "metrics": body["metrics"],
        "schedule": body.get("schedule", "daily"),
        "alert_on_regression": body.get("alert_on_regression", True),
        "regression_threshold": body.get("regression_threshold", 0.05),
        "topic_filter": body.get("topic_filter"),
    })
    return config


@app.put("/v1/auto-eval/configs/{config_id}", tags=["auto-eval"])
async def update_auto_eval_config(config_id: str, request: _Req):
    from ..auto_eval import repository as ae_repo
    tenant = _get_tenant(request)
    if not ae_repo.get_config(tenant, config_id):
        raise HTTPException(status_code=404, detail="Config not found")

    body = await request.json()
    allowed = {"name", "dataset_id", "dataset_name", "models", "metrics",
               "schedule", "alert_on_regression", "regression_threshold",
               "topic_filter", "enabled"}
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    updated = ae_repo.update_config(tenant, config_id, updates)
    return updated


@app.delete("/v1/auto-eval/configs/{config_id}", tags=["auto-eval"])
async def delete_auto_eval_config(config_id: str, request: _Req):
    from ..auto_eval import repository as ae_repo
    tenant = _get_tenant(request)
    if not ae_repo.get_config(tenant, config_id):
        raise HTTPException(status_code=404, detail="Config not found")
    ae_repo.delete_config(tenant, config_id)
    return {"message": "Config deleted"}


@app.post("/v1/auto-eval/configs/{config_id}/trigger", tags=["auto-eval"], status_code=202)
async def trigger_auto_eval_run(config_id: str, request: _Req):
    from datetime import datetime
    from ..auto_eval import repository as ae_repo
    from ..datasets.repository import get_samples
    from ..evaluations import repository as eval_repo
    from ..evaluations.runner import EvaluationRunner

    tenant = _get_tenant(request)
    config = ae_repo.get_config(tenant, config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")

    samples = get_samples(tenant, config["dataset_id"])

    now = datetime.utcnow().isoformat()
    evaluation_data = {
        "name": f"[Auto] {config['name']} - {now[:16]}",
        "description": f"Auto-eval run for config {config_id}",
        "dataset_id": config["dataset_id"],
        "models": config["models"],
        "metrics": config["metrics"],
        "config": {},
        "total_samples": len(samples),
        "auto_eval_config_id": config_id,
    }

    evaluation = eval_repo.create(tenant, evaluation_data)
    eid = evaluation["evaluation_id"]

    auth = _get_auth(request)
    try:
        runner = EvaluationRunner()
        runner.run(tenant, eid, authorization=auth)
    except Exception as e:
        logger.exception("Error running auto-eval evaluation: %s", e)
        eval_repo.update_status(tenant, eid, status="failed", error_message=str(e))

    run = ae_repo.create_run(tenant, config_id, {"evaluation_id": eid})
    return run


@app.get("/v1/auto-eval/configs/{config_id}/runs", tags=["auto-eval"])
async def list_auto_eval_runs(config_id: str, request: _Req):
    from datetime import datetime
    from ..auto_eval import repository as ae_repo
    from ..evaluations import repository as eval_repo

    tenant = _get_tenant(request)
    config = ae_repo.get_config(tenant, config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")

    runs = ae_repo.list_runs(tenant, config_id)

    for run in runs:
        if run.get("status") != "running":
            continue
        eid = run.get("evaluation_id")
        if not eid:
            continue
        try:
            evaluation = eval_repo.get(tenant, eid)
            if not evaluation:
                continue
            es = evaluation.get("status")
            if es not in ("completed", "failed"):
                continue

            run_updates: dict = {"status": es}

            if es == "completed":
                run_updates["completed_at"] = evaluation.get("completed_at", datetime.utcnow().isoformat())
                result = eval_repo.get_result(tenant, eid)
                if result:
                    summary = result.get("summary", {})
                    scores = {}
                    for metric_id, metric_data in summary.items():
                        if isinstance(metric_data, dict) and "average" in metric_data:
                            scores[metric_id] = metric_data["average"]
                        elif isinstance(metric_data, (int, float)):
                            scores[metric_id] = metric_data
                    run_updates["scores"] = scores

                    threshold = config.get("regression_threshold", 0.05)
                    last_score = config.get("last_run_score")
                    regression = False
                    if scores and config.get("alert_on_regression"):
                        avg_score = sum(scores.values()) / len(scores)
                        if last_score is not None and last_score > 0:
                            regression = (last_score - avg_score) > threshold
                    run_updates["regression_detected"] = regression

                    avg_score = sum(scores.values()) / len(scores) if scores else 0
                    ae_repo.update_config_last_run(tenant, config_id, avg_score, run_updates["completed_at"])
            elif es == "failed":
                run_updates["completed_at"] = evaluation.get("completed_at", datetime.utcnow().isoformat())
                run_updates["error_message"] = evaluation.get("error_message", "Evaluation failed")

            ae_repo.update_run(tenant, config_id, run["run_id"], run_updates)
            run.update(run_updates)
        except Exception as e:
            logger.warning("Failed to hydrate run %s: %s", run.get("run_id"), e)

    return {"runs": runs, "count": len(runs)}


# ===================================================================
# Proposals — all endpoints (from proposals router)
# ===================================================================

@app.get("/v1/proposals", tags=["proposals"])
async def list_proposals(request: _Req, status: str | None = Query(None)):
    from ..proposals import repository as prop_repo
    tenant = _get_tenant(request)
    proposals = prop_repo.list_all(tenant, status=status)
    return {"proposals": proposals, "count": len(proposals)}


@app.get("/v1/proposals/{proposal_id}", tags=["proposals"])
async def get_proposal(proposal_id: str, request: _Req):
    from ..proposals import repository as prop_repo
    tenant = _get_tenant(request)
    proposal = prop_repo.get(tenant, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


@app.post("/v1/proposals/{proposal_id}/approve", tags=["proposals"])
async def approve_proposal(proposal_id: str, request: _Req):
    from ..proposals import repository as prop_repo
    from ..proposals.decision_engine import DecisionEngine

    tenant = _get_tenant(request)
    proposal = prop_repo.get(tenant, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    if proposal.get("status") != "pending":
        raise HTTPException(status_code=400, detail=f"Cannot approve proposal with status: {proposal.get('status')}")

    prop_repo.update_status(tenant, proposal_id, "approved")
    auth = _get_auth(request)

    try:
        engine = DecisionEngine()
        result = engine.execute_proposal(tenant, proposal, authorization=auth)
        if result.get("success"):
            prop_repo.update_status(tenant, proposal_id, "executed", execution_result=result)
            return {"message": "Proposal approved and executed", "proposal_id": proposal_id, "execution_result": result}
        else:
            prop_repo.update_status(tenant, proposal_id, "failed", execution_result=result)
            raise HTTPException(status_code=500, detail={"error": "Proposal execution failed", "execution_result": result})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Proposal execution error")
        error_result = {"success": False, "error": str(e)}
        prop_repo.update_status(tenant, proposal_id, "failed", execution_result=error_result)
        raise HTTPException(status_code=500, detail=f"Proposal execution failed: {e}")


@app.post("/v1/proposals/{proposal_id}/reject", tags=["proposals"])
async def reject_proposal(proposal_id: str, request: _Req):
    from ..proposals import repository as prop_repo

    tenant = _get_tenant(request)
    proposal = prop_repo.get(tenant, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    if proposal.get("status") != "pending":
        raise HTTPException(status_code=400, detail=f"Cannot reject proposal with status: {proposal.get('status')}")

    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    reason = body.get("reason")
    execution_result = {"reject_reason": reason} if reason else None

    prop_repo.update_status(tenant, proposal_id, "rejected", execution_result=execution_result)
    return {"message": "Proposal rejected", "proposal_id": proposal_id}


# ===================================================================
# Eval Agent — all endpoints (from eval_agent router)
# ===================================================================

@app.post("/v1/eval-agent/analyze", tags=["eval-agent"])
async def eval_agent_analyze(request: _Req):
    from ..eval_agent.agent import EvalAgent

    tenant = _get_tenant(request)
    body = await request.json()
    dataset_id = body.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    auth = _get_auth(request)
    try:
        agent = EvalAgent()
        result = agent.analyze(tenant, dataset_id, authorization=auth)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Eval agent analyze failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/v1/eval-agent/setup", tags=["eval-agent"], status_code=201)
async def eval_agent_setup(request: _Req):
    from ..eval_agent.agent import EvalAgent

    tenant = _get_tenant(request)
    body = await request.json()
    dataset_id = body.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    auto_trigger = body.get("auto_trigger", True)
    auth = _get_auth(request)

    try:
        agent = EvalAgent()
        result = agent.setup(tenant, dataset_id, authorization=auth, auto_trigger=auto_trigger)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Eval agent setup failed")
        raise HTTPException(status_code=500, detail=f"Setup failed: {e}")


@app.post("/v1/eval-agent/scan", tags=["eval-agent"])
async def eval_agent_scan(request: _Req):
    from ..eval_agent.agent import EvalAgent

    tenant = _get_tenant(request)
    auth = _get_auth(request)
    try:
        agent = EvalAgent()
        result = agent.scan_all(tenant, authorization=auth)
        return result
    except Exception as e:
        logger.exception("Eval agent scan failed")
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")


# ===================================================================
# Distillation — all endpoints (from distillation router)
# ===================================================================

@app.get("/v1/distillation", tags=["distillation"])
async def list_distillation_jobs(
    tenant_id: str = Query("default"),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    from ..distillation import repository as dist_repo
    from ..distillation.serialization import serialize_job as _serialize_job

    tid = tenant_id or "default"
    jobs, total = dist_repo.list_jobs(tid, status=status, limit=limit, offset=offset)
    return [_serialize_job(j) for j in jobs if j.get("status") != "deleted"]


@app.post("/v1/distillation", tags=["distillation"])
async def create_distillation_job(body: dict):
    from ..distillation import repository as dist_repo
    from ..distillation.schemas import DistillationConfig
    from ..distillation.serialization import serialize_job as _serialize_job

    tid = body.get("tenant_id") or "default"
    name = body.get("name", "Untitled")
    description = body.get("description", "")
    config_data = body.get("config", {})

    try:
        config = DistillationConfig(**config_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}")

    job = dist_repo.create_job(tid, {
        "name": name,
        "description": description,
        "config": config.model_dump(),
    })

    job_id = job["job_id"]
    logger.info("Created distillation job %s for tenant %s", job_id, tid)

    from ..distillation.pipeline import launch_pipeline
    launch_pipeline(job_id, tid, config)

    return _serialize_job(job)


@app.post("/v1/distillation/estimate", tags=["distillation"])
async def estimate_distillation_job(request: _Req):
    from ..distillation.schemas import EstimateRequest, MODEL_PARAM_SIZES

    body = await request.json()
    b = EstimateRequest(**body)

    param_size = MODEL_PARAM_SIZES.get(b.student_model, 1.0)
    gen_cost = b.num_prompts * b.n_samples * 0.001
    train_cost = param_size * 0.01
    estimated_cost = gen_cost + train_cost

    return {
        "estimated_cost": round(estimated_cost, 2),
        "is_sandbox": False,
        "tier": "local",
        "balance": 999999,
        "sufficient": True,
    }


@app.get("/v1/distillation/teacher-models", tags=["distillation"])
async def list_teacher_models():
    from ..distillation.schemas import TEACHER_MODEL_MAP

    models = []
    for m in TEACHER_MODEL_MAP:
        models.append({
            "id": m["id"],
            "name": m["name"],
            "provider": m["provider"],
            "type": "external",
            "available": True,
        })
    return {"models": models}


@app.get("/v1/distillation/student-models", tags=["distillation"])
async def list_student_models():
    from ..distillation.schemas import STUDENT_MODEL_MAP, MODEL_PARAM_SIZES

    models = []
    for key in STUDENT_MODEL_MAP:
        family = key.split("-")[0]
        params = MODEL_PARAM_SIZES.get(key, 0)
        models.append({
            "id": key,
            "name": key,
            "family": family,
            "params": f"{params}B",
            "available": True,
        })
    return {"models": models}


@app.get("/v1/distillation/{job_id}", tags=["distillation"])
async def get_distillation_job(job_id: str, tenant_id: str = Query("default")):
    from ..distillation import repository as dist_repo
    from ..distillation.serialization import serialize_job as _serialize_job

    tid = tenant_id or "default"
    job = dist_repo.get_job(tid, job_id)
    if not job or job.get("status") == "deleted":
        raise HTTPException(status_code=404, detail="Job not found")
    return _serialize_job(job)


@app.delete("/v1/distillation/{job_id}", tags=["distillation"])
async def delete_distillation_job(job_id: str, tenant_id: str = Query("default")):
    from ..distillation import repository as dist_repo
    from ..distillation.pipeline import cancel_pipeline

    tid = tenant_id or "default"
    cancel_pipeline(job_id)

    ok = dist_repo.delete_job(tid, job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": True}


@app.post("/v1/distillation/{job_id}/cancel", tags=["distillation"])
async def cancel_distillation_job(job_id: str, tenant_id: str = Query("default")):
    from ..distillation import repository as dist_repo
    from ..distillation.pipeline import cancel_pipeline
    from ..distillation.serialization import serialize_job as _serialize_job

    tid = tenant_id or "default"
    job = dist_repo.get_job(tid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    active_statuses = {"pending", "queued", "running"}
    if job.get("status") in active_statuses:
        cancel_pipeline(job_id)
        dist_repo.update_job_status(tid, job_id, status="cancelled", error="Cancelled by user")
        job = dist_repo.get_job(tid, job_id) or job

    return _serialize_job(job)


@app.get("/v1/distillation/{job_id}/logs", tags=["distillation"])
async def get_distillation_logs(job_id: str, tenant_id: str = Query("default")):
    from ..distillation import repository as dist_repo

    tid = tenant_id or "default"
    job = dist_repo.get_job(tid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    logs = job.get("pipeline_logs", [])
    if not isinstance(logs, list):
        logs = []
    return {"logs": logs}


@app.get("/v1/distillation/{job_id}/candidates", tags=["distillation"])
async def get_distillation_candidates(
    job_id: str,
    tenant_id: str = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    from typing import Any as _Any
    from ..distillation import repository as dist_repo

    candidates = dist_repo.get_candidates(job_id, limit=limit * 10)

    grouped: dict[str, list[dict[str, _Any]]] = {}
    for c in candidates:
        pid = c.get("prompt_id", "")
        grouped.setdefault(pid, []).append(c)

    samples = []
    for prompt_id, prompt_candidates in list(grouped.items())[:limit]:
        prompt_text = prompt_candidates[0].get("prompt", "") if prompt_candidates else ""
        cands = []
        for c in prompt_candidates:
            cands.append({
                "text": c.get("response", ""),
                "scores": {"quality": float(c.get("score", 0))},
                "isWinner": bool(c.get("selected", 0)),
                "candidate_idx": int(c.get("candidate_idx", 0)),
            })
        samples.append({
            "prompt": prompt_text,
            "trace_id": prompt_id,
            "candidates": cands,
        })

    return {"samples": samples, "total": len(grouped)}


@app.get("/v1/distillation/{job_id}/artifacts", tags=["distillation"])
async def get_distillation_artifacts(job_id: str, tenant_id: str = Query("default")):
    from pathlib import Path
    from ..distillation import repository as dist_repo

    tid = tenant_id or "default"
    job = dist_repo.get_job(tid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    artifacts_data = job.get("artifacts", {})
    gguf_map = artifacts_data.get("gguf", {}) if isinstance(artifacts_data, dict) else {}

    result = []
    for quant_type, file_path in gguf_map.items():
        path = Path(file_path) if file_path else None
        size = path.stat().st_size if path and path.exists() else 0
        result.append({
            "key": quant_type,
            "size": size,
            "url": f"/v1/distillation/{job_id}/download/{quant_type}",
            "expires_in": 0,
        })

    return result


@app.post("/v1/distillation/{job_id}/deploy", tags=["distillation"])
async def deploy_distilled_job(
    job_id: str,
    body: dict | None = None,
    tenant_id: str = Query("default"),
):
    """Deploy a completed distillation job's GGUF model via llama-server.

    Picks the smallest-quality GGUF artifact (preferring q4_k_m → q5_k_m → q8_0 → first available),
    launches llama-server in the background, and registers it as a local deployment.
    Returns immediately; UI polls /v1/deployments/{deployment_id}/status for readiness.
    """
    from pathlib import Path as _Path
    from ..distillation import repository as dist_repo
    from ..deployment.manager import deploy_model

    tid = tenant_id or "default"
    job = dist_repo.get_job(tid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job must be completed before deploy (current status: {job.get('status')})",
        )

    artifacts_data = job.get("artifacts", {}) or {}
    gguf_map = artifacts_data.get("gguf", {}) if isinstance(artifacts_data, dict) else {}
    if not gguf_map:
        raise HTTPException(
            status_code=400,
            detail="No GGUF artifacts found for this job — export must complete successfully first",
        )

    preferred_order = ["q4_k_m", "q5_k_m", "q4_0", "q5_0", "q8_0", "f16"]
    selected_quant = None
    selected_path = None
    for quant in preferred_order:
        if quant in gguf_map and _Path(gguf_map[quant]).exists():
            selected_quant = quant
            selected_path = gguf_map[quant]
            break
    if not selected_path:
        for quant, fp in gguf_map.items():
            if fp and _Path(fp).exists():
                selected_quant = quant
                selected_path = fp
                break

    if not selected_path:
        raise HTTPException(
            status_code=400,
            detail="GGUF artifacts referenced in job results no longer exist on disk",
        )

    body = body or {}
    instance_type = body.get("instance_type", "cpu-small")
    config = body.get("config", {}) or {}

    job_name = job.get("name") or f"distilled-{job_id[:8]}"
    model_id = f"distilled/{job_name}"

    deploy_config = {
        **config,
        "instance_type": instance_type,
        "model_alias": model_id,
        "source_job_id": job_id,
        "quantization": selected_quant,
    }

    result = await deploy_model(
        model_id=model_id,
        model_path=selected_path,
        config=deploy_config,
    )

    return {
        "deployment_id": result["deployment_id"],
        "model_id": result["model_id"],
        "status": result["status"],
        "endpoint_url": result.get("endpoint_url", ""),
        "already_deployed": result.get("already_deployed", False),
        "quantization": selected_quant,
        "model_path": selected_path,
    }


@app.get("/v1/distillation/{job_id}/metrics", tags=["distillation"])
async def get_distillation_metrics(
    job_id: str,
    tenant_id: str = Query("default"),
    limit: int = Query(5000, ge=1, le=50000),
):
    from ..distillation import repository as dist_repo

    metrics = dist_repo.get_metrics(job_id, limit=limit)
    return {"metrics": metrics, "total": len(metrics)}


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
