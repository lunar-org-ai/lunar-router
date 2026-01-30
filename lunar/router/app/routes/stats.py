from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from ..database.TenantStatsHandler import TenantStatsHandler
from ..database.GlobalStatsHandler import GlobalStatsHandler
from ..database.models import DashboardStatsModel, CostAnalyticsModel, PerformanceAnalyticsModel, CompleteDashboardResponse, AnalyticsMetricsResponse, TraceFilters
from openpyxl import Workbook
from io import BytesIO
from typing import Optional

router = APIRouter(prefix="/v1/stats", tags=["stats"])

@router.get("")
async def get_stats(
    request: Request,
    model: str = Query(..., description="Logical model name"),
    days: int = Query(15, description="Number of days to include in stats"),
):
    tenant_id = getattr(request.state, 'tenant_id', 'default')
    return await TenantStatsHandler.all_for_model(model=model, days=days, tenantId=tenant_id)


@router.get("/global")
async def get_stats_global(
    model: str = Query(..., description="Logical model name"),
):
    return await GlobalStatsHandler.all_for_model(model)

@router.get("/overview", response_model=DashboardStatsModel)
async def get_tenant_dashboard_stats(
    request: Request,
    days: int = Query(7, description="Number of days to include in stats"),
):
    """
    Returns dashboard statistics for a tenant for the last N days.
    Includes metrics like total cost, requests, latency P95, error rate,
    and breakdown by provider and model, with percentage changes from previous period.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')
    return await TenantStatsHandler.get_dashboard_stats(tenant_id, days)

@router.get("/costs", response_model=CostAnalyticsModel)
async def get_tenant_cost_analytics(
    request: Request,
    days: int = Query(30, description="Number of days to include in cost analysis"),
):
    """
    Returns detailed cost analytics for a tenant including:
    - Cost over time (daily breakdown)
    - Monthly projection based on current usage
    - Cost breakdown by model
    - Most expensive individual requests
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')
    return await TenantStatsHandler.get_cost_analytics(tenant_id, days)

@router.get("/performance", response_model=PerformanceAnalyticsModel)
async def get_tenant_performance_analytics(
    request: Request,
    days: int = Query(30, description="Number of days to include in performance analysis"),
):
    """
    Returns detailed performance analytics for a tenant including:
    - Latency metrics over time (P95, P50, average)
    - Latency distribution in buckets
    - Error rate over time
    - Recent errors with details
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')
    return await TenantStatsHandler.get_performance_analytics(tenant_id, days)

@router.get("/dashboard", response_model=CompleteDashboardResponse)
async def get_complete_dashboard_metrics(
    request: Request,
    days: int = Query(7, description="Number of days to include in dashboard metrics"),
):
    """
    TEMPORARY ENDPOINT - Returns comprehensive dashboard metrics in a single call.
    This endpoint combines all dashboard data (series, totals, distributions, trends, leaders)
    to support quick frontend integration. Should be replaced with proper endpoints later.

    Returns data matching the DashboardMetricsResponse interface expected by the frontend.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')
    return await TenantStatsHandler.get_complete_dashboard_metrics(tenant_id, days)

@router.get("/{tenant_id}/dashboard", response_model=CompleteDashboardResponse)
async def get_complete_dashboard_metrics_by_tenant(
    tenant_id: str,
    days: int = Query(7, description="Number of days to include in dashboard metrics"),
):
    """
    Returns comprehensive dashboard metrics for a specific tenant.
    """
    return await TenantStatsHandler.get_complete_dashboard_metrics(tenant_id, days)

@router.get("/analytics", response_model=AnalyticsMetricsResponse)
async def get_analytics_metrics(
    request: Request,
    days: int = Query(30, description="Number of days to include in analytics metrics"),
    trace_limit: int = Query(20, description="Number of traces to return in raw_sample (use 0 for all)"),
    trace_offset: int = Query(0, description="Offset for trace pagination"),
    # Filters for traces
    model_id: Optional[str] = Query(None, description="Filter traces by model ID"),
    backend: Optional[str] = Query(None, description="Filter traces by backend/provider (openai, bedrock, anthropic, pureai)"),
    is_success: Optional[bool] = Query(None, description="Filter traces by success status (true/false)"),
    search: Optional[str] = Query(None, description="Search in input/output text"),
):
    """
    Returns comprehensive analytics metrics in a single call.
    This endpoint provides detailed analytics data including time series, model breakdowns,
    backend performance, distributions, trends, and leader boards.

    Pagination for traces:
    - trace_limit: Number of traces to return (default: 20, use 0 for all traces)
    - trace_offset: Starting offset for pagination (default: 0)
    - total_traces: Total count of traces matching filters (returned in response)

    Filters for traces (applied server-side before pagination):
    - model_id: Filter by specific model
    - backend: Filter by provider/backend
    - is_success: Filter by success/failure status
    - search: Search text in input/output
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')
    filters = TraceFilters(
        model_id=model_id,
        backend=backend,
        is_success=is_success,
        search=search
    )
    return await TenantStatsHandler.get_analytics_metrics(tenant_id, days, trace_limit, trace_offset, filters)

@router.get("/{tenant_id}/analytics", response_model=AnalyticsMetricsResponse)
async def get_analytics_metrics_by_tenant(
    tenant_id: str,
    days: int = Query(30, description="Number of days to include in analytics metrics"),
    trace_limit: int = Query(20, description="Number of traces to return in raw_sample (use 0 for all)"),
    trace_offset: int = Query(0, description="Offset for trace pagination"),
    # Filters for traces
    model_id: Optional[str] = Query(None, description="Filter traces by model ID"),
    backend: Optional[str] = Query(None, description="Filter traces by backend/provider (openai, bedrock, anthropic, pureai)"),
    is_success: Optional[bool] = Query(None, description="Filter traces by success status (true/false)"),
    search: Optional[str] = Query(None, description="Search in input/output text"),
):
    """
    Returns comprehensive analytics metrics for a specific tenant.

    Pagination for traces:
    - trace_limit: Number of traces to return (default: 20, use 0 for all traces)
    - trace_offset: Starting offset for pagination (default: 0)
    - total_traces: Total count of traces matching filters (returned in response)

    Filters for traces (applied server-side before pagination):
    - model_id: Filter by specific model
    - backend: Filter by provider/backend
    - is_success: Filter by success/failure status
    - search: Search text in input/output
    """
    filters = TraceFilters(
        model_id=model_id,
        backend=backend,
        is_success=is_success,
        search=search
    )
    return await TenantStatsHandler.get_analytics_metrics(tenant_id, days, trace_limit, trace_offset, filters)

@router.get("/excel")
async def get_tenant_stats_excel(
    request: Request,
    days: int = Query(15, description="Number of days to include in stats"),
):
    """
    Returns an Excel file with the tenant's metrics for the last N days.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    items = await TenantStatsHandler.get_raw_data(tenant_id, days)

    wb = Workbook()
    ws = wb.active
    ws.title = "Tenant Stats"

    ws.append(["TenantId", "Model", "Provider", "CreationDate", "Latency", "TTFT", "Success", "Cost", "TotalTokens", "InputPreview", "OutputPreview"])
    for i in items:
        # Create input and output previews (first 100 characters)
        input_text = i.get("InputText", "")
        output_text = i.get("OutputText", "")
        
        input_preview = input_text[:100] if input_text else ""
        if len(input_text) > 100:
            input_preview += "..."
            
        output_preview = output_text[:100] if output_text else ""
        if len(output_text) > 100:
            output_preview += "..."
        
        ws.append([
            i.get("TenantId"),
            i.get("Model"),
            i.get("Provider"),
            i.get("CreationDate"),
            i.get("Latency"),
            i.get("TTFT"),
            i.get("Success"),
            i.get("Cost"),
            i.get("TotalTokens"),
            input_preview,
            output_preview,
        ])

    stream = BytesIO()
    wb.save(stream)
    stream.seek(0)

    filename = f"tenant_{tenant_id}_stats.xlsx"

    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )