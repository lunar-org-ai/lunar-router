from pydantic import BaseModel
from typing import Dict, List, Optional
from decimal import Decimal


class ProviderAttempt(BaseModel):
    """Represents a single provider attempt during a request."""

    provider: str
    success: bool
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: float
    timestamp: str


class TraceFilters(BaseModel):
    """Filters for trace queries - applied server-side before pagination."""

    model_id: Optional[str] = None  # Filter by model ID
    backend: Optional[str] = None   # Filter by provider/backend
    is_success: Optional[bool] = None  # Filter by success status
    search: Optional[str] = None    # Search in input/output text

    def has_filters(self) -> bool:
        """Check if any filters are active."""
        return any([self.model_id, self.backend, self.is_success is not None, self.search])


class TenantStatsModel(BaseModel):
    TenantId: str
    CreationDate: str
    Provider: Optional[str] = None
    Model: Optional[str] = None
    Cost: Optional[Decimal] = None
    TTFT: Optional[Decimal] = None
    Latency: Optional[Decimal] = None
    Success: Optional[bool] = None
    ErrorType: Optional[str] = None
    InputText: Optional[str] = None
    OutputText: Optional[str] = None
    TotalTokens: Optional[int] = None
    # New fields for enhanced error tracking
    ErrorCategory: Optional[str] = None  # Classified error type
    ErrorMessage: Optional[str] = None  # Detailed error message
    ProviderAttempts: Optional[str] = None  # JSON serialized List[ProviderAttempt]
    FallbackCount: Optional[int] = None  # Number of providers attempted
    FinalProvider: Optional[str] = None  # Provider that handled the request
    # Semantic routing fields
    RoutingInfo: Optional[str] = None  # JSON with routing decision metadata

class GlobalStatsModel(BaseModel):
    Provider: str
    CreationDate: str
    Model: Optional[str] = None
    Cost: Optional[Decimal] # fixed cost per 1m tokens
    MeanTTFT: Optional[Decimal]
    MeanLatency: Optional[Decimal]
    SuccessCount: Optional[int]
    ErrorCount: Optional[int]
    ErrorTypes: Dict[str, int]

class StatsResponseModel(BaseModel):
    p50_lat: float
    p50_ttft: float
    err_rate: float
    n: int
    updated_at: float
    updated_at_date: str

class PricingItem(BaseModel):
    Provider: str
    Model: str
    ModelId: Optional[str] = None
    UpdatedAt: str
    input_per_million: float
    output_per_million: float
    cache_input_per_million: float


class PricingResponse(BaseModel):
    ModelId: Optional[str] = None
    UpdatedAt: Optional[str] = None
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cache_input_per_million: float = 0.0

class PriceRequest(BaseModel):
    provider: str
    model: str
    modelid: Optional[str] = None
    input_per_million: float
    output_per_million: float
    cache_input_per_million: float

class DashboardStatsModel(BaseModel):
    total_cost: float
    total_requests: int
    avg_latency_p95: float
    error_rate: float
    cost_by_provider: Dict[str, float]
    usage_by_model: Dict[str, int]
    total_cost_change: str  # Percentage change from previous period
    total_requests_change: str  # Percentage change from previous period
    avg_latency_change: str  # Percentage change from previous period
    error_rate_change: str  # Percentage change from previous period

class CostTimeSeriesPoint(BaseModel):
    date: str  # YYYY-MM-DD format
    cost: float

class ExpensiveRequest(BaseModel):
    request_id: str
    model: str
    provider: str
    cost: float
    tokens_in: int
    tokens_out: int
    creation_date: str
    input_preview: str  # First 100 chars of input

class CostAnalyticsModel(BaseModel):
    cost_over_time: list[CostTimeSeriesPoint]
    monthly_projection: float
    cost_by_model: Dict[str, float]
    most_expensive_requests: list[ExpensiveRequest]

class LatencyTimeSeriesPoint(BaseModel):
    date: str  # YYYY-MM-DD format
    p95_latency: float
    p50_latency: float
    avg_latency: float

class LatencyDistributionBucket(BaseModel):
    label: str  # e.g., "<100ms", "100-500ms"
    count: int
    percentage: float

class ErrorRateTimeSeriesPoint(BaseModel):
    date: str  # YYYY-MM-DD format
    error_rate: float  # Percentage
    total_requests: int
    failed_requests: int

class RecentError(BaseModel):
    error_id: str
    model: str
    provider: str
    error_type: str
    timestamp: str
    input_preview: str  # First 100 chars of input
    latency: float

class PerformanceAnalyticsModel(BaseModel):
    latency_over_time: list[LatencyTimeSeriesPoint]
    latency_distribution: list[LatencyDistributionBucket]
    error_rate_over_time: list[ErrorRateTimeSeriesPoint]
    recent_errors: list[RecentError]

# Models for comprehensive dashboard response
class TimeSeriesDataPoint(BaseModel):
    time: str
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_s: float
    p95_latency_s: float
    error_rate: float

class ModelStatsPoint(BaseModel):
    model_id: str
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_s: float
    p95_latency_s: float
    error_rate: float

class DashboardTotals(BaseModel):
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Optional[float] = None
    success_rate: float
    avg_latency_s: float
    p95_latency_s: float
    avg_cost_per_1k_tokens_usd: Optional[float] = None
    streaming_share: Optional[float] = None

class DistributionStats(BaseModel):
    p50: float
    p90: float
    p95: float
    p99: float
    mean: float
    std: float

class DashboardDistributions(BaseModel):
    latency_s: DistributionStats
    ttft_s: DistributionStats
    input_tokens: DistributionStats
    output_tokens: DistributionStats
    cost_per_request_usd: DistributionStats

class PeriodTrends(BaseModel):
    requests: int
    cost_usd: Optional[float] = None
    p95_latency_s: float
    error_rate: float

class DashboardTrends(BaseModel):
    last_7d: PeriodTrends
    prev_7d: PeriodTrends
    pct_change: Dict[str, Optional[float]]

class TopCostModel(BaseModel):
    model_id: str
    cost_usd: float
    count: int

class SlowestModel(BaseModel):
    model_id: str
    p95_latency_s: float
    count: int

class ErrorModel(BaseModel):
    model_id: str
    error_count: int

class DashboardLeaders(BaseModel):
    top_cost_models: list[TopCostModel]
    slowest_models_p95_latency: list[SlowestModel]
    most_errors_models: list[ErrorModel]

class DashboardSeries(BaseModel):
    by_time: list[TimeSeriesDataPoint]
    by_model: list[ModelStatsPoint]
    by_backend: list  # Empty for now
    by_deployment: list  # Empty for now

class RawSampleItem(BaseModel):
    timestamp: str
    model: str
    provider: str
    success: bool
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float] = None
    input_preview: str
    output_preview: str

class CompleteDashboardResponse(BaseModel):
    series: DashboardSeries
    totals: DashboardTotals
    distributions: DashboardDistributions
    trends: DashboardTrends
    leaders: DashboardLeaders
    insights: list  # Empty for now
    raw_sample: Optional[list[RawSampleItem]] = None

# Analytics Response Models
class AnalyticsTotals(BaseModel):
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    success_rate: float
    avg_latency_s: float
    p95_latency_s: float
    avg_cost_per_1k_tokens_usd: float
    streaming_share: float

class AnalyticsTimeSeriesData(BaseModel):
    time: str
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_s: float
    p95_latency_s: float
    error_rate: float
    total_cost_usd: float

class AnalyticsModelSeriesData(BaseModel):
    model_id: str
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_s: float
    p95_latency_s: float
    error_rate: float
    total_cost_usd: float

class AnalyticsBackendSeriesData(BaseModel):
    backend: str
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_s: float
    p95_latency_s: float
    error_rate: float
    total_cost_usd: float

class AnalyticsDistributionData(BaseModel):
    p50: float
    p90: float
    p95: float
    p99: float
    mean: float
    std: float

class AnalyticsDistributions(BaseModel):
    latency_s: AnalyticsDistributionData
    ttft_s: AnalyticsDistributionData
    input_tokens: AnalyticsDistributionData
    output_tokens: AnalyticsDistributionData
    cost_per_request_usd: AnalyticsDistributionData

class AnalyticsTrendData(BaseModel):
    requests: int
    cost_usd: float
    p95_latency_s: float
    error_rate: float

class AnalyticsTrends(BaseModel):
    last_7d: AnalyticsTrendData
    prev_7d: AnalyticsTrendData
    pct_change: Dict[str, Optional[float]]

class AnalyticsCostLeaderData(BaseModel):
    model_id: str
    total_cost_usd: float
    request_count: int

class AnalyticsLatencyLeaderData(BaseModel):
    model_id: str
    p95_latency_s: float
    count: int

class AnalyticsErrorLeaderData(BaseModel):
    model_id: str
    error_count: int
    total_requests: int
    error_rate: float

class AnalyticsLeaders(BaseModel):
    top_cost_models: list[AnalyticsCostLeaderData]
    slowest_models_p95_latency: list[AnalyticsLatencyLeaderData]
    most_errors_models: list[AnalyticsErrorLeaderData]

class AnalyticsInsightData(BaseModel):
    type: str  # 'anomaly' | 'trend' | 'projection'
    message: str
    severity: str  # 'low' | 'medium' | 'high'
    data: Optional[Dict] = None

class AnalyticsRawSampleData(BaseModel):
    event_id: str
    id: Optional[str] = None
    model_id: str
    created_at: str
    latency_s: float
    ttft_s: float
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    cost_usd: Optional[float] = None
    is_success: bool
    success: Optional[bool] = None
    error_code: Optional[str] = None
    backend: str
    deployment_id: Optional[str] = None
    is_stream: bool
    input_preview: Optional[str] = None
    output_preview: Optional[str] = None
    input_messages: Optional[list] = None
    output_text: Optional[str] = None
    endpoint: Optional[str] = None
    history: Optional[str] = None

class AnalyticsSeries(BaseModel):
    by_time: list[AnalyticsTimeSeriesData]
    by_model: list[AnalyticsModelSeriesData]
    by_backend: list[AnalyticsBackendSeriesData]

class AnalyticsMetricsResponse(BaseModel):
    totals: AnalyticsTotals
    series: AnalyticsSeries
    distributions: AnalyticsDistributions
    trends: AnalyticsTrends
    leaders: AnalyticsLeaders
    insights: list[AnalyticsInsightData]
    raw_sample: Optional[list[AnalyticsRawSampleData]] = None
    total_traces: Optional[int] = None  # Total number of traces for pagination