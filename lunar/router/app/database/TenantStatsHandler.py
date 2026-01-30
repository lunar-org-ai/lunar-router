import aioboto3
from boto3.dynamodb.conditions import Attr
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from .StatsHandlerBase import StatsHandlerBase
from .models import TenantStatsModel, StatsResponseModel
from .GlobalStatsHandler import GlobalStatsHandler
from collections import defaultdict
import re
import statistics
import calendar
from .models import DashboardStatsModel
from .models import (
    CostAnalyticsModel, 
    CostTimeSeriesPoint, 
    ExpensiveRequest,
    PerformanceAnalyticsModel,
    LatencyTimeSeriesPoint,
    LatencyDistributionBucket, 
    ErrorRateTimeSeriesPoint, 
    RecentError)

class TenantStatsHandler(StatsHandlerBase):
    region: str = "us-east-1"
    table_name: str = "TenantStats"

    @staticmethod
    def _extract_history_blocks(input_text: str) -> Tuple[str, str]:
        if not input_text:
            return input_text, ""
    
        pattern = r'<(memory|history)>(.*?)</\1>'
        
        matches = re.findall(pattern, input_text, re.DOTALL | re.IGNORECASE)
        if not matches:
            return input_text, ""
        
        history_blocks = ""
        for tag_name, content in matches:
            history_blocks += f"<{tag_name}>{content}</{tag_name}>"
        
        cleaned_text = re.sub(pattern, "", input_text, flags=re.DOTALL | re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text, history_blocks

    @staticmethod
    async def insert(stats: TenantStatsModel):
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=TenantStatsHandler.region) as client:
            cleaned_input_text, history_blocks = TenantStatsHandler._extract_history_blocks(stats.InputText or "")
            
            # Build item with all available fields
            item = {
                "TenantId": {"S": stats.TenantId},
                "CreationDate": {"S": stats.CreationDate},
                "Success": {"BOOL": stats.Success},
            }
            
            # Define field mappings for optional fields
            field_mappings = {
                # String fields
                'Model': ('S', stats.Model),
                'Provider': ('S', stats.Provider),
                'ErrorType': ('S', stats.ErrorType),
                'InputText': ('S', cleaned_input_text),  
                'OutputText': ('S', stats.OutputText),
                # New error tracking fields
                'ErrorCategory': ('S', stats.ErrorCategory),
                'ErrorMessage': ('S', stats.ErrorMessage),
                'ProviderAttempts': ('S', stats.ProviderAttempts),
                'FinalProvider': ('S', stats.FinalProvider),
                # Numeric fields
                'Cost': ('N', stats.Cost),
                'Latency': ('N', stats.Latency),
                'TTFT': ('N', stats.TTFT),
                'TotalTokens': ('N', stats.TotalTokens),
                'FallbackCount': ('N', stats.FallbackCount),
            }
            
            # Add non-null/non-empty fields to item
            for field_name, (dynamo_type, value) in field_mappings.items():
                if value is not None and value != "":
                    if dynamo_type == 'N':
                        item[field_name] = {dynamo_type: str(value)}
                    else:
                        item[field_name] = {dynamo_type: value}
            await client.put_item(
                TableName=TenantStatsHandler.table_name,
                Item=item
            )
        await GlobalStatsHandler.insert(stats)

    @staticmethod
    async def summary(model: str, provider: str, days: int) -> StatsResponseModel:
        today = datetime.utcnow()
        start_date = today - timedelta(days=days)

        start_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_str = today.strftime("%Y-%m-%dT23:59:59")

        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=TenantStatsHandler.region) as client:
            response = await client.scan(
                TableName=TenantStatsHandler.table_name,
                FilterExpression="Model = :model AND Provider = :provider AND CreationDate BETWEEN :start AND :end",
                ExpressionAttributeValues={
                    ":model": {"S": model},
                    ":provider": {"S": provider},
                    ":start": {"S": start_str},
                    ":end": {"S": end_str},
                },
            )

            raw_items = response.get("Items", [])
            items = [TenantStatsHandler.deserialize_item(item) for item in raw_items]

            if not items:
                return StatsResponseModel(
                    p50_lat=float("inf"),
                    p50_ttft=float("inf"),
                    err_rate=0.0,
                    n=0,
                    updated_at=0.0,
                    updated_at_date=""
                )

            latencies = [float(i["Latency"]) for i in items if "Latency" in i]
            ttfts = [float(i["TTFT"]) for i in items if "TTFT" in i]
            errors = [i for i in items if not i.get("Success", True)]

            n = len(items)
            err_rate = len(errors) / n if n else 0.0

            last_date_str = max(i.get("CreationDate", "1970-01-01T00:00:00") for i in items)
            try:
                dt_obj = datetime.fromisoformat(last_date_str)
                updated_at = dt_obj.timestamp()
                updated_at_date = dt_obj.strftime("%Y-%m-%d")
            except Exception:
                updated_at = 0.0
                updated_at_date = ""

            return StatsResponseModel(
                p50_lat=TenantStatsHandler._percentile(latencies, 50),
                p50_ttft=TenantStatsHandler._percentile(ttfts, 50),
                err_rate=err_rate,
                n=n,
                updated_at=updated_at,
                updated_at_date=updated_at_date
            )

    @staticmethod
    async def all_for_model(model: str, days: int = 15, tenantId: str = "default") -> Dict[str, Dict[str, float]]:
        today = datetime.utcnow()
        start_date = today - timedelta(days=days)

        start_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_str = today.strftime("%Y-%m-%dT23:59:59")

        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=TenantStatsHandler.region) as client:
            response = await client.scan(
                TableName=TenantStatsHandler.table_name,
                FilterExpression="TenantId = :tenantId AND Model = :model AND CreationDate BETWEEN :start AND :end",
                ExpressionAttributeValues={
                    ":tenantId": {"S": tenantId},
                    ":model": {"S": model},
                    ":start": {"S": start_str},
                    ":end": {"S": end_str},
                },
            )

            items = response.get("Items", [])

            providers = set(i["Provider"]["S"] for i in items if "Provider" in i)

            out: Dict[str, Dict[str, float]] = {}
            for provider in providers:
                out[provider] = await TenantStatsHandler.summary(model, provider, days)

            return out
        
    @staticmethod
    async def get_raw_data(tenantId: str, days: int = 15) -> List[dict]:
        """
        Retrieve raw tenant stats data for a given tenantId and time range.
        Uses Query (not Scan) for efficiency - leverages TenantId (HASH) and CreationDate (RANGE) keys.
        Handles pagination to fetch all matching items.
        """
        today = datetime.utcnow()
        start_date = today - timedelta(days=days)

        start_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_str = today.strftime("%Y-%m-%dT23:59:59")

        session = aioboto3.Session()
        all_items = []

        async with session.client("dynamodb", region_name=TenantStatsHandler.region) as client:
            query_kwargs = {
                "TableName": TenantStatsHandler.table_name,
                "KeyConditionExpression": "TenantId = :tenantId AND CreationDate BETWEEN :start AND :end",
                "ExpressionAttributeValues": {
                    ":tenantId": {"S": tenantId},
                    ":start": {"S": start_str},
                    ":end": {"S": end_str},
                },
                "ScanIndexForward": False,  # Return items in descending order by CreationDate
            }

            while True:
                response = await client.query(**query_kwargs)
                raw_items = response.get("Items", [])
                all_items.extend([TenantStatsHandler.deserialize_item(item) for item in raw_items])

                # Check if there are more items to fetch
                if "LastEvaluatedKey" not in response:
                    break
                query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

        return all_items
    
    @staticmethod
    async def get_dashboard_stats(tenantId: str, days: int = 7) -> dict:
        """
        Get dashboard statistics for a tenant for the specified number of days.
        """
        # Get current period data
        current_items = await TenantStatsHandler.get_raw_data(tenantId, days)
        
        # Get previous period data for comparison
        previous_items = await TenantStatsHandler.get_raw_data_with_offset(tenantId, days, days)
        
        # Calculate current period metrics
        current_metrics = TenantStatsHandler._calculate_period_metrics(current_items)
        previous_metrics = TenantStatsHandler._calculate_period_metrics(previous_items)
        
        # Calculate percentage changes
        def calculate_change(current: float, previous: float) -> str:
            if previous == 0:
                return "NaN%" if current == 0 else "+∞%"
            change = ((current - previous) / previous) * 100
            if change == 0:
                return "0%"
            elif change > 0:
                return f"+{change:.1f}%"
            else:
                return f"{change:.1f}%"
        
        return DashboardStatsModel(
            total_cost=current_metrics["total_cost"],
            total_requests=current_metrics["total_requests"],
            avg_latency_p95=current_metrics["avg_latency_p95"],
            error_rate=current_metrics["error_rate"],
            cost_by_provider=current_metrics["cost_by_provider"],
            usage_by_model=current_metrics["usage_by_model"],
            total_cost_change=calculate_change(
                current_metrics["total_cost"], 
                previous_metrics["total_cost"]
            ),
            total_requests_change=calculate_change(
                current_metrics["total_requests"], 
                previous_metrics["total_requests"]
            ),
            avg_latency_change=calculate_change(
                current_metrics["avg_latency_p95"], 
                previous_metrics["avg_latency_p95"]
            ),
            error_rate_change=calculate_change(
                current_metrics["error_rate"], 
                previous_metrics["error_rate"]
            )
        ).dict()
    
    @staticmethod
    async def get_raw_data_with_offset(tenantId: str, days: int = 15, offset_days: int = 0) -> List[dict]:
        """
        Retrieve raw tenant stats data for a given tenantId and time range with offset.
        Uses Query (not Scan) for efficiency - leverages TenantId (HASH) and CreationDate (RANGE) keys.
        Handles pagination to fetch all matching items.
        """
        today = datetime.utcnow() - timedelta(days=offset_days)
        start_date = today - timedelta(days=days)

        start_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_str = today.strftime("%Y-%m-%dT23:59:59")

        session = aioboto3.Session()
        all_items = []

        async with session.client("dynamodb", region_name=TenantStatsHandler.region) as client:
            query_kwargs = {
                "TableName": TenantStatsHandler.table_name,
                "KeyConditionExpression": "TenantId = :tenantId AND CreationDate BETWEEN :start AND :end",
                "ExpressionAttributeValues": {
                    ":tenantId": {"S": tenantId},
                    ":start": {"S": start_str},
                    ":end": {"S": end_str},
                },
                "ScanIndexForward": False,  # Return items in descending order by CreationDate
            }

            while True:
                response = await client.query(**query_kwargs)
                raw_items = response.get("Items", [])
                all_items.extend([TenantStatsHandler.deserialize_item(item) for item in raw_items])

                # Check if there are more items to fetch
                if "LastEvaluatedKey" not in response:
                    break
                query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

        return all_items
    
    @staticmethod
    def _calculate_period_metrics(items: List[dict]) -> dict:
        """
        Calculate metrics for a given period of data.
        """
        
        if not items:
            return {
                "total_cost": 0.0,
                "total_requests": 0,
                "avg_latency_p95": 0.0,
                "error_rate": 0.0,
                "cost_by_provider": {},
                "usage_by_model": {}
            }
        
        # Initialize counters
        total_cost = 0.0
        total_requests = len(items)
        error_count = 0
        latencies = []
        cost_by_provider = defaultdict(float)
        usage_by_model = defaultdict(int)
        
        for item in items:
            # Total cost
            cost = item.get("Cost", 0)
            if cost:
                cost_value = float(cost)
                total_cost += cost_value
                
                # Cost by provider
                provider = item.get("Provider", "unknown")
                cost_by_provider[provider] += cost_value
            
            # Error rate
            if not item.get("Success", True):
                error_count += 1
            
            # Latencies for P95 calculation
            latency = item.get("Latency")
            if latency:
                latencies.append(float(latency))
            
            # Usage by model
            model = item.get("Model", "unknown")
            usage_by_model[model] += 1
        
        # Calculate P95 latency
        avg_latency_p95 = 0.0
        if latencies:
            latencies.sort()
            p95_index = int(0.95 * len(latencies))
            avg_latency_p95 = latencies[min(p95_index, len(latencies) - 1)]
        
        # Calculate error rate
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "total_cost": round(total_cost, 8),
            "total_requests": total_requests,
            "avg_latency_p95": round(avg_latency_p95, 2),
            "error_rate": round(error_rate, 1),
            "cost_by_provider": dict(cost_by_provider),
            "usage_by_model": dict(usage_by_model)
        }
    
    @staticmethod
    async def get_cost_analytics(tenantId: str, days: int = 30) -> dict:
        """
        Get detailed cost analytics for a tenant including time series, projections, and breakdowns.
        """
        
        # Get raw data for the specified period
        items = await TenantStatsHandler.get_raw_data(tenantId, days)
        
        if not items:
            return CostAnalyticsModel(
                cost_over_time=[],
                monthly_projection=0.0,
                cost_by_model={},
                most_expensive_requests=[]
            ).dict()
        
        # 1. Cost Over Time (daily breakdown)
        daily_costs = defaultdict(float)
        cost_by_model = defaultdict(float)
        expensive_requests = []
        
        for item in items:
            # Parse date to get day only
            creation_date = item.get("CreationDate", "")
            try:
                date_part = creation_date.split("T")[0]  # Get YYYY-MM-DD part
            except:
                continue
                
            cost = item.get("Cost", 0)
            if cost:
                cost_value = float(cost)
                daily_costs[date_part] += cost_value
                
                # Cost by model
                model = item.get("Model", "unknown")
                cost_by_model[model] += cost_value
                
                # Collect expensive requests data
                expensive_requests.append({
                    "cost": cost_value,
                    "model": model,
                    "provider": item.get("Provider", "unknown"),
                    "creation_date": creation_date,
                    "input_text": item.get("InputText", ""),
                    "total_tokens": item.get("TotalTokens", 0),
                    "tokens_in": item.get("TotalTokens", 0) // 2,  # Approximate split
                    "tokens_out": item.get("TotalTokens", 0) // 2
                })
        
        # Sort daily costs by date and create time series
        cost_over_time = []
        for date in sorted(daily_costs.keys()):
            cost_over_time.append(CostTimeSeriesPoint(
                date=date,
                cost=round(daily_costs[date], 8)
            ))
        
        # 2. Monthly Projection
        total_cost = sum(daily_costs.values())
        if len(daily_costs) > 0:
            # Calculate average daily cost and project to full month
            avg_daily_cost = total_cost / len(daily_costs)
            # Get number of days in current month
            today = datetime.utcnow()
            days_in_month = calendar.monthrange(today.year, today.month)[1]
            monthly_projection = avg_daily_cost * days_in_month
        else:
            monthly_projection = 0.0
        
        # 3. Most Expensive Requests (top 10)
        expensive_requests.sort(key=lambda x: x["cost"], reverse=True)
        top_expensive = []
        for i, req in enumerate(expensive_requests[:10]):
            # Create a unique request ID
            request_id = f"req_{tenantId}_{i+1}"
            
            # Get full input text
            input_preview = req["input_text"] if req["input_text"] else ""
            
            top_expensive.append(ExpensiveRequest(
                request_id=request_id,
                model=req["model"],
                provider=req["provider"],
                cost=round(req["cost"], 8),
                tokens_in=req["tokens_in"],
                tokens_out=req["tokens_out"],
                creation_date=req["creation_date"],
                input_preview=input_preview
            ))
        
        return CostAnalyticsModel(
            cost_over_time=cost_over_time,
            monthly_projection=round(monthly_projection, 8),
            cost_by_model=dict(cost_by_model),
            most_expensive_requests=top_expensive
        ).dict()
    
    @staticmethod
    async def get_performance_analytics(tenantId: str, days: int = 30) -> dict:
        """
        Get detailed performance analytics for a tenant including latency and error metrics.
        """
        
        # Get raw data for the specified period
        items = await TenantStatsHandler.get_raw_data(tenantId, days)
        
        if not items:
            return PerformanceAnalyticsModel(
                latency_over_time=[],
                latency_distribution=[],
                error_rate_over_time=[],
                recent_errors=[]
            ).dict()
        
        # Group data by date
        daily_data = defaultdict(lambda: {
            'latencies': [],
            'total_requests': 0,
            'failed_requests': 0
        })
        
        all_latencies = []
        recent_errors = []
        
        for item in items:
            # Parse date
            creation_date = item.get("CreationDate", "")
            try:
                date_part = creation_date.split("T")[0]  # Get YYYY-MM-DD part
            except:
                continue
            
            # Process latency data
            latency = item.get("Latency")
            if latency:
                latency_value = float(latency)
                daily_data[date_part]['latencies'].append(latency_value)
                all_latencies.append(latency_value)
            
            # Process success/failure data
            daily_data[date_part]['total_requests'] += 1
            success = item.get("Success", True)
            
            if not success:
                daily_data[date_part]['failed_requests'] += 1
                
                # Collect error data
                error_type = item.get("ErrorType", "Unknown Error")
                input_text = item.get("InputText", "")
                input_preview = input_text if input_text else ""
                
                recent_errors.append({
                    "model": item.get("Model", "unknown"),
                    "provider": item.get("Provider", "unknown"), 
                    "error_type": error_type,
                    "timestamp": creation_date,
                    "input_preview": input_preview,
                    "latency": latency_value if latency else 0.0
                })
        
        # 1. Latency Over Time
        latency_over_time = []
        for date in sorted(daily_data.keys()):
            day_latencies = daily_data[date]['latencies']
            if day_latencies:
                day_latencies.sort()
                p50_index = int(0.5 * len(day_latencies))
                p95_index = int(0.95 * len(day_latencies))
                
                latency_over_time.append(LatencyTimeSeriesPoint(
                    date=date,
                    p95_latency=round(day_latencies[min(p95_index, len(day_latencies) - 1)], 2),
                    p50_latency=round(day_latencies[p50_index], 2),
                    avg_latency=round(statistics.mean(day_latencies), 2)
                ))
        
        # 2. Latency Distribution
        latency_buckets = {
            "<100ms": 0,
            "100-500ms": 0, 
            "500ms-1s": 0,
            ">1s": 0
        }
        
        for latency in all_latencies:
            if latency < 100:
                latency_buckets["<100ms"] += 1
            elif latency < 500:
                latency_buckets["100-500ms"] += 1
            elif latency < 1000:
                latency_buckets["500ms-1s"] += 1
            else:
                latency_buckets[">1s"] += 1
        
        total_latency_samples = len(all_latencies)
        latency_distribution = []
        for label, count in latency_buckets.items():
            percentage = (count / total_latency_samples * 100) if total_latency_samples > 0 else 0
            latency_distribution.append(LatencyDistributionBucket(
                label=label,
                count=count,
                percentage=round(percentage, 1)
            ))
        
        # 3. Error Rate Over Time
        error_rate_over_time = []
        for date in sorted(daily_data.keys()):
            day_data = daily_data[date]
            total = day_data['total_requests']
            failed = day_data['failed_requests']
            error_rate = (failed / total * 100) if total > 0 else 0
            
            error_rate_over_time.append(ErrorRateTimeSeriesPoint(
                date=date,
                error_rate=round(error_rate, 2),
                total_requests=total,
                failed_requests=failed
            ))
        
        # 4. Recent Errors (top 10 most recent)
        recent_errors.sort(key=lambda x: x["timestamp"], reverse=True)
        top_recent_errors = []
        for i, error in enumerate(recent_errors[:10]):
            error_id = f"err_{tenantId}_{i+1}"
            
            top_recent_errors.append(RecentError(
                error_id=error_id,
                model=error["model"],
                provider=error["provider"],
                error_type=error["error_type"],
                timestamp=error["timestamp"],
                input_preview=error["input_preview"],
                latency=round(error["latency"], 2)
            ))
        
        return PerformanceAnalyticsModel(
            latency_over_time=latency_over_time,
            latency_distribution=latency_distribution,
            error_rate_over_time=error_rate_over_time,
            recent_errors=top_recent_errors
        ).dict()
    
    @staticmethod
    async def get_complete_dashboard_metrics(tenantId: str, days: int = 7) -> dict:
        """
        Get comprehensive dashboard metrics that match the DashboardMetricsResponse interface.
        This is a temporary endpoint to support quick frontend integration.
        """
        from .models import (
            CompleteDashboardResponse, DashboardSeries, TimeSeriesDataPoint, ModelStatsPoint,
            DashboardTotals, DashboardDistributions, DistributionStats, DashboardTrends, 
            PeriodTrends, DashboardLeaders, TopCostModel, SlowestModel, ErrorModel, RawSampleItem
        )
        
        # Get raw data for current and previous periods
        current_items = await TenantStatsHandler.get_raw_data(tenantId, days)
        previous_items = await TenantStatsHandler.get_raw_data_with_offset(tenantId, days, days)
        
        if not current_items:
            # Return empty structure if no data
            return CompleteDashboardResponse(
                series=DashboardSeries(by_time=[], by_model=[], by_backend=[], by_deployment=[]),
                totals=DashboardTotals(
                    request_count=0, total_input_tokens=0, total_output_tokens=0,
                    success_rate=0, avg_latency_s=0, p95_latency_s=0
                ),
                distributions=DashboardDistributions(
                    latency_s=DistributionStats(p50=0, p90=0, p95=0, p99=0, mean=0, std=0),
                    ttft_s=DistributionStats(p50=0, p90=0, p95=0, p99=0, mean=0, std=0),
                    input_tokens=DistributionStats(p50=0, p90=0, p95=0, p99=0, mean=0, std=0),
                    output_tokens=DistributionStats(p50=0, p90=0, p95=0, p99=0, mean=0, std=0),
                    cost_per_request_usd=DistributionStats(p50=0, p90=0, p95=0, p99=0, mean=0, std=0)
                ),
                trends=DashboardTrends(
                    last_7d=PeriodTrends(requests=0, p95_latency_s=0, error_rate=0),
                    prev_7d=PeriodTrends(requests=0, p95_latency_s=0, error_rate=0),
                    pct_change={"requests": None, "cost_usd": None, "p95_latency_s": None, "error_rate": None}
                ),
                leaders=DashboardLeaders(
                    top_cost_models=[], slowest_models_p95_latency=[], most_errors_models=[]
                ),
                insights=[]
            ).dict()
        
        # 1. Build Series Data
        
        # Group by time (daily)
        daily_data = defaultdict(lambda: {
            'requests': [], 'latencies': [], 'ttfts': [], 'costs': [],
            'input_tokens': [], 'output_tokens': [], 'errors': 0
        })
        
        # Group by model
        model_data = defaultdict(lambda: {
            'requests': [], 'latencies': [], 'ttfts': [], 'costs': [],
            'input_tokens': [], 'output_tokens': [], 'errors': 0
        })
        
        all_latencies = []
        all_ttfts = []
        all_input_tokens = []
        all_output_tokens = []
        all_costs = []
        total_requests = len(current_items)
        total_errors = 0
        total_cost = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Process each item
        for item in current_items:
            creation_date = item.get("CreationDate", "")
            date_key = creation_date.split("T")[0] if "T" in creation_date else creation_date
            model = item.get("Model", "unknown")
            
            # Extract metrics
            latency = float(item.get("Latency", 0)) if item.get("Latency") else 0
            ttft = float(item.get("TTFT", 0)) if item.get("TTFT") else 0
            cost = float(item.get("Cost", 0)) if item.get("Cost") else 0
            total_tokens = int(item.get("TotalTokens", 0)) if item.get("TotalTokens") else 0
            # Approximate input/output token split
            input_tokens = total_tokens // 2
            output_tokens = total_tokens - input_tokens
            success = item.get("Success", True)
            
            # Aggregate for daily stats
            daily_data[date_key]['requests'].append(item)
            if latency > 0:
                daily_data[date_key]['latencies'].append(latency)
                all_latencies.append(latency)
            if ttft > 0:
                daily_data[date_key]['ttfts'].append(ttft)
                all_ttfts.append(ttft)
            if cost > 0:
                daily_data[date_key]['costs'].append(cost)
                all_costs.append(cost)
                total_cost += cost
            if input_tokens > 0:
                daily_data[date_key]['input_tokens'].append(input_tokens)
                all_input_tokens.append(input_tokens)
                total_input_tokens += input_tokens
            if output_tokens > 0:
                daily_data[date_key]['output_tokens'].append(output_tokens)
                all_output_tokens.append(output_tokens)
                total_output_tokens += output_tokens
            if not success:
                daily_data[date_key]['errors'] += 1
                total_errors += 1
            
            # Aggregate for model stats  
            model_data[model]['requests'].append(item)
            if latency > 0:
                model_data[model]['latencies'].append(latency)
            if ttft > 0:
                model_data[model]['ttfts'].append(ttft)
            if cost > 0:
                model_data[model]['costs'].append(cost)
            if input_tokens > 0:
                model_data[model]['input_tokens'].append(input_tokens)
            if output_tokens > 0:
                model_data[model]['output_tokens'].append(output_tokens)
            if not success:
                model_data[model]['errors'] += 1
        
        # Build time series
        by_time = []
        for date in sorted(daily_data.keys()):
            data = daily_data[date]
            request_count = len(data['requests'])
            
            latencies = data['latencies']
            p95_lat = TenantStatsHandler._percentile(latencies, 95) / 1000 if latencies else 0  # Convert to seconds
            avg_lat = statistics.mean(latencies) / 1000 if latencies else 0  # Convert to seconds
            
            error_rate = data['errors'] / request_count if request_count > 0 else 0
            
            by_time.append(TimeSeriesDataPoint(
                time=date,
                request_count=request_count,
                total_input_tokens=sum(data['input_tokens']),
                total_output_tokens=sum(data['output_tokens']),
                avg_latency_s=round(avg_lat, 3),
                p95_latency_s=round(p95_lat, 3),
                error_rate=round(error_rate, 3)
            ))
        
        # Build model series
        by_model = []
        for model_id, data in model_data.items():
            request_count = len(data['requests'])
            
            latencies = data['latencies']
            p95_lat = TenantStatsHandler._percentile(latencies, 95) / 1000 if latencies else 0
            avg_lat = statistics.mean(latencies) / 1000 if latencies else 0
            
            error_rate = data['errors'] / request_count if request_count > 0 else 0
            
            by_model.append(ModelStatsPoint(
                model_id=model_id,
                request_count=request_count,
                total_input_tokens=sum(data['input_tokens']),
                total_output_tokens=sum(data['output_tokens']),
                avg_latency_s=round(avg_lat, 3),
                p95_latency_s=round(p95_lat, 3),
                error_rate=round(error_rate, 3)
            ))
        
        # 2. Build Totals
        success_rate = (total_requests - total_errors) / total_requests if total_requests > 0 else 0
        avg_latency_s = statistics.mean(all_latencies) / 1000 if all_latencies else 0
        p95_latency_s = TenantStatsHandler._percentile(all_latencies, 95) / 1000 if all_latencies else 0
        avg_cost_per_1k_tokens = (total_cost / (total_input_tokens + total_output_tokens) * 1000) if (total_input_tokens + total_output_tokens) > 0 else None
        
        totals = DashboardTotals(
            request_count=total_requests,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=round(total_cost, 6) if total_cost > 0 else None,
            success_rate=round(success_rate, 3),
            avg_latency_s=round(avg_latency_s, 3),
            p95_latency_s=round(p95_latency_s, 3),
            avg_cost_per_1k_tokens_usd=round(avg_cost_per_1k_tokens, 6) if avg_cost_per_1k_tokens else None,
            streaming_share=None  # Not implemented
        )
        
        # 3. Build Distributions
        def _calculate_distribution_stats(values: list) -> DistributionStats:
            if not values:
                return DistributionStats(p50=0, p90=0, p95=0, p99=0, mean=0, std=0)
            
            sorted_vals = sorted(values)
            return DistributionStats(
                p50=TenantStatsHandler._percentile(sorted_vals, 50),
                p90=TenantStatsHandler._percentile(sorted_vals, 90),
                p95=TenantStatsHandler._percentile(sorted_vals, 95),
                p99=TenantStatsHandler._percentile(sorted_vals, 99),
                mean=round(statistics.mean(sorted_vals), 3),
                std=round(statistics.stdev(sorted_vals), 3) if len(sorted_vals) > 1 else 0
            )
        
        # Convert latencies to seconds for distributions
        latency_seconds = [l / 1000 for l in all_latencies]
        ttft_seconds = [t / 1000 for t in all_ttfts]
        
        distributions = DashboardDistributions(
            latency_s=_calculate_distribution_stats(latency_seconds),
            ttft_s=_calculate_distribution_stats(ttft_seconds),
            input_tokens=_calculate_distribution_stats(all_input_tokens),
            output_tokens=_calculate_distribution_stats(all_output_tokens),
            cost_per_request_usd=_calculate_distribution_stats(all_costs)
        )
        
        # 4. Build Trends (current vs previous period)
        current_metrics = TenantStatsHandler._calculate_period_metrics(current_items)
        previous_metrics = TenantStatsHandler._calculate_period_metrics(previous_items)
        
        def _calculate_pct_change(current: float, previous: float) -> Optional[float]:
            if previous == 0:
                return None
            return round(((current - previous) / previous) * 100, 1)
        
        trends = DashboardTrends(
            last_7d=PeriodTrends(
                requests=current_metrics["total_requests"],
                cost_usd=round(current_metrics["total_cost"], 6) if current_metrics["total_cost"] > 0 else None,
                p95_latency_s=round(current_metrics["avg_latency_p95"] / 1000, 3),
                error_rate=round(current_metrics["error_rate"] / 100, 3)
            ),
            prev_7d=PeriodTrends(
                requests=previous_metrics["total_requests"],
                cost_usd=round(previous_metrics["total_cost"], 6) if previous_metrics["total_cost"] > 0 else None,
                p95_latency_s=round(previous_metrics["avg_latency_p95"] / 1000, 3),
                error_rate=round(previous_metrics["error_rate"] / 100, 3)
            ),
            pct_change={
                "requests": _calculate_pct_change(current_metrics["total_requests"], previous_metrics["total_requests"]),
                "cost_usd": _calculate_pct_change(current_metrics["total_cost"], previous_metrics["total_cost"]),
                "p95_latency_s": _calculate_pct_change(current_metrics["avg_latency_p95"], previous_metrics["avg_latency_p95"]),
                "error_rate": _calculate_pct_change(current_metrics["error_rate"], previous_metrics["error_rate"])
            }
        )
        
        # 5. Build Leaders
        # Top cost models
        model_costs = defaultdict(lambda: {"cost": 0, "count": 0})
        model_latencies = defaultdict(list)
        model_errors = defaultdict(int)
        
        for model_id, data in model_data.items():
            model_costs[model_id]["cost"] = sum(data['costs'])
            model_costs[model_id]["count"] = len(data['requests'])
            model_latencies[model_id] = data['latencies']
            model_errors[model_id] = data['errors']
        
        top_cost_models = [
            TopCostModel(
                model_id=model_id,
                cost_usd=round(data["cost"], 6),
                count=data["count"]
            )
            for model_id, data in sorted(model_costs.items(), key=lambda x: x[1]["cost"], reverse=True)[:5]
            if data["cost"] > 0
        ]
        
        slowest_models = [
            SlowestModel(
                model_id=model_id,
                p95_latency_s=round(TenantStatsHandler._percentile(latencies, 95) / 1000, 3),
                count=len(latencies)
            )
            for model_id, latencies in sorted(
                model_latencies.items(), 
                key=lambda x: TenantStatsHandler._percentile(x[1], 95) if x[1] else 0, 
                reverse=True
            )[:5]
            if latencies
        ]
        
        most_errors_models = [
            ErrorModel(
                model_id=model_id,
                error_count=error_count
            )
            for model_id, error_count in sorted(model_errors.items(), key=lambda x: x[1], reverse=True)[:5]
            if error_count > 0
        ]
        
        leaders = DashboardLeaders(
            top_cost_models=top_cost_models,
            slowest_models_p95_latency=slowest_models,
            most_errors_models=most_errors_models
        )
        
        # 6. Build Raw Sample (most recent 10 items, sorted by CreationDate descending)
        sorted_items_dashboard = sorted(
            current_items,
            key=lambda x: x.get("CreationDate", ""),
            reverse=True
        )
        raw_sample = []
        for i, item in enumerate(sorted_items_dashboard[:10]):
            input_text = item.get("InputText", "")
            output_text = item.get("OutputText", "")
            total_tokens = int(item.get("TotalTokens", 0)) if item.get("TotalTokens") else 0
            
            raw_sample.append(RawSampleItem(
                timestamp=item.get("CreationDate", ""),
                model=item.get("Model", "unknown"),
                provider=item.get("Provider", "unknown"),
                success=item.get("Success", True),
                latency_s=round(float(item.get("Latency", 0)) / 1000, 3) if item.get("Latency") else 0,
                input_tokens=total_tokens // 2,
                output_tokens=total_tokens - (total_tokens // 2),
                cost_usd=round(float(item.get("Cost", 0)), 6) if item.get("Cost") else None,
                input_preview=input_text,
                output_preview=output_text
            ))
        
        # Build final response
        response = CompleteDashboardResponse(
            series=DashboardSeries(
                by_time=by_time,
                by_model=by_model,
                by_backend=[],  # Not implemented
                by_deployment=[]  # Not implemented
            ),
            totals=totals,
            distributions=distributions,
            trends=trends,
            leaders=leaders,
            insights=[],  # Not implemented
            raw_sample=raw_sample
        )
        
        return response.dict()
    
    @staticmethod
    async def get_analytics_metrics(tenantId: str, days: int = 30, trace_limit: int = 20, trace_offset: int = 0, filters: "TraceFilters" = None) -> dict:
        """
        Get comprehensive analytics metrics that match the AnalyticsMetricsResponse interface.

        Args:
            tenantId: Tenant identifier
            days: Number of days to include in analytics
            trace_limit: Number of traces to return in raw_sample (0 for all)
            trace_offset: Offset for trace pagination
            filters: Optional filters to apply to traces (model_id, backend, is_success, search)
        """
        from .models import (
            AnalyticsMetricsResponse, AnalyticsSeries, AnalyticsTimeSeriesData,
            AnalyticsModelSeriesData, AnalyticsBackendSeriesData, AnalyticsTotals,
            AnalyticsDistributions, AnalyticsDistributionData, AnalyticsTrends,
            AnalyticsTrendData, AnalyticsLeaders, AnalyticsCostLeaderData,
            AnalyticsLatencyLeaderData, AnalyticsErrorLeaderData, AnalyticsInsightData,
            AnalyticsRawSampleData, TraceFilters
        )

        # Initialize filters if not provided
        if filters is None:
            filters = TraceFilters()

        # Get raw data for current and previous periods
        current_items = await TenantStatsHandler.get_raw_data(tenantId, days)
        previous_items = await TenantStatsHandler.get_raw_data_with_offset(tenantId, 7, days)  # Previous 7 days for trends

        if not current_items:
            # Return empty structure if no data
            return AnalyticsMetricsResponse(
                totals=AnalyticsTotals(
                    request_count=0, total_input_tokens=0, total_output_tokens=0,
                    total_cost_usd=0.0, success_rate=0.0, avg_latency_s=0.0,
                    p95_latency_s=0.0, avg_cost_per_1k_tokens_usd=0.0, streaming_share=0.0
                ),
                series=AnalyticsSeries(by_time=[], by_model=[], by_backend=[]),
                distributions=AnalyticsDistributions(
                    latency_s=AnalyticsDistributionData(p50=0, p90=0, p95=0, p99=0, mean=0, std=0),
                    ttft_s=AnalyticsDistributionData(p50=0, p90=0, p95=0, p99=0, mean=0, std=0),
                    input_tokens=AnalyticsDistributionData(p50=0, p90=0, p95=0, p99=0, mean=0, std=0),
                    output_tokens=AnalyticsDistributionData(p50=0, p90=0, p95=0, p99=0, mean=0, std=0),
                    cost_per_request_usd=AnalyticsDistributionData(p50=0, p90=0, p95=0, p99=0, mean=0, std=0)
                ),
                trends=AnalyticsTrends(
                    last_7d=AnalyticsTrendData(requests=0, cost_usd=0.0, p95_latency_s=0.0, error_rate=0.0),
                    prev_7d=AnalyticsTrendData(requests=0, cost_usd=0.0, p95_latency_s=0.0, error_rate=0.0),
                    pct_change={"requests": None, "cost_usd": None, "p95_latency_s": None, "error_rate": None}
                ),
                leaders=AnalyticsLeaders(
                    top_cost_models=[], slowest_models_p95_latency=[], most_errors_models=[]
                ),
                insights=[],
                total_traces=0
            ).dict()
        
        # Process data for analytics
        daily_data = defaultdict(lambda: {
            'requests': [], 'latencies': [], 'ttfts': [], 'costs': [],
            'input_tokens': [], 'output_tokens': [], 'errors': 0
        })
        
        model_data = defaultdict(lambda: {
            'requests': [], 'latencies': [], 'ttfts': [], 'costs': [],
            'input_tokens': [], 'output_tokens': [], 'errors': 0
        })
        
        backend_data = defaultdict(lambda: {
            'requests': [], 'latencies': [], 'ttfts': [], 'costs': [],
            'input_tokens': [], 'output_tokens': [], 'errors': 0
        })
        
        # Aggregate data
        all_latencies = []
        all_ttfts = []
        all_input_tokens = []
        all_output_tokens = []
        all_costs = []
        total_requests = len(current_items)
        total_errors = 0
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for item in current_items:
            creation_date = item.get("CreationDate", "")
            date_key = creation_date.split("T")[0] if "T" in creation_date else creation_date
            model = item.get("Model", "unknown")
            backend = item.get("Provider", "unknown")  # Using Provider as backend
            
            # Extract metrics
            latency = float(item.get("Latency", 0)) if item.get("Latency") else 0
            ttft = float(item.get("TTFT", 0)) if item.get("TTFT") else 0
            cost = float(item.get("Cost", 0)) if item.get("Cost") else 0
            total_tokens = int(item.get("TotalTokens", 0)) if item.get("TotalTokens") else 0
            input_tokens = total_tokens // 2
            output_tokens = total_tokens - input_tokens
            success = item.get("Success", True)
            
            # Aggregate for all collections
            for data_dict in [daily_data[date_key], model_data[model], backend_data[backend]]:
                data_dict['requests'].append(item)
                if latency > 0:
                    data_dict['latencies'].append(latency)
                if ttft > 0:
                    data_dict['ttfts'].append(ttft)
                if cost > 0:
                    data_dict['costs'].append(cost)
                if input_tokens > 0:
                    data_dict['input_tokens'].append(input_tokens)
                if output_tokens > 0:
                    data_dict['output_tokens'].append(output_tokens)
                if not success:
                    data_dict['errors'] += 1
            
            # Global aggregations
            if latency > 0:
                all_latencies.append(latency)
            if ttft > 0:
                all_ttfts.append(ttft)
            if cost > 0:
                all_costs.append(cost)
                total_cost += cost
            if input_tokens > 0:
                all_input_tokens.append(input_tokens)
                total_input_tokens += input_tokens
            if output_tokens > 0:
                all_output_tokens.append(output_tokens)
                total_output_tokens += output_tokens
            if not success:
                total_errors += 1
        
        # Helper function to build series data
        def _build_series_data(data_dict, id_field):
            result = []
            for key, data in data_dict.items():
                request_count = len(data['requests'])
                latencies = data['latencies']
                costs = data['costs']
                
                avg_lat = statistics.mean(latencies) / 1000 if latencies else 0  # Convert to seconds
                p95_lat = TenantStatsHandler._percentile(latencies, 95) / 1000 if latencies else 0
                error_rate = data['errors'] / request_count if request_count > 0 else 0
                total_cost_usd = sum(costs)
                
                series_data = {
                    id_field: key,
                    'request_count': request_count,
                    'total_input_tokens': sum(data['input_tokens']),
                    'total_output_tokens': sum(data['output_tokens']),
                    'avg_latency_s': round(avg_lat, 3),
                    'p95_latency_s': round(p95_lat, 3),
                    'error_rate': round(error_rate, 3),
                    'total_cost_usd': round(total_cost_usd, 6)
                }
                
                if id_field == 'time':
                    result.append(AnalyticsTimeSeriesData(**series_data))
                elif id_field == 'model_id':
                    result.append(AnalyticsModelSeriesData(**series_data))
                elif id_field == 'backend':
                    result.append(AnalyticsBackendSeriesData(**series_data))
            
            return result
        
        # Build series data
        by_time = _build_series_data({k: v for k, v in sorted(daily_data.items())}, 'time')
        by_model = _build_series_data(model_data, 'model_id')
        by_backend = _build_series_data(backend_data, 'backend')
        
        # Build totals
        success_rate = (total_requests - total_errors) / total_requests if total_requests > 0 else 0
        avg_latency_s = statistics.mean(all_latencies) / 1000 if all_latencies else 0
        p95_latency_s = TenantStatsHandler._percentile(all_latencies, 95) / 1000 if all_latencies else 0
        avg_cost_per_1k_tokens = (total_cost / (total_input_tokens + total_output_tokens) * 1000) if (total_input_tokens + total_output_tokens) > 0 else 0
        
        totals = AnalyticsTotals(
            request_count=total_requests,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=round(total_cost, 6),
            success_rate=round(success_rate, 3),
            avg_latency_s=round(avg_latency_s, 3),
            p95_latency_s=round(p95_latency_s, 3),
            avg_cost_per_1k_tokens_usd=round(avg_cost_per_1k_tokens, 6),
            streaming_share=0.0  # Not implemented yet
        )
        
        # Build distributions
        def _build_distribution(values: list) -> AnalyticsDistributionData:
            if not values:
                return AnalyticsDistributionData(p50=0, p90=0, p95=0, p99=0, mean=0, std=0)
            
            sorted_vals = sorted(values)
            return AnalyticsDistributionData(
                p50=TenantStatsHandler._percentile(sorted_vals, 50),
                p90=TenantStatsHandler._percentile(sorted_vals, 90),
                p95=TenantStatsHandler._percentile(sorted_vals, 95),
                p99=TenantStatsHandler._percentile(sorted_vals, 99),
                mean=round(statistics.mean(sorted_vals), 3),
                std=round(statistics.stdev(sorted_vals), 3) if len(sorted_vals) > 1 else 0
            )
        
        latency_seconds = [l / 1000 for l in all_latencies]
        ttft_seconds = [t / 1000 for t in all_ttfts]
        
        distributions = AnalyticsDistributions(
            latency_s=_build_distribution(latency_seconds),
            ttft_s=_build_distribution(ttft_seconds),
            input_tokens=_build_distribution(all_input_tokens),
            output_tokens=_build_distribution(all_output_tokens),
            cost_per_request_usd=_build_distribution(all_costs)
        )
        
        # Build trends (current vs previous 7 days)
        current_7d = await TenantStatsHandler.get_raw_data(tenantId, 7)
        current_metrics = TenantStatsHandler._calculate_period_metrics(current_7d)
        previous_metrics = TenantStatsHandler._calculate_period_metrics(previous_items)
        
        def _calculate_pct_change(current: float, previous: float) -> Optional[float]:
            if previous == 0:
                return None
            return round(((current - previous) / previous) * 100, 1)
        
        trends = AnalyticsTrends(
            last_7d=AnalyticsTrendData(
                requests=current_metrics["total_requests"],
                cost_usd=round(current_metrics["total_cost"], 6),
                p95_latency_s=round(current_metrics["avg_latency_p95"] / 1000, 3),
                error_rate=round(current_metrics["error_rate"] / 100, 3)
            ),
            prev_7d=AnalyticsTrendData(
                requests=previous_metrics["total_requests"],
                cost_usd=round(previous_metrics["total_cost"], 6),
                p95_latency_s=round(previous_metrics["avg_latency_p95"] / 1000, 3),
                error_rate=round(previous_metrics["error_rate"] / 100, 3)
            ),
            pct_change={
                "requests": _calculate_pct_change(current_metrics["total_requests"], previous_metrics["total_requests"]),
                "cost_usd": _calculate_pct_change(current_metrics["total_cost"], previous_metrics["total_cost"]),
                "p95_latency_s": _calculate_pct_change(current_metrics["avg_latency_p95"], previous_metrics["avg_latency_p95"]),
                "error_rate": _calculate_pct_change(current_metrics["error_rate"], previous_metrics["error_rate"])
            }
        )
        
        # Build leaders
        model_costs = defaultdict(lambda: {"cost": 0.0, "count": 0})
        model_latencies = defaultdict(list)
        model_errors = defaultdict(lambda: {"errors": 0, "total": 0})
        
        for model_id, data in model_data.items():
            model_costs[model_id]["cost"] = sum(data['costs'])
            model_costs[model_id]["count"] = len(data['requests'])
            model_latencies[model_id] = data['latencies']
            model_errors[model_id]["errors"] = data['errors']
            model_errors[model_id]["total"] = len(data['requests'])
        
        top_cost_models = [
            AnalyticsCostLeaderData(
                model_id=model_id,
                total_cost_usd=round(data["cost"], 6),
                request_count=data["count"]
            )
            for model_id, data in sorted(model_costs.items(), key=lambda x: x[1]["cost"], reverse=True)[:5]
            if data["cost"] > 0
        ]
        
        slowest_models = [
            AnalyticsLatencyLeaderData(
                model_id=model_id,
                p95_latency_s=round(TenantStatsHandler._percentile(latencies, 95) / 1000, 3),
                count=len(latencies)
            )
            for model_id, latencies in sorted(
                model_latencies.items(),
                key=lambda x: TenantStatsHandler._percentile(x[1], 95) if x[1] else 0,
                reverse=True
            )[:5]
            if latencies
        ]
        
        most_errors_models = [
            AnalyticsErrorLeaderData(
                model_id=model_id,
                error_count=data["errors"],
                total_requests=data["total"],
                error_rate=round(data["errors"] / data["total"], 3) if data["total"] > 0 else 0
            )
            for model_id, data in sorted(model_errors.items(), key=lambda x: x[1]["errors"], reverse=True)[:5]
            if data["errors"] > 0
        ]
        
        leaders = AnalyticsLeaders(
            top_cost_models=top_cost_models,
            slowest_models_p95_latency=slowest_models,
            most_errors_models=most_errors_models
        )
        
        # Build raw sample with filtering and pagination
        # Sort all items by CreationDate descending
        sorted_items = sorted(
            current_items,
            key=lambda x: x.get("CreationDate", ""),
            reverse=True
        )

        # Apply filters if any are active
        if filters.has_filters():
            filtered_items = []
            search_lower = filters.search.lower() if filters.search else None

            for item in sorted_items:
                # Filter by model_id
                if filters.model_id and item.get("Model") != filters.model_id:
                    continue

                # Filter by backend/provider
                if filters.backend and item.get("Provider") != filters.backend:
                    continue

                # Filter by success status
                if filters.is_success is not None:
                    item_success = item.get("Success", True)
                    if filters.is_success != item_success:
                        continue

                # Filter by search text in input/output
                if search_lower:
                    input_text = (item.get("InputText") or "").lower()
                    output_text = (item.get("OutputText") or "").lower()
                    if search_lower not in input_text and search_lower not in output_text:
                        continue

                filtered_items.append(item)

            sorted_items = filtered_items

        total_traces = len(sorted_items)

        # Apply pagination: offset first, then limit
        # If trace_limit is 0, return all traces (no limit)
        if trace_limit == 0:
            paginated_items = sorted_items[trace_offset:]
        else:
            paginated_items = sorted_items[trace_offset:trace_offset + trace_limit]

        raw_sample = []
        for i, item in enumerate(paginated_items):
            input_text = item.get("InputText", "")
            output_text = item.get("OutputText", "")
            history_text = item.get("History", None)
            total_tokens = int(item.get("TotalTokens", 0)) if item.get("TotalTokens") else 0
            cost = float(item.get("Cost", 0)) if item.get("Cost") else 0
            latency = float(item.get("Latency", 0)) if item.get("Latency") else 0
            ttft = float(item.get("TTFT", 0)) if item.get("TTFT") else 0

            # Use absolute index for event_id (offset + i)
            absolute_index = trace_offset + i + 1

            raw_sample.append(AnalyticsRawSampleData(
                event_id=f"evt_{tenantId}_{absolute_index}",
                id=f"evt_{tenantId}_{absolute_index}",
                model_id=item.get("Model", "unknown"),
                created_at=item.get("CreationDate", ""),
                latency_s=round(latency / 1000, 3),
                ttft_s=round(ttft / 1000, 3),
                input_tokens=total_tokens // 2,
                output_tokens=total_tokens - (total_tokens // 2),
                total_cost_usd=round(cost, 6),
                cost_usd=round(cost, 6),
                is_success=item.get("Success", True),
                success=item.get("Success", True),
                error_code=item.get("ErrorType") if not item.get("Success", True) else None,
                backend=item.get("Provider", "unknown"),
                deployment_id=None,
                is_stream=False,  # Not implemented yet
                input_preview=input_text,
                output_preview=output_text,
                input_messages=None,
                output_text=output_text,
                endpoint="/v1/chat/completions",
                history=history_text
            ))

        # Build final response
        response = AnalyticsMetricsResponse(
            totals=totals,
            series=AnalyticsSeries(
                by_time=by_time,
                by_model=by_model,
                by_backend=by_backend
            ),
            distributions=distributions,
            trends=trends,
            leaders=leaders,
            insights=[],  # Not implemented yet
            raw_sample=raw_sample,
            total_traces=total_traces
        )

        return response.dict()