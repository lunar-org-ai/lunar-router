import aioboto3
from boto3.dynamodb.conditions import Attr, Key
from datetime import datetime
from .models import TenantStatsModel
from typing import Dict
from datetime import timedelta
from .StatsHandlerBase import StatsHandlerBase 
from .models import StatsResponseModel

class GlobalStatsHandler(StatsHandlerBase):
    table_name: str = "GlobalStats"

    @staticmethod
    async def insert(stats: TenantStatsModel):
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=GlobalStatsHandler.region) as client:

            provider = stats.Provider or "Unknown"
            model = stats.Model or "Unknown"
            date_str = stats.CreationDate.split('T')[0] if 'T' in stats.CreationDate else stats.CreationDate
            model_date = f"{model}#{date_str}"

            # Try to get existing item
            try:
                response = await client.get_item(
                    TableName=GlobalStatsHandler.table_name,
                    Key={
                        'Provider': {'S': provider},
                        'ModelDate': {'S': model_date}
                    }
                )
                if 'Item' in response:
                    existing_item = GlobalStatsHandler.deserialize_item(response['Item'])

                    success_count = existing_item.get('SuccessCount', 0)
                    error_count = existing_item.get('ErrorCount', 0)
                    error_types = existing_item.get('ErrorTypes', {})
                    
                    if stats.Success:
                        success_count += 1
                    else:
                        error_count += 1
                        if stats.ErrorType:
                            error_types[stats.ErrorType] = error_types.get(stats.ErrorType, 0) + 1
                    
                    # Calculate new means
                    total_requests = success_count + error_count
                    existing_mean_ttft = existing_item.get('MeanTTFT', 0)
                    existing_mean_latency = existing_item.get('MeanLatency', 0)
                    
                    new_mean_ttft = existing_mean_ttft
                    new_mean_latency = existing_mean_latency
                    
                    if stats.TTFT is not None:
                        new_mean_ttft = ((existing_mean_ttft * (total_requests - 1)) + stats.TTFT) / total_requests
                    
                    if stats.Latency is not None:
                        new_mean_latency = ((existing_mean_latency * (total_requests - 1)) + stats.Latency) / total_requests
                    
                    # Update item
                    await client.put_item(
                        TableName=GlobalStatsHandler.table_name,
                        Item={
                            'Provider': {'S': provider},
                            'ModelDate': {'S': model_date},
                            'Cost': {'N': str(existing_item.get('Cost', 0))},
                            'MeanTTFT': {'N': str(new_mean_ttft)},
                            'MeanLatency': {'N': str(new_mean_latency)},
                            'SuccessCount': {'N': str(success_count)},
                            'ErrorCount': {'N': str(error_count)},
                            'ErrorTypes': {'M': {k: {'N': str(v)} for k, v in error_types.items()}},
                            'UpdatedAt': {'S': datetime.utcnow().isoformat()}
                        }
                    )
                else:
                    # Create new item
                    error_types = {}
                    if not stats.Success and stats.ErrorType:
                        error_types[stats.ErrorType] = 1
                    
                    await client.put_item(
                        TableName=GlobalStatsHandler.table_name,
                        Item={
                            'Provider': {'S': provider},
                            'ModelDate': {'S': model_date},
                            'Cost': {'N': str(stats.Cost or 0)},
                            'MeanTTFT': {'N': str(stats.TTFT or 0)},
                            'MeanLatency': {'N': str(stats.Latency or 0)},
                            'SuccessCount': {'N': str(1 if stats.Success else 0)},
                            'ErrorCount': {'N': str(0 if stats.Success else 1)},
                            'ErrorTypes': {'M': {k: {'N': str(v)} for k, v in error_types.items()}},
                            'UpdatedAt': {'S': datetime.utcnow().isoformat()}
                        }
                    )
                    
            except Exception as e:
                print(f"Error updating global stats: {e}")

    @staticmethod
    async def summary(model: str, provider: str, days: int = 15) -> StatsResponseModel:
        """
        Return aggregated metrics from the GlobalStats table for a given model + provider.
        Uses the pre-computed averages (MeanLatency, MeanTTFT) and counters (SuccessCount, ErrorCount).
        """
        today = datetime.utcnow().date()
        start_date = today - timedelta(days=days)

        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=GlobalStatsHandler.region) as client:
            response = await client.query(
                TableName=GlobalStatsHandler.table_name,
                KeyConditionExpression="Provider = :provider AND begins_with(ModelDate, :model)",
                ExpressionAttributeValues={
                    ":provider": {"S": provider},
                    ":model": {"S": f"{model}#"},
                },
            )
            raw_items = response.get("Items", [])
            items = [GlobalStatsHandler.deserialize_item(item) for item in raw_items]

            filtered = []
            for i in items:
                try:
                    updated_at_date = datetime.fromisoformat(i.get("UpdatedAt")).date()
                    if start_date <= updated_at_date <= today:
                        filtered.append(i)
                except Exception:
                    continue

            if not filtered:
                return StatsResponseModel(
                    p50_lat=float("inf"),
                    p50_ttft=float("inf"),
                    err_rate=0.0,
                    n=0,
                    updated_at=0.0,
                    updated_at_date="1970-01-01"
                )

            latencies = [float(i["MeanLatency"]) for i in filtered if "MeanLatency" in i]
            ttfts = [float(i["MeanTTFT"]) for i in filtered if "MeanTTFT" in i]
            errors = sum(int(i.get("ErrorCount", 0)) for i in filtered)
            successes = sum(int(i.get("SuccessCount", 0)) for i in filtered)
            n = errors + successes

            err_rate = errors / float(n) if n > 0 else 0.0

            last_update_str = max(i.get("UpdatedAt", "1970-01-01T00:00:00") for i in filtered)
            try:
                updated_at = datetime.fromisoformat(last_update_str).timestamp()
                updated_at_date = last_update_str  # mantém string ISO original
            except Exception:
                updated_at = 0.0
                updated_at_date = "1970-01-01"

            return StatsResponseModel(
                p50_lat=StatsHandlerBase._percentile(latencies, 50),
                p50_ttft=StatsHandlerBase._percentile(ttfts, 50),
                err_rate=err_rate,
                n=n,
                updated_at=updated_at,
                updated_at_date=updated_at_date
            )
    
    @staticmethod
    async def all_for_model(model: str, days: int = 15) -> Dict[str, StatsResponseModel]:
        """
        Return summaries for all providers of a given model from the GlobalStats table.
        """
        today = datetime.utcnow().date()
        start_date = today - timedelta(days=days)

        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=GlobalStatsHandler.region) as client:
            response = await client.scan(
                TableName=GlobalStatsHandler.table_name,
                FilterExpression="begins_with(ModelDate, :model)",
                ExpressionAttributeValues={
                    ":model": {"S": model},
                },
            )

            raw_items = response.get("Items", [])
            items = [GlobalStatsHandler.deserialize_item(item) for item in raw_items]

            by_model: Dict[str, list[dict]] = {}
            for i in items:
                md = i.get("ModelDate", "")
                parts = md.split("#", 1)
                if len(parts) != 2:
                    continue
                model_name = parts[0]
                try:
                    updated_at_date = datetime.fromisoformat(i.get("UpdatedAt")).date()
                except Exception:
                    continue
                if start_date <= updated_at_date <= today:
                    by_model.setdefault(model_name, []).append(i)

            out: Dict[str, StatsResponseModel] = {}
            for model_name, items in by_model.items():
                if not items:
                    out[model_name] = StatsResponseModel(
                        p50_lat=float("inf"),
                        p50_ttft=float("inf"),
                        err_rate=0.0,
                        n=0,
                        updated_at=0.0
                    )
                    continue

                latencies = [float(i["MeanLatency"]) for i in items if "MeanLatency" in i]
                ttfts = [float(i["MeanTTFT"]) for i in items if "MeanTTFT" in i]
                errors = sum(int(i.get("ErrorCount", 0)) for i in items)
                successes = sum(int(i.get("SuccessCount", 0)) for i in items)
                n = errors + successes

                err_rate = errors / float(n) if n > 0 else 0.0
                last_update_str = max(i.get("UpdatedAt", "1970-01-01T00:00:00") for i in items)
                
                try:
                    updated_at = datetime.fromisoformat(last_update_str).timestamp()
                    updated_at_date = last_update_str
                except Exception:
                    updated_at = 0.0
                    updated_at_date = "1970-01-01"

                out[model_name] = StatsResponseModel(
                    p50_lat=StatsHandlerBase._percentile(latencies, 50),
                    p50_ttft=StatsHandlerBase._percentile(ttfts, 50),
                    err_rate=err_rate,
                    n=n,
                    updated_at=updated_at,
                    updated_at_date=updated_at_date
                )

            return out