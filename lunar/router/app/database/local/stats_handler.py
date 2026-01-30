"""
Local JSON-based Stats Handler for development/open-source use.
Replaces DynamoDB with local JSON file storage.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import statistics

from ..models import TenantStatsModel, StatsResponseModel


# Default data directory
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
TENANT_STATS_FILE = DATA_DIR / "tenant_stats.json"
GLOBAL_STATS_FILE = DATA_DIR / "global_stats.json"


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json_file(filepath: Path) -> List[dict]:
    _ensure_data_dir()
    if not filepath.exists():
        filepath.write_text("[]")
        return []
    try:
        return json.loads(filepath.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_json_file(filepath: Path, data: List[dict]):
    _ensure_data_dir()
    filepath.write_text(json.dumps(data, indent=2, default=str))


def _percentile(data: list, p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100.0)
    return sorted_data[min(idx, len(sorted_data) - 1)]


class LocalGlobalStatsHandler:
    """Local JSON-based global stats handler."""

    @staticmethod
    async def insert(stats: TenantStatsModel):
        """Insert or update global stats."""
        data = _load_json_file(GLOBAL_STATS_FILE)

        provider = stats.Provider or "Unknown"
        model = stats.Model or "Unknown"
        date_str = stats.CreationDate.split('T')[0] if 'T' in stats.CreationDate else stats.CreationDate
        model_date = f"{model}#{date_str}"

        # Find existing item
        existing_idx = None
        for i, item in enumerate(data):
            if item.get("Provider") == provider and item.get("ModelDate") == model_date:
                existing_idx = i
                break

        if existing_idx is not None:
            existing_item = data[existing_idx]
            success_count = existing_item.get('SuccessCount', 0)
            error_count = existing_item.get('ErrorCount', 0)
            error_types = existing_item.get('ErrorTypes', {})

            if stats.Success:
                success_count += 1
            else:
                error_count += 1
                if stats.ErrorType:
                    error_types[stats.ErrorType] = error_types.get(stats.ErrorType, 0) + 1

            total_requests = success_count + error_count
            existing_mean_ttft = existing_item.get('MeanTTFT', 0)
            existing_mean_latency = existing_item.get('MeanLatency', 0)

            new_mean_ttft = existing_mean_ttft
            new_mean_latency = existing_mean_latency

            if stats.TTFT is not None:
                new_mean_ttft = ((existing_mean_ttft * (total_requests - 1)) + float(stats.TTFT)) / total_requests

            if stats.Latency is not None:
                new_mean_latency = ((existing_mean_latency * (total_requests - 1)) + float(stats.Latency)) / total_requests

            data[existing_idx] = {
                'Provider': provider,
                'ModelDate': model_date,
                'Cost': existing_item.get('Cost', 0),
                'MeanTTFT': new_mean_ttft,
                'MeanLatency': new_mean_latency,
                'SuccessCount': success_count,
                'ErrorCount': error_count,
                'ErrorTypes': error_types,
                'UpdatedAt': datetime.utcnow().isoformat()
            }
        else:
            error_types = {}
            if not stats.Success and stats.ErrorType:
                error_types[stats.ErrorType] = 1

            data.append({
                'Provider': provider,
                'ModelDate': model_date,
                'Cost': float(stats.Cost) if stats.Cost else 0,
                'MeanTTFT': float(stats.TTFT) if stats.TTFT else 0,
                'MeanLatency': float(stats.Latency) if stats.Latency else 0,
                'SuccessCount': 1 if stats.Success else 0,
                'ErrorCount': 0 if stats.Success else 1,
                'ErrorTypes': error_types,
                'UpdatedAt': datetime.utcnow().isoformat()
            })

        _save_json_file(GLOBAL_STATS_FILE, data)

    @staticmethod
    async def summary(model: str, provider: str, days: int = 15) -> StatsResponseModel:
        """Return aggregated metrics for a given model + provider."""
        today = datetime.utcnow().date()
        start_date = today - timedelta(days=days)

        data = _load_json_file(GLOBAL_STATS_FILE)
        filtered = []

        for item in data:
            if item.get("Provider") != provider:
                continue
            model_date = item.get("ModelDate", "")
            if not model_date.startswith(f"{model}#"):
                continue
            try:
                updated_at_date = datetime.fromisoformat(item.get("UpdatedAt")).date()
                if start_date <= updated_at_date <= today:
                    filtered.append(item)
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
            updated_at_date = last_update_str
        except Exception:
            updated_at = 0.0
            updated_at_date = "1970-01-01"

        return StatsResponseModel(
            p50_lat=_percentile(latencies, 50),
            p50_ttft=_percentile(ttfts, 50),
            err_rate=err_rate,
            n=n,
            updated_at=updated_at,
            updated_at_date=updated_at_date
        )

    @staticmethod
    async def all_for_model(model: str, days: int = 15) -> Dict[str, StatsResponseModel]:
        """Return summaries for all providers of a given model."""
        today = datetime.utcnow().date()
        start_date = today - timedelta(days=days)

        data = _load_json_file(GLOBAL_STATS_FILE)
        by_provider: Dict[str, list] = {}

        for item in data:
            model_date = item.get("ModelDate", "")
            if not model_date.startswith(f"{model}#"):
                continue
            try:
                updated_at_date = datetime.fromisoformat(item.get("UpdatedAt")).date()
                if start_date <= updated_at_date <= today:
                    provider = item.get("Provider", "unknown")
                    by_provider.setdefault(provider, []).append(item)
            except Exception:
                continue

        out: Dict[str, StatsResponseModel] = {}
        for provider, items in by_provider.items():
            if not items:
                out[provider] = StatsResponseModel(
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

            out[provider] = StatsResponseModel(
                p50_lat=_percentile(latencies, 50),
                p50_ttft=_percentile(ttfts, 50),
                err_rate=err_rate,
                n=n,
                updated_at=updated_at,
                updated_at_date=updated_at_date
            )

        return out


class LocalStatsHandler:
    """Local JSON-based tenant stats handler."""

    @staticmethod
    async def insert(stats: TenantStatsModel):
        """Insert tenant stats."""
        data = _load_json_file(TENANT_STATS_FILE)

        item = {
            "TenantId": stats.TenantId,
            "CreationDate": stats.CreationDate,
            "Success": stats.Success,
            "Model": stats.Model,
            "Provider": stats.Provider,
            "ErrorType": stats.ErrorType,
            "InputText": stats.InputText,
            "OutputText": stats.OutputText,
            "ErrorCategory": stats.ErrorCategory,
            "ErrorMessage": stats.ErrorMessage,
            "ProviderAttempts": stats.ProviderAttempts,
            "FinalProvider": stats.FinalProvider,
            "Cost": float(stats.Cost) if stats.Cost else None,
            "Latency": float(stats.Latency) if stats.Latency else None,
            "TTFT": float(stats.TTFT) if stats.TTFT else None,
            "TotalTokens": stats.TotalTokens,
            "FallbackCount": stats.FallbackCount,
        }

        # Remove None values
        item = {k: v for k, v in item.items() if v is not None}

        data.append(item)
        _save_json_file(TENANT_STATS_FILE, data)

        # Also update global stats
        await LocalGlobalStatsHandler.insert(stats)

    @staticmethod
    async def summary(model: str, provider: str, days: int) -> StatsResponseModel:
        today = datetime.utcnow()
        start_date = today - timedelta(days=days)

        start_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_str = today.strftime("%Y-%m-%dT23:59:59")

        data = _load_json_file(TENANT_STATS_FILE)
        items = [
            item for item in data
            if item.get("Model") == model
            and item.get("Provider") == provider
            and start_str <= item.get("CreationDate", "") <= end_str
        ]

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
            p50_lat=_percentile(latencies, 50),
            p50_ttft=_percentile(ttfts, 50),
            err_rate=err_rate,
            n=n,
            updated_at=updated_at,
            updated_at_date=updated_at_date
        )

    @staticmethod
    async def get_raw_data(tenantId: str, days: int = 15) -> List[dict]:
        """Retrieve raw tenant stats data for a given tenantId and time range."""
        today = datetime.utcnow()
        start_date = today - timedelta(days=days)

        start_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_str = today.strftime("%Y-%m-%dT23:59:59")

        data = _load_json_file(TENANT_STATS_FILE)
        items = [
            item for item in data
            if item.get("TenantId") == tenantId
            and start_str <= item.get("CreationDate", "") <= end_str
        ]

        # Sort by CreationDate descending
        items.sort(key=lambda x: x.get("CreationDate", ""), reverse=True)
        return items

    @staticmethod
    async def get_raw_data_with_offset(tenantId: str, days: int = 15, offset_days: int = 0) -> List[dict]:
        """Retrieve raw tenant stats data with offset for comparison."""
        today = datetime.utcnow() - timedelta(days=offset_days)
        start_date = today - timedelta(days=days)

        start_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_str = today.strftime("%Y-%m-%dT23:59:59")

        data = _load_json_file(TENANT_STATS_FILE)
        items = [
            item for item in data
            if item.get("TenantId") == tenantId
            and start_str <= item.get("CreationDate", "") <= end_str
        ]

        items.sort(key=lambda x: x.get("CreationDate", ""), reverse=True)
        return items
