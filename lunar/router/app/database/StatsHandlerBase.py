from abc import ABC, abstractmethod
from typing import Dict, List
from .models import TenantStatsModel
from boto3.dynamodb.types import TypeDeserializer


class StatsHandlerBase(ABC):
    region: str = "us-east-1"
    table_name: str  
    deserializer = TypeDeserializer()

    @staticmethod
    def deserialize_item(item):
        return {k: StatsHandlerBase.deserializer.deserialize(v) for k, v in item.items()}
    
    @staticmethod
    def _percentile(vals: List[float], p: float) -> float:
        """
        Calculate the percentile value from a list of floats.
        If the list is empty, return infinity.
        """
        if not vals:
            return float("inf")
        vals = sorted(vals)
        idx = max(0, min(len(vals)-1, int(round((p/100.0)*(len(vals)-1)))))
        return vals[idx]

    @staticmethod
    @abstractmethod
    def insert(stats: TenantStatsModel):
        pass

    @staticmethod
    @abstractmethod
    async def summary(model: str, provider: str, days: int) -> Dict[str, float]:
        """
        must return a summary of metrics for a given model + provider within the specified number of days.
        """
        pass

    @staticmethod
    @abstractmethod
    async def all_for_model(model: str, days: int) -> Dict[str, Dict[str, float]]:
        """
        must return summaries for all providers of a given model within the specified number of days.
        """
        pass

