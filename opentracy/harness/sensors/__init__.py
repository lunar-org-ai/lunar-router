"""Deterministic signal producers.

Sensors turn raw measurements into ledger `signal` entries. They're the
left edge of the trigger engine: everything downstream (policies,
dispatches, agent runs) happens only because a sensor decided a signal
was worth emitting.
"""

from .base import ObjectiveSensor, SENSOR_TAG_REGRESSION
from .cadence import CadenceSensor, SENSOR_TAG_CADENCE
from .new_dataset import NewDatasetSensor, SENSOR_TAG_NEW_DATASET
from .traces_threshold import (
    NewTracesThresholdSensor,
    SENSOR_TAG_TRACES_THRESHOLD,
)

__all__ = [
    "CadenceSensor",
    "NewDatasetSensor",
    "NewTracesThresholdSensor",
    "ObjectiveSensor",
    "SENSOR_TAG_CADENCE",
    "SENSOR_TAG_NEW_DATASET",
    "SENSOR_TAG_REGRESSION",
    "SENSOR_TAG_TRACES_THRESHOLD",
]
