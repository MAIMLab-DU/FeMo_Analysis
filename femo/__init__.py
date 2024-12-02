from enum import Enum
from typing import Optional
from dataclasses import dataclass

__version__ = "1.0.0"

@dataclass(frozen=True)
class BeltConfig:
    accelerometer: Optional[list] = None
    piezoelectric_large: Optional[list] = None
    piezoelectric_small: Optional[list] = None
    acoustic: Optional[list] = None

class BeltTypes(Enum):
    A = BeltConfig(
        accelerometer=['sensor_1', 'sensor_2'],
        piezoelectric_large=['sensor_3', 'sensor_6'],
        piezoelectric_small=['sensor_4', 'sensor_5']
    )
    B = BeltConfig(
        accelerometer=['sensor_1', 'sensor_2'],
        acoustic=['sensor_3', 'sensor_6'],
        piezoelectric_small=['sensor_4', 'sensor_5']
    )
    C = BeltConfig(
        accelerometer=['sensor_1', 'sensor_2'],
        piezoelectric_large=['sensor_3', 'sensor_6'],
        acoustic=['sensor_4', 'sensor_5']
    )

__all__ = (
    "__version__",
    "BeltTypes"
)
