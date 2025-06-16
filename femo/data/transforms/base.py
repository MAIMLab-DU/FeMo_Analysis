from ...logger import LOGGER
from abc import ABC, abstractmethod

class BaseTransform(ABC):

    @property
    def sensor_map(self):
        return self._sensor_map

    @property
    def logger(self):
        return LOGGER
    
    @property
    def sensor_selection(self) -> list:
        return self._sensor_selection
    
    @property
    def sensor_freq(self) -> int:
        return self._sensor_freq
    
    @property
    def sensation_freq(self) -> int:
        return self._sensation_freq
    
    @property
    def sensors(self) -> list:
        sensors = []
        for s in self.sensor_selection:
            if '_left' in s or '_right' in s:
                key, side = s.rsplit('_', 1)
                index = 0 if side == 'left' else 1
                sensors.append(self.sensor_map[key][index])
            else:
                sensors.extend(self.sensor_map[s])
        return sorted(sensors)

    @property
    def num_sensors(self) -> int:
        return len(self.sensors)
    
    @property
    def use_all_sensors(self) -> bool:
        return True if len(self.sensor_selection) == len(self.sensor_map) else False
    
    def __init__(self,
                 description: str = "Only large piezos from belt types 'A' and 'C'",
                 sensor_freq: int = 1024,
                 sensation_freq: int = 1024,
                 sensor_map: dict = {
                     'accelerometer': ['sensor_1', 'sensor_2'],
                     'piezoelectric_large': ['sensor_3', 'sensor_6'],
                     'other': ['sensor_4', 'sensor_5']
                 },
                 sensor_selection: list = ['piezoelectric_large']) -> None:
        super().__init__()

        self._description = description
        self._sensor_map = sensor_map
        self._sensor_selection = sensor_selection
        self._sensor_freq = sensor_freq
        self._sensation_freq = sensation_freq
    
    def __repr__(self):
        return self._description

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    
    @abstractmethod
    def transform(self, *args, **kwargs):
        raise NotImplementedError
