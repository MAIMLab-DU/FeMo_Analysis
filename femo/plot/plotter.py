import numpy as np
import matplotlib.pyplot as plt
from logger import LOGGER
from typing import Union, List, Literal


class FeMoPlotter(object):

    @property
    def logger(self):
        return LOGGER
    
    @property
    def sensor_map(self):
        return {
            'accelerometer': ['sensor_1', 'sensor_2'],
            'piezoelectric_large': ['sensor_3', 'sensor_6'],
            'piezoelectric_small': ['sensor_4', 'sensor_5']
        }

    @property
    def sensors(self) -> list:
        return sorted([item for s in self._sensor_selection for item in self.sensor_map[s]])

    @property
    def num_sensors(self) -> int:
        return len(self.sensors)
    
    def __init__(self,
                 sensor_freq: int = 1024,
                 sensation_freq: int = 1024,
                 sensor_selection: list = ['accelerometer', 
                                           'piezoelectric_small', 
                                           'piezoelectric_large']) -> None:
        
        self._sensor_selection = sensor_selection
        self._sensor_freq = sensor_freq
        self._sensation_freq = sensation_freq
        self.fig, self.ax = None, None

    def create_figure(self,
                      num_detections: int = 2,
                      figsize: tuple = (16, 15)):
        
        self.fig, self.ax = plt.subplots(self.num_sensors + num_detections, 1,
                                         figsize=figsize, sharex=True)
    
    def plot_sensor_data(self,
                         ax,
                         axis_idx: int,
                         data: Union[List, np.ndarray],
                         sensor_name: str,
                         x_unit: Literal['min', 'sec'] = 'min'):
        
        time_data = np.arange(0, len(data), 1)
        if x_unit == 'sec':
            time_data = time_data / self._sensor_freq
        if x_unit == 'min':
            time_data = time_data / self._sensor_freq / 60
        
        ax[axis_idx].plot(time_data, data, label=sensor_name)
        ax[axis_idx].set_ylabel("Signal")
        ax[axis_idx].set_xlabel("")
        ax[axis_idx].legend()

    def plot_detections(self,
                        ax,
                        axis_idx: int,
                        detection_map: Union[List, np.ndarray],
                        det_type: str,
                        x_unit: Literal['min', 'sec'] = 'min'):
        
        time_data = np.arange(0, len(detection_map), 1)
        if x_unit == 'sec':
            time_data = time_data / self._sensor_freq
        if x_unit == 'min':
            time_data = time_data / self._sensor_freq / 60
        
        ax[axis_idx].plot(time_data, detection_map, label=det_type)
        ax[axis_idx].set_ylabel("Detection")
        ax[axis_idx].set_xlabel("")
        ax[axis_idx].legend()

    def save_figure(self, filename: str) -> None:
        plt.tight_layout()
        self.fig.savefig(filename)
        plt.close(self.fig)
        self.logger.info(f"Saved figure to {filename}")

    
