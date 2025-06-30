import numpy as np
import matplotlib.pyplot as plt
from ..logger import LOGGER
from matplotlib.figure import Figure
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
                      figsize: tuple = (16, 15),
                      num_subplots: int = 8) -> tuple:
        
        # Plot num_sensors and 2 more rows (detections) of subplots
        fig, axes = plt.subplots(num_subplots, 1,
                               figsize=figsize, sharex=True)
        return fig, axes
    
    def plot_sensor_data(self,
                         axes: np.ndarray,
                         axis_idx: int,
                         data: Union[List, np.ndarray],
                         sensor_name: str,
                         x_unit: Literal['min', 'sec'] = 'min'):
        if axes is None:
            raise ValueError('ax must be specified')
        
        self.logger.info(f"Plotting sensor data for '{sensor_name}'")
        
        time_data = np.arange(0, len(data), 1)
        if x_unit == 'sec':
            time_data = time_data / self._sensor_freq
        if x_unit == 'min':
            time_data = time_data / self._sensor_freq / 60
        
        axes[axis_idx].plot(time_data, data, label=sensor_name)
        axes[axis_idx].set_ylabel("Signal")
        axes[axis_idx].set_xlabel("")
        axes[axis_idx].legend()

        return axes

    def plot_detections(self,
                        axes: np.ndarray,
                        axis_idx: int,
                        detection_map: Union[List, np.ndarray],
                        det_type: str,
                        xlabel: str,
                        ylabel: str,
                        x_unit: Literal['min', 'sec'] = 'min'):
        if axes is None:
            raise ValueError('ax must be specified')
        
        self.logger.info(f"Plotting detections for '{det_type}'")
        
        time_data = np.arange(0, len(detection_map), 1)
        if x_unit == 'sec':
            time_data = time_data / self._sensor_freq
        if x_unit == 'min':
            time_data = time_data / self._sensor_freq / 60
        
        axes[axis_idx].plot(time_data, detection_map, label=det_type)
        axes[axis_idx].set_ylabel(ylabel)
        axes[axis_idx].set_xlabel(xlabel)
        axes[axis_idx].legend()

        return axes

    def save_figure(self, fig: Figure, filename: str) -> None:
        assert fig is not None, "fig is not specified"
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        self.logger.info(f"Saved figure to {filename}")

    
