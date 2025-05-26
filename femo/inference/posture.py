import copy
import numpy as np
import pandas as pd
from typing import Dict, Any, Literal
from .utils import (
    smooth_signal,
    smooth_labels,
    classify_posture
)
from ..logger import LOGGER

class PostureDetection:
    """
    Class for detecting maternal body orientation from IMU rotation angles.
    """

    def __init__(
        self,
        sensor_freq: int = 1024,
        filter_cutoff: float = 0.3,
        filter_order: int = 2,
        smoothing_window: int = 2,
        smooth_labels: bool = False,
        accel_sensitivity: dict = {
            'sensor_1': {'x': 0.3, 'y': 0.3, 'z': 0.3},
            'sensor_2': {'x': 0.3, 'y': 0.3, 'z': 0.3},
        },
        accel_bias: dict = {
            'sensor_1': {'x': 2.1, 'y': 2.2, 'z': 2.1},
            'sensor_2': {'x': 2.2, 'y': 2.2, 'z': 2.1},
        }
    ):
        self.sensor_freq = sensor_freq
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.smoothing_window = smoothing_window
        self.smooth_labels = smooth_labels 
        self.accel_sensitivity = accel_sensitivity
        self.accel_bias = accel_bias

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "sensor_freq": self.sensor_freq,
            "filter_cutoff": self.filter_cutoff,
            "filter_order": self.filter_order,
            "smoothing_window": self.smoothing_window,
        }
    
    @property
    def logger(self):
        return LOGGER

    def create_map(self, loaded_data: Dict[str, Any], mode: Literal['rot', 'acc']) -> np.ndarray:
        """
        Detect maternal orientation from preprocessed_data dict.

        Args:
            preprocessed_data (Dict[str, Any]): Dictionary containing at least 'imu_rotation' DataFrame.

        Returns:
            np.ndarray: Array of orientation labels.
        """

        if mode == 'rot':
            rotation_df: pd.DataFrame = copy.deepcopy(loaded_data.get('imu_rotation'))

            if rotation_df is None or rotation_df.empty:
                return np.array([], dtype=int)
            required_columns = {'roll', 'pitch'}
            if not required_columns.issubset(rotation_df.columns):
                missing = required_columns - set(rotation_df.columns)
                raise KeyError(f"Missing columns in rotation dataframe: {missing}")       

            removal_period = 30 if len(rotation_df) > self.sensor_freq * 5 * 60 else 5
            start_idx, end_idx = removal_period * self.sensor_freq, -removal_period * self.sensor_freq
            rotation_df = rotation_df.iloc[start_idx:end_idx]
            if rotation_df.empty:
                self.logger.warning("Not enough data after removal period. Returning empty orientation labels.")
                return np.array([], dtype=int)

            roll = smooth_signal(
                rotation_df['roll'].values,
                cutoff=self.filter_cutoff,
                fs=self.sensor_freq,
                order=self.filter_order
            )
            pitch = smooth_signal(
                rotation_df['pitch'].values,
                cutoff=self.filter_cutoff,
                fs=self.sensor_freq,
                order=self.filter_order
            )
            vectorize_posture = np.vectorize(classify_posture)
            orientation_labels = vectorize_posture(roll=roll, pitch=pitch)

        elif mode == 'acc':
            fm_acceleration: pd.DataFrame = copy.deepcopy(loaded_data.get('fm_acceleration'))
            acceleration_df = {
                'x': (
                    ((fm_acceleration['sensor_1']['x'] - self.accel_bias['sensor_1']['x']) / self.accel_sensitivity['sensor_1']['x']) -
                    ((fm_acceleration['sensor_2']['x'] - self.accel_bias['sensor_2']['x']) / self.accel_sensitivity['sensor_2']['x'])
                ) / 2,
                'y': (
                    ((fm_acceleration['sensor_1']['y'] - self.accel_bias['sensor_1']['y']) / self.accel_sensitivity['sensor_1']['x']) -
                    ((fm_acceleration['sensor_2']['y'] - self.accel_bias['sensor_2']['y']) / self.accel_sensitivity['sensor_2']['x'])
                ) / 2,
                'z': (
                    ((fm_acceleration['sensor_1']['z'] - self.accel_bias['sensor_1']['z']) / self.accel_sensitivity['sensor_1']['x']) +
                    ((fm_acceleration['sensor_2']['z'] - self.accel_bias['sensor_2']['z']) / self.accel_sensitivity['sensor_2']['x'])
                ) / 2
            }
            acceleration_df = pd.DataFrame(acceleration_df)

            if acceleration_df is None or acceleration_df.empty:
                return np.array([], dtype=int)
            required_columns = {'x', 'y', 'z'}
            if not required_columns.issubset(acceleration_df.columns):
                missing = required_columns - set(acceleration_df.columns)
                raise KeyError(f"Missing columns in rotation dataframe: {missing}")

            removal_period = 30 if len(acceleration_df) > self.sensor_freq * 5 * 60 else 5
            start_idx, end_idx = removal_period * self.sensor_freq, -removal_period * self.sensor_freq
            acceleration_df = acceleration_df.iloc[start_idx:end_idx]
            if acceleration_df.empty:
                self.logger.warning("Not enough data after removal period. Returning empty orientation labels.")
                return np.array([], dtype=int)
            
            ax = smooth_signal(
                acceleration_df['x'].values,
                cutoff=self.filter_cutoff,
                fs=self.sensor_freq,
                order=self.filter_order
            )
            ay = smooth_signal(
                acceleration_df['y'].values,
                cutoff=self.filter_cutoff,
                fs=self.sensor_freq,
                order=self.filter_order
            )
            az = smooth_signal(
                acceleration_df['z'].values,
                cutoff=self.filter_cutoff,
                fs=self.sensor_freq,
                order=self.filter_order
            )
            vectorize_posture = np.vectorize(classify_posture)
            orientation_labels = vectorize_posture(x=ax, y=ay, z=az)
            # orientation_labels = classify_posture(x=ax, y=ay, z=az)

        if self.smooth_labels:
            orientation_labels = smooth_labels(
                orientation_labels, window_time=self.smoothing_window, sensor_freq=self.sensor_freq
            )
        return orientation_labels.astype(int)
