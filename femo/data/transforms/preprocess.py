import time
import copy
import numpy as np
from scipy.signal import butter, sosfiltfilt
from ._utils import apply_pca, str2bool
from .base import BaseTransform


class DataPreprocessor(BaseTransform): 

    def __init__(self,
                 resolve_debounce: bool = True,
                 debounce_thresh: float = 97.65625,  # milliseconds (default value might be too low, because avg human reaction time for touch is ~150ms)
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.resolve_debounce = resolve_debounce
        self.debounce_thresh = debounce_thresh

    def _resolve_debouncing(self, sensation_data: np.ndarray):
        """
        Resolves debouncing issues in the sensation data by ensuring
        isolated '1's surrounded by '0's within a threshold are retained.

        Parameters:
        - sensation_data (np.ndarray): Input array with debouncing noise.

        Returns:
        - np.ndarray: Processed array with debouncing resolved.
        """
        
        new_sensation_data = np.zeros_like(sensation_data)
        sample_range = int(self.debounce_thresh * self.sensation_freq / 1000)
        self.logger.debug(f"Resolving debounce issues with debounce_time: {self.debounce_thresh}ms, sample_range: {sample_range}")
        
        for i in range(len(sensation_data)):
            count = 0
            if sensation_data[i] == 1: 
                for j in range(1, sample_range + 1):
                    if (i+j) < len(sensation_data) and sensation_data[i+j] == 0:
                        count+=1
                    else:
                        break
                if count < sample_range:
                    new_sensation_data[i:i+j] = 1
                    i = i + j
        return new_sensation_data
    
    def transform(self, loaded_data: dict):
        """
        Applies band-pass filtering, trims edge data, and optionally debounces maternal sensation signal.

        This method preprocesses raw multimodal sensor data collected from a fetal movement monitoring system.
        It applies separate band-pass filters for fetal movement sensors (FM) and IMU channels, trims
        the signal from both ends to eliminate potential noise, and converts IMU rotation quaternions to
        Euler angles. It also applies PCA on rotation data to produce a 1D summary.

        Filtering:
            - FM data (e.g., accelerometers and piezo sensors): 1–30 Hz band-pass
            - IMU data (e.g., acceleration, rotation): 1–10 Hz band-pass
        Trimming:
            - First and last 30 seconds removed if total duration > 5 minutes, otherwise 5 seconds
        Debouncing:
            - Optional logic to reduce noise in binary maternal sensation data

        Args:
            loaded_data (dict): Dictionary containing raw sensor arrays for keys like
                'sensor_1' through 'sensor_6', 'imu_acceleration', 'imu_rotation' (DataFrame with roll, pitch, yaw),
                and optionally 'sensation_data'.

        Returns:
            dict: A dictionary with preprocessed sensor data including:
                - Filtered and trimmed signals
                - Updated 'imu_rotation' (DataFrame)
                - 1D PCA result in 'imu_rotation_1D'
                - Debounced and trimmed 'sensation_data' if available
        """

        start = time.time()
        self.logger.debug(f"Started Preprocessing for data with keys: {list(loaded_data.keys())}")

        preprocessed_data = copy.deepcopy(loaded_data)

        # ---------------------------- Filter Settings -----------------------------#
        filter_order = 10
        lowCutoff_FM = 1
        highCutoff_FM = 30
        lowCutoff_IMU = 1
        highCutoff_IMU = 10
        highCutoff_force = 10   # noqa: F841 

        duration_sec = len(loaded_data['sensor_1']) / self.sensor_freq
        removal_period = 30 if duration_sec > 300 else 5

        self.logger.debug(f"Estimated signal duration: {duration_sec:.1f} seconds")
        self.logger.debug(f"Using removal period: {removal_period} seconds")
        self.logger.debug(f"Bandpass filter for FM: {lowCutoff_FM}-{highCutoff_FM} Hz (order {filter_order})")
        self.logger.debug(f"Bandpass filter for IMU: {lowCutoff_IMU}-{highCutoff_IMU} Hz (order {filter_order})")

        # ---------------------------- Filter Design -----------------------------#
        sos_FM = butter(filter_order // 2, [lowCutoff_FM, highCutoff_FM], btype='bandpass', fs=self.sensor_freq, output='sos')
        sos_IMU = butter(filter_order // 2, [lowCutoff_IMU, highCutoff_IMU], btype='bandpass', fs=self.sensor_freq, output='sos')

        # -------------------------- Apply Filtering ----------------------------#
        # Band-pass filters are applied separately:
        # - FM sensors (piezos + accelerometers): 1–30 Hz
        # - IMU sensors (acceleration + rotation): 1–10 Hz
        # These ranges are chosen to isolate typical movement frequencies while suppressing noise.
        # Dynamically filter all keys matching the pattern 'sensor_{i}'
        sensor_keys = [k for k in preprocessed_data.keys() if k.startswith("sensor_") and preprocessed_data[k] is not None]

        for key in sensor_keys:
            preprocessed_data[key] = sosfiltfilt(sos_FM, preprocessed_data[key])
        preprocessed_data['imu_acceleration'] = sosfiltfilt(sos_IMU, preprocessed_data['imu_acceleration'])

        for axis in ['roll', 'pitch', 'yaw']:
            preprocessed_data['imu_rotation'][axis] = sosfiltfilt(sos_IMU, preprocessed_data['imu_rotation'][axis].values)
        self.logger.debug("Filtering applied to all FM and IMU sensor channels")

        # -------------------------- Trim Data ----------------------------------#
        start_index = removal_period * self.sensor_freq
        end_index = -removal_period * self.sensor_freq

        def safe_trim(arr, name):
            try:
                trimmed = arr[start_index:end_index]
                self.logger.debug(f"{name} trimmed to {len(trimmed)} samples")
                return trimmed
            except Exception as e:
                self.logger.warning(f"Could not trim {name}: {e}")
                return arr

        for key in sensor_keys:
            preprocessed_data[key] = safe_trim(preprocessed_data[key], key)
        preprocessed_data['imu_acceleration'] = safe_trim(preprocessed_data['imu_acceleration'], 'imu_acceleration')
        preprocessed_data['imu_rotation'] = preprocessed_data['imu_rotation'].iloc[start_index:end_index]
        self.logger.debug(f"Data trimmed with removal period: {removal_period} seconds")

        try:
            preprocessed_data['sensation_data'] = safe_trim(preprocessed_data['sensation_data'], 'sensation_data')
            if str2bool(self.resolve_debounce):
                preprocessed_data['sensation_data'] = self._resolve_debouncing(preprocessed_data['sensation_data'])
        except Exception as e:
            self.logger.warning(f"Failed to process sensation_data: {e}")
            preprocessed_data['sensation_data'] = np.array([])

        # ---------------------- PCA on IMU Rotation ---------------------------#
        preprocessed_data['imu_rotation_1D'] = apply_pca(preprocessed_data['imu_rotation'])
        self.logger.debug(f"PCA applied to imu_rotation. Explained shape: {preprocessed_data['imu_rotation_1D'].shape}")

        elapsed_ms = (time.time() - start) * 1000
        self.logger.info(f"Preprocessing completed in {elapsed_ms:.2f} ms")

        return preprocessed_data
