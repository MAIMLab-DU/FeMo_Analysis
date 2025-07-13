import time
import numpy as np
import pandas as pd
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
    
    def transform(self, loaded_data: dict) -> dict:
        """
        Applies band-pass filtering, trims edge data, and optionally debounces maternal sensation signal,
        without mutating the original loaded_data.

        Filtering:
            - FM sensors (sensor_1 through sensor_6): 1-30 Hz band-pass
            - IMU acceleration + rotation: 1-10 Hz band-pass

        Trimming:
            - Remove first/last removal_period seconds (30 s if total duration > 5 min, else 5 s)

        Debouncing:
            - Optional logic to clean up binary maternal sensation signal

        Args:
            loaded_data (dict): Contains raw data for keys:
                'sensor_1'…'sensor_6' (np.ndarray),
                'imu_acceleration' (np.ndarray),
                'imu_rotation' (pd.DataFrame with ['roll','pitch','yaw']),
                and optionally 'sensation_data' (np.ndarray).

        Returns:
            dict: Preprocessed outputs:
                - 'sensor_1'…'sensor_6' (np.ndarray)
                - 'imu_acceleration' (np.ndarray)
                - 'imu_rotation' (pd.DataFrame)
                - 'imu_rotation_1D' (np.ndarray)
                - 'sensation_data' (np.ndarray)
        """
        # Start timer
        start = time.time()
        self.logger.debug(f"Starting preprocessing for keys: {list(loaded_data.keys())}")

        # 1) Extract raw inputs without mutating loaded_data
        s1_old = loaded_data['sensor_1']
        s2_old = loaded_data['sensor_2']
        s3_old = loaded_data['sensor_3']
        s4_old = loaded_data['sensor_4']
        s5_old = loaded_data['sensor_5']
        s6_old = loaded_data['sensor_6']
        acc_old = loaded_data['imu_acceleration']
        rot_df_old = loaded_data['imu_rotation']
        sens_old = loaded_data.get('sensation_data', np.array([], dtype=np.int8))

        # 2) Pre-allocate new arrays (float32 to minimize memory)
        N = s1_old.shape[0]
        s1 = np.empty(N, dtype=np.float32)
        s2 = np.empty(N, dtype=np.float32)
        s3 = np.empty(N, dtype=np.float32)
        s4 = np.empty(N, dtype=np.float32)
        s5 = np.empty(N, dtype=np.float32)
        s6 = np.empty(N, dtype=np.float32)
        imu_acc = np.empty(N, dtype=np.float32)
        imu_rot_arr = np.empty((N, 3), dtype=np.float32)
        sens_data = sens_old.astype(np.int8).copy()

        # 3) Design band-pass filters
        fs = self.sensor_freq
        order = 10
        sos_FM  = butter(order//2, [1, 30], btype='bandpass', fs=fs, output='sos')
        sos_IMU = butter(order//2, [1, 10], btype='bandpass', fs=fs, output='sos')

        # 4) Batch-filter all 6 FM channels at once
        fm_stack = np.vstack([s1_old, s2_old, s3_old, s4_old, s5_old, s6_old])
        fm_filt  = sosfiltfilt(sos_FM, fm_stack, axis=1)
        s1, s2, s3, s4, s5, s6 = fm_filt.astype(np.float32)
        self.logger.debug("FM channels filtered")

        # 5) Batch-filter IMU accel + 3 rotation dims in one go
        imu_stack = np.vstack([acc_old, 
                                rot_df_old['roll'].values,
                                rot_df_old['pitch'].values,
                                rot_df_old['yaw'].values])
        imu_filt    = sosfiltfilt(sos_IMU, imu_stack, axis=1)
        imu_acc     = imu_filt[0].astype(np.float32)
        imu_rot_arr = imu_filt[1:].T.astype(np.float32)
        self.logger.debug("FM & IMU channels batch-filtered")

        # 6) Trim edges to remove startup/shutdown transients
        duration_sec = N / fs
        removal_period = 30 if duration_sec > 300 else 5
        i0, i1 = int(removal_period * fs), -int(removal_period * fs)
        s1 = s1[i0:i1]
        s2 = s2[i0:i1]
        s3 = s3[i0:i1]
        s4 = s4[i0:i1]
        s5 = s5[i0:i1]
        s6 = s6[i0:i1]
        imu_acc    = imu_acc[i0:i1]
        imu_rot_arr = imu_rot_arr[i0:i1, :]
        sens_data   = sens_data[i0:i1]
        self.logger.debug(f"Signals trimmed to indices [{i0}:{i1}] (removal_period={removal_period}s)")

        # 7) Optionally resolve debounce noise
        if sens_data.size and str2bool(self.resolve_debounce):
            sens_data = self._resolve_debouncing(sens_data)
            self.logger.debug("Debouncing applied")

        # 8) PCA on IMU rotation → 1D summary
        imu_pca = apply_pca(imu_rot_arr)
        self.logger.debug(f"PCA on rotation produced shape {imu_pca.shape}")

        # 9) Wrap Euler angles back into DataFrame for downstream compatibility
        # preserve the original time-step indices
        orig_idx      = rot_df_old.index
        trimmed_idx   = orig_idx[i0:i1]
        imu_rot_df = pd.DataFrame(
            imu_rot_arr,
            index=trimmed_idx,
            columns=['roll', 'pitch', 'yaw']
        )

        # 10) Assemble and return processed dictionary
        processed = {
            'sensor_1':         s1,
            'sensor_2':         s2,
            'sensor_3':         s3,
            'sensor_4':         s4,
            'sensor_5':         s5,
            'sensor_6':         s6,
            'imu_acceleration': imu_acc,
            'imu_rotation':     imu_rot_df,
            'imu_rotation_1D':  imu_pca,
            'sensation_data':   sens_data,
        }

        elapsed_ms = (time.time() - start) * 1000
        self.logger.info(f"Preprocessing finished in {elapsed_ms:.2f} ms")
        return processed
