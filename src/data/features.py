import logging
import numpy as np
from functools import wraps
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from .feature_utils import (
    get_frequency_mode,
    get_band_energy,
    get_inst_amplitude,
    get_inst_frequency,
    get_wavelet_coeff,
    get_convolved_signal,
    get_conv1D,
    get_time_freq_mod_filter,
    get_morphology
)
from .config import SENSOR_MAP


class FeatureExtractor:
    @property
    def _logger(self):
        return logging.getLogger(__name__)
    
    def __init__(self, base_dir,
                 inferece: bool = True,
                 sensor_freq: int = 1024,
                 sensor_selection: list = ['accelerometer', 
                                           'piezoelectric_small', 
                                           'piezoelectric_large']) -> None:
        
        self._base_dir = base_dir
        self.inference = inferece
        self.sensor_freq = sensor_freq
        self.sensor_selection = sensor_selection
        self.sensors = [item for s in self.sensor_selection for item in SENSOR_MAP[s]]
        self.num_sensors = len(self.sensors)

    # TODO: implement functionality
    def _extract_features_for_inference(self, 
                                          extracted_detections: dict, 
                                          fm_dict: dict):
        
        threshold = fm_dict['fm_threshold']
        
        extracted_sensor_data = extracted_detections['extracted_sensor_data']
        extracted_imu_acceleration = extracted_detections['extracted_imu_acceleration']
        extracted_imu_rotation = extracted_detections['extracted_imu_rotation']
        num_extracted = len(extracted_sensor_data)

        num_common_features = 53  # i.e. max, mean, sum, sd, percentile, skew, kurtosis, ...
        # +1 feature for segment duration
        total_features = (self.num_sensors + 2) * num_common_features + 1        
        
        # Extraction of features from TPDs and FPDs
        X_extracted = np.zeros((num_extracted, total_features))  # Features of TPD

        self._logger.debug("Extracting features...")
        # For each TP segment in a data file
        for i in range(len(extracted_sensor_data)):
            # Duration of each TPD in s
            X_extracted[i, 0] = len(extracted_sensor_data[i]) / self.sensor_freq

            for k in range(self.num_sensors + 2):
                # IMU_aclm_TPD, IMU_rot_TPD
                if k == self.num_sensors:
                    S = extracted_imu_rotation[i]
                    S_thd = S
                    S_thd_above = S
                
                elif k == self.num_sensors + 1: 
                    S = extracted_imu_acceleration[i]
                    S_thd = S
                    S_thd_above = S

                else: 
                    S = extracted_sensor_data[i][:, k]
                    S_thd = np.abs(S) - threshold[i, k]
                    S_thd_above = S_thd[S_thd > 0]

                # =============================================================================
                                # S_MPDatom = np.abs(TPD_extracted[i][j]) - threshold[i] 
                # =============================================================================
                # getMPDatom expects 2D array. 
                # 2D array representing a set of epoch signals. Each row of the array corresponds to an epoch signal, 
                # and each column corresponds to a sample in the signal.
                
                # ------------ Time domain features ------------
                # Max value
                X_extracted[i, k * num_common_features + 1] = np.max(S_thd)
                # Mean value
                X_extracted[i, k * num_common_features + 2] = np.mean(S_thd)
                # Energy
                X_extracted[i, k * num_common_features + 3] = np.sum(S_thd ** 2)
                # Standard deviation
                X_extracted[i, k * num_common_features + 4] = np.std(S_thd)
                # Interquartile range
                X_extracted[i, k * num_common_features + 5] = np.percentile(S_thd, 75) - np.percentile(S_thd, 25)
                # Skewness
                X_extracted[i, k * num_common_features + 6] = skew(S_thd)
                # Kurtosis
                X_extracted[i, k * num_common_features + 7] = kurtosis(S_thd)

                if len(S_thd_above) == 0:
                    # Duration above threshold
                    X_extracted[i, k * num_common_features + 8]  = 0
                    # Mean above threshold value
                    X_extracted[i, k * num_common_features + 9]  = 0
                    # Energy above threshold value
                    X_extracted[i, k * num_common_features + 10] = 0
                else:
                    # Duration above threshold
                    X_extracted[i, k * num_common_features + 8]  = len(S_thd_above)
                    # Mean above threshold
                    X_extracted[i, k * num_common_features + 9]  = np.mean(S_thd_above)
                    # Energy above threshold
                    X_extracted[i, k * num_common_features + 10] = np.sum(S_thd_above ** 2)

                # ------------ Frequency domain features ------------
                # Gives the main frequency mode above 1 Hz
                _, _, X_extracted[i, k * num_common_features + 11] = get_frequency_mode(S, self.sensor_freq, 1)
                X_extracted[i, k * num_common_features + 12] = get_band_energy(S, self.sensor_freq, 1, 2)
                X_extracted[i, k * num_common_features + 13] = get_band_energy(S, self.sensor_freq, 2, 5)
                X_extracted[i, k * num_common_features + 14] = get_band_energy(S, self.sensor_freq, 5, 10)
                X_extracted[i, k * num_common_features + 15] = get_band_energy(S, self.sensor_freq, 10, 20)
                X_extracted[i, k * num_common_features + 16] = get_band_energy(S, self.sensor_freq, 20, 30)
                
                # =============================================================================
                #                 New features added by Omar 
                # =============================================================================

                # Calculating instantaneous frequency and amplitude
                X_extracted[i, k * num_common_features + 17] = np.mean(get_inst_amplitude(S_thd))
                X_extracted[i, k * num_common_features + 18] = np.mean(get_inst_frequency(S, self.sensor_freq))

                X_extracted[i, k * num_common_features + 19] = np.std(get_inst_amplitude(S_thd))
                X_extracted[i, k * num_common_features + 20] = np.std(get_inst_frequency(S, self.sensor_freq))

                X_extracted[i, k * num_common_features + 21] = np.max(get_inst_amplitude(S_thd)) - np.min(get_inst_amplitude(S_thd))
                X_extracted[i, k * num_common_features + 22] = np.max(get_inst_frequency(S, self.sensor_freq)) - np.min(get_inst_frequency(S, self.sensor_freq))

                # Calculating Wavelet Coefficient
                X_extracted[i, k * num_common_features + 23] = np.mean(get_wavelet_coeff(S))
                X_extracted[i, k * num_common_features + 24] = np.median(get_wavelet_coeff(S))
                X_extracted[i, k * num_common_features + 25] = np.std(get_wavelet_coeff(S))      
                
                # Calculate 1D Convolution -> Feature count starts at 26 and ends at 46
                convolved_signal = get_convolved_signal(S_thd)
                for conv_index in range(21):
                    X_extracted[i, k * num_common_features + 26 + conv_index] = get_conv1D(convolved_signal, conv_index + 1)
                    
                # Calculate number of atoms
                X_extracted[i, k * num_common_features + 47] = 1 # getMPDatom(S_MPDatom) #  Passing S_MPDatom (2D array) 
                
                # Calculate time frequency matched filter
                X_extracted[i, k * num_common_features + 48] = get_time_freq_mod_filter(S_thd)
                
                # =============================================================================
                #                 New features added by Monaf 
                # =============================================================================
                
                X_extracted[i, k * num_common_features + 49] = np.min(S_thd)  # Min value
                X_extracted[i, k * num_common_features + 50] = np.median(S_thd)  # Median value
                
                # Calculate the morphological features. 
                absolute_area, relative_area, absolute_area_differential = get_morphology(S_thd, self.sensor_freq)
                X_extracted[i, k * num_common_features + 51] = absolute_area
                X_extracted[i, k * num_common_features + 52] = relative_area
                X_extracted[i, k * num_common_features + 53] = absolute_area_differential

        return X_extracted
    
    # TODO: implement functionality
    def _extract_features_for_train(self):
        ...

    def decorator(method):
        """Decorator that checks the `inference` flag and calls the appropriate method."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self.inference:
                return self._extract_features_for_inference(*args, **kwargs)
            else:
                return self._extract_features_for_train(*args, **kwargs)
        return wrapper
    
    @decorator
    def extract_features(self):
        pass