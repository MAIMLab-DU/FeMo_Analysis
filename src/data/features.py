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
    
    def __init__(self,
                 base_dir,
                 inference: bool = True,
                 sensor_freq: int = 1024,
                 sensor_selection: list = ['accelerometer', 
                                           'piezoelectric_small', 
                                           'piezoelectric_large']) -> None:
        
        self._base_dir = base_dir
        self.inference = inference
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
        num_segments = len(extracted_sensor_data)

        num_common_features = 53  # i.e. max, mean, sum, sd, percentile, skew, kurtosis, ...
        # +1 feature for segment duration
        total_features = (self.num_sensors + 2) * num_common_features + 1        
        
        # Extraction of features from TPDs and FPDs
        X_extracted = np.zeros((num_segments, total_features))  # Features of TPD

        self._logger.debug("Extracting features...")
        # For each TP segment in a data file
        for i in range(len(extracted_sensor_data)):
            # Duration of each TPD in s
            X_extracted[i, 0] = len(extracted_sensor_data[i]) / self.sensor_freq

            for k in range(self.num_sensors + 2):
                # Assigning sensor data, IMU rotation, or acceleration based on the sensor index
                if k == self.num_sensors:
                    sensor_data = extracted_imu_rotation[i]
                elif k == self.num_sensors + 1:
                    sensor_data = extracted_imu_acceleration[i]
                else:
                    sensor_data = extracted_sensor_data[i][:, k]
                    sensor_data_threshold = np.abs(sensor_data) - threshold[k]

                # For IMU rotation/acceleration, no threshold adjustment needed
                if k < self.num_sensors:
                    sensor_data_above_threshold = sensor_data_threshold[sensor_data_threshold > 0]
                else:
                    sensor_data_threshold = sensor_data  # No threshold adjustment
                    sensor_data_above_threshold = sensor_data

                # =============================================================================
                                # S_MPDatom = np.abs(TPD_extracted[i][j]) - threshold[i] 
                # =============================================================================
                # getMPDatom expects 2D array. 
                # 2D array representing a set of epoch signals. Each row of the array corresponds to an epoch signal, 
                # and each column corresponds to a sample in the signal.
                
                # ------------ Time domain features ------------
                # feat_indice_range = (k * num_common_features + 1, k * num_common_features + 8)
                # X_extracted[i, feat_indice_range[0]: feat_indice_range[1]] = self._calc_time_domain_feats(sensor_data_threshold)
                # Max value
                X_extracted[i, k * num_common_features + 1] = np.max(sensor_data_threshold)
                # Mean value
                X_extracted[i, k * num_common_features + 2] = np.mean(sensor_data_threshold)
                # Energy
                X_extracted[i, k * num_common_features + 3] = np.sum(sensor_data_threshold ** 2)
                # Standard deviation
                X_extracted[i, k * num_common_features + 4] = np.std(sensor_data_threshold)
                # Interquartile range
                X_extracted[i, k * num_common_features + 5] = np.percentile(sensor_data_threshold, 75) - np.percentile(sensor_data_threshold, 25)
                # Skewness
                X_extracted[i, k * num_common_features + 6] = skew(sensor_data_threshold)
                # Kurtosis
                X_extracted[i, k * num_common_features + 7] = kurtosis(sensor_data_threshold)

                if len(sensor_data_above_threshold) == 0:
                    # Duration above threshold
                    X_extracted[i, k * num_common_features + 8]  = 0
                    # Mean above threshold value
                    X_extracted[i, k * num_common_features + 9]  = 0
                    # Energy above threshold value
                    X_extracted[i, k * num_common_features + 10] = 0
                else:
                    # Duration above threshold
                    X_extracted[i, k * num_common_features + 8]  = len(sensor_data_above_threshold)
                    # Mean above threshold
                    X_extracted[i, k * num_common_features + 9]  = np.mean(sensor_data_above_threshold)
                    # Energy above threshold
                    X_extracted[i, k * num_common_features + 10] = np.sum(sensor_data_above_threshold ** 2)

                # ------------ Frequency domain features ------------
                # Gives the main frequency mode above 1 Hz
                _, _, X_extracted[i, k * num_common_features + 11] = get_frequency_mode(sensor_data, self.sensor_freq, 1)
                X_extracted[i, k * num_common_features + 12] = get_band_energy(sensor_data, self.sensor_freq, 1, 2)
                X_extracted[i, k * num_common_features + 13] = get_band_energy(sensor_data, self.sensor_freq, 2, 5)
                X_extracted[i, k * num_common_features + 14] = get_band_energy(sensor_data, self.sensor_freq, 5, 10)
                X_extracted[i, k * num_common_features + 15] = get_band_energy(sensor_data, self.sensor_freq, 10, 20)
                X_extracted[i, k * num_common_features + 16] = get_band_energy(sensor_data, self.sensor_freq, 20, 30)
                
                # =============================================================================
                #                 New features added by Omar 
                # =============================================================================

                # Calculating instantaneous frequency and amplitude
                X_extracted[i, k * num_common_features + 17] = np.mean(get_inst_amplitude(sensor_data_threshold))
                X_extracted[i, k * num_common_features + 18] = np.mean(get_inst_frequency(sensor_data, self.sensor_freq))

                X_extracted[i, k * num_common_features + 19] = np.std(get_inst_amplitude(sensor_data_threshold))
                X_extracted[i, k * num_common_features + 20] = np.std(get_inst_frequency(sensor_data, self.sensor_freq))

                X_extracted[i, k * num_common_features + 21] = np.max(get_inst_amplitude(sensor_data_threshold)) - np.min(get_inst_amplitude(sensor_data_threshold))
                X_extracted[i, k * num_common_features + 22] = np.max(get_inst_frequency(sensor_data, self.sensor_freq)) - np.min(get_inst_frequency(sensor_data, self.sensor_freq))

                # Calculating Wavelet Coefficient
                X_extracted[i, k * num_common_features + 23] = np.mean(get_wavelet_coeff(sensor_data))
                X_extracted[i, k * num_common_features + 24] = np.median(get_wavelet_coeff(sensor_data))
                X_extracted[i, k * num_common_features + 25] = np.std(get_wavelet_coeff(sensor_data))      
                
                # Calculate 1D Convolution -> Feature count starts at 26 and ends at 46
                convolved_signal = get_convolved_signal(sensor_data_threshold)
                for conv_index in range(21):
                    X_extracted[i, k * num_common_features + 26 + conv_index] = get_conv1D(convolved_signal, conv_index + 1)
                    
                # Calculate number of atoms
                X_extracted[i, k * num_common_features + 47] = 1 # getMPDatom(S_MPDatom) #  Passing S_MPDatom (2D array) 
                
                # Calculate time frequency matched filter
                X_extracted[i, k * num_common_features + 48] = get_time_freq_mod_filter(sensor_data_threshold)
                
                # =============================================================================
                #                 New features added by Monaf 
                # =============================================================================
                
                X_extracted[i, k * num_common_features + 49] = np.min(sensor_data_threshold)  # Min value
                X_extracted[i, k * num_common_features + 50] = np.median(sensor_data_threshold)  # Median value
                
                # Calculate the morphological features. 
                absolute_area, relative_area, absolute_area_differential = get_morphology(sensor_data_threshold, self.sensor_freq)
                X_extracted[i, k * num_common_features + 51] = absolute_area
                X_extracted[i, k * num_common_features + 52] = relative_area
                X_extracted[i, k * num_common_features + 53] = absolute_area_differential

        return {
            'num_segments': num_segments,
            'X_extracted': X_extracted
        }

    
    # TODO: implement functionality
    def _extract_features_for_train(self):
        ...

    
    def _calc_time_domain_feats(self, sensor_data_threshold):
        return [
            # Max value
            np.max(sensor_data_threshold),
            # Mean value
            np.mean(sensor_data_threshold),
            # Energy
            np.sum(sensor_data_threshold ** 2),
            # Standard deviation
            np.std(sensor_data_threshold),
            # Interquartile range
            np.percentile(sensor_data_threshold, 75) - np.percentile(sensor_data_threshold, 25),
            # Skewness
            skew(sensor_data_threshold),
            # Kurtosis
            kurtosis(sensor_data_threshold)
        ]


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
        # Placeholder method for decorator function
        pass