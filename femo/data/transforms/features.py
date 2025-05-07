import tsfel
import numpy as np
from functools import wraps
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import tsfel.feature_extraction
import tsfel.feature_extraction.features_settings
from .feature_utils import (
    get_frequency_mode,
    get_band_energy,
    get_inst_amplitude,
    get_inst_frequency,
    get_wavelet_coeff,
    get_convolved_signal,
    get_conv1D,
    get_time_freq_matched_filter,
    get_morphology
)
from .base import BaseTransform
from ._utils import str2bool


class FeatureExtractor(BaseTransform):
    
    def __init__(self,
                 freq_bands: list[int] = [
                     [1, 2],
                     [2, 5],
                     [5, 10],
                     [10, 20],
                     [20, 30]
                 ],
                 freq_mode_threshold: int = 1,
                 feature_sets: list[str] = ['crafted', 'tsfel'],
                 add_imu_features: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        # Band energy feature for min, max in frequency bands
        self.freq_bands = freq_bands        
        # Main frequency mode above threshold (in Hz)
        self.freq_mode_threshold = freq_mode_threshold
        self.feature_sets = feature_sets

        allowed_feature_sets = {'crafted', 'tsfel'}
        assert set(feature_sets).issubset(allowed_feature_sets), "Only 'crafted' and 'tsfel' are allowed in features_sets"

        if 'tsfel' in feature_sets:
            domains = ['statistical', 'temporal', 'spectral']
            self.tsfel_cfg = tsfel.feature_extraction.features_settings.get_features_by_domain(domains)
        self.add_imu_features = add_imu_features
        assert len(self.freq_bands) == 5, "Only 5 band energy features supported"

    def _extract_features_of_signal(self,
                                    threshold,
                                    extracted_sensor_data,
                                    extracted_imu_acceleration,
                                    extracted_imu_rotation,
                                    mode: str = 'inference'):
        
        num_segments = len(extracted_sensor_data)

        num_common_feats = 53  # i.e. max, mean, sum, sd, percentile, skew, kurtosis, ...
        num_maps = self.num_sensors + 2 if str2bool(self.add_imu_features) else self.num_sensors
        # +1 feature for segment duration
        total_feats = num_maps * num_common_feats + 1
        
        # Features array
        X_extracted = np.zeros((num_segments, total_feats))
        columns = ['' for _ in range(total_feats)]
        self.logger.info(f"Extracting crafted features for '{mode}'...")

        # For each segment
        for i in tqdm(range(num_segments), desc=f"Processing '{mode}' segments"):
            # Duration of signal in seconds
            columns[0] = 'duration'
            X_extracted[i, 0] = len(extracted_sensor_data[i]) / self.sensor_freq

            for k in range(num_maps):
                
                # Assigning sensor data, IMU rotation, or acceleration based on the sensor index
                # For FM sensors, get thesholded signal
                if k < self.num_sensors:
                    signal = extracted_sensor_data[i][:, k]
                    signal_below_threshold = np.abs(signal) - threshold[k]
                    signal_above_threshold = signal_below_threshold[signal_below_threshold > 0]
                    sensor_name = f"snsr_{k+1}"                    
                # For IMU rotation/acceleration, no threshold adjustment needed
                else:
                    signal = extracted_imu_rotation[i] if k == self.num_sensors else extracted_imu_acceleration[i]
                    signal_below_threshold = signal_above_threshold = signal
                    sensor_name = 'imu_rot' if k == self.num_sensors else 'imu_accltn'
                
                # ------------ Time domain features ------------
                X_extracted[i, k*num_common_feats+1: k*num_common_feats+11], time_cols = self._calc_time_domain_feats(signal_below_threshold,
                                                                                                           signal_above_threshold)
                
                for c, col in enumerate(time_cols):
                    columns[k*num_common_feats+1+c] = f"{sensor_name}_{col}"
                
                # ------------ Frequency domain features ------------
                X_extracted[i, k*num_common_feats+11: k*num_common_feats+17], freq_cols = self._calc_freq_domain_feats(signal)
                for c, col in enumerate(freq_cols):
                    columns[k*num_common_feats+11+c] = f"{sensor_name}_{col}"
                
                # ------------ Features added by Omar ------------
                X_extracted[i, k*num_common_feats+17: k*num_common_feats+23], inst_cols = self._calc_instantenous_feats(signal,
                                                                                                             signal_below_threshold)
                for c, col in enumerate(inst_cols):
                    columns[k*num_common_feats+17+c] = f"{sensor_name}_{col}"

                # ------------ Wavelet features ------------
                X_extracted[i, k*num_common_feats+23] = np.mean(get_wavelet_coeff(signal))
                columns[k*num_common_feats+23] = f"{sensor_name}_mean_wavelet_coeff"
                X_extracted[i, k*num_common_feats+24] = np.median(get_wavelet_coeff(signal))
                columns[k*num_common_feats+24] = f"{sensor_name}_med_wavelet_coeff"
                X_extracted[i, k*num_common_feats+25] = np.std(get_wavelet_coeff(signal))
                columns[k*num_common_feats+25] = f"{sensor_name}_std_wavelet_coeff"
                
                # ------------ 1D convolutional features ------------
                convolved_signal = get_convolved_signal(signal_below_threshold)
                for conv_index in range(21):
                    X_extracted[i, k*num_common_feats+26+conv_index], col = get_conv1D(convolved_signal, conv_index + 1)
                    columns[k*num_common_feats+26+conv_index] = f"{sensor_name}_{col}_conv1D"
                    
                # ------------ Number of atoms feature ------------
                X_extracted[i, k*num_common_feats+47] = 1  # getMPDatom(S_MPDatom)
                columns[k*num_common_feats+47] = f"{sensor_name}_num_atoms"
                
                # ------------ Time frequency matched filter feature ------------
                X_extracted[i, k*num_common_feats+48] = get_time_freq_matched_filter(signal_below_threshold)
                columns[k*num_common_feats+48] = f"{sensor_name}_tfmf"
                
                # ------------ Features added by Monaf ------------                
                X_extracted[i, k*num_common_feats+49] = np.min(signal_below_threshold)
                columns[k*num_common_feats+49] = f"{sensor_name}_min_blw_thrsh"
                X_extracted[i, k*num_common_feats+50] = np.median(signal_below_threshold)
                columns[k*num_common_feats+50] = f"{sensor_name}_med_blw_thrsh"
                X_extracted[i, k*num_common_feats+51: k*num_common_feats+54], morph_cols = self._calc_morph_feats(signal_below_threshold)
                for c, col in enumerate(morph_cols):
                    columns[k*num_common_feats+51+c] = f"{sensor_name}_{col}"

        return X_extracted, columns
        
    def _extract_features_tsfel(self,
                                extracted_sensor_data,
                                extracted_imu_acceleration,
                                extracted_imu_rotation,
                                mode: str = 'inference'):
        
        num_segments = len(extracted_sensor_data)

        num_common_feats = 156  # i.e. num tsfel features per signal data
        num_maps = self.num_sensors + 2 if self.add_imu_features else self.num_sensors
        # +1 feature for segment duration
        total_feats = num_maps * num_common_feats

        # Features array
        X_extracted = np.zeros((num_segments, total_feats))
        columns = ['' for _ in range(total_feats)]
        self.logger.info(f"Extracting tsfel features for '{mode}'...")

        for i in tqdm(range(num_segments), desc=f"Processing '{mode}' segments"):

            for k in range(num_maps):
                
                # Assigning sensor data, IMU rotation, or acceleration based on the sensor index
                if k < self.num_sensors:
                    signal = extracted_sensor_data[i][:, k]
                    sensor_name = f"snsr_{k+1}"
                else:
                    signal = extracted_imu_rotation[i] if k == self.num_sensors else extracted_imu_acceleration[i]
                    sensor_name = 'imu_rot' if k == self.num_sensors else 'imu_accltn'
                
                try:
                    features = tsfel.time_series_features_extractor(self.tsfel_cfg, signal, fs=self.sensor_freq, n_jobs=-1, verbose=0)
                    X_extracted[i, k*num_common_feats: (k+1)*num_common_feats] = features.to_numpy().flatten()
                    for c in range(num_common_feats):
                        columns[k*num_common_feats+c] = f"{sensor_name}_tsfel_{c}"
                except Exception as e:
                    self.logger.debug(f"Exception occured during tsfel extraction: {e}")
                    continue
        
        return X_extracted, columns

    def _extract_features_for_inference(self,
                                        extracted_detections: dict,
                                        fm_dict: dict,                                        
                                        feat: str = 'crafted'):
        
        threshold = fm_dict['fm_threshold']        
        detections_sensor_data = extracted_detections['detections_sensor_data']
        detections_imu_acceleration = extracted_detections['detections_imu_acceleration']
        detections_imu_rotation = extracted_detections['detections_imu_rotation']

        if feat == 'crafted':
            X_extracted, columns = self._extract_features_of_signal(threshold, detections_sensor_data, 
                                                                    detections_imu_acceleration, detections_imu_rotation)
        if feat == 'tsfel':
            X_extracted, columns = self._extract_features_tsfel(detections_sensor_data, detections_imu_acceleration,
                                                                detections_imu_rotation)
        
        return {
            'features': X_extracted,
            'columns': columns
        }

    def _extract_features_for_train(self,
                                    extracted_detections: dict,
                                    fm_dict: dict,
                                    feat: str = 'crafted'):
        
        threshold = fm_dict['fm_threshold']

        tp_detections_indices = extracted_detections['tp_detections_indices']
        tp_detections_sensor_data = extracted_detections['tp_detections_sensor_data']
        tp_detections_imu_acceleration = extracted_detections['tp_detections_imu_acceleration']
        tp_detections_imu_rotation = extracted_detections['tp_detections_imu_rotation']
        
        fp_detections_indices = extracted_detections['fp_detections_indices']
        fp_detections_sensor_data = extracted_detections['fp_detections_sensor_data']
        fp_detections_imu_acceleration = extracted_detections['fp_detections_imu_acceleration']
        fp_detections_imu_rotation = extracted_detections['fp_detections_imu_rotation']

        if feat == 'crafted':
            X_tpd, columns = self._extract_features_of_signal(threshold, tp_detections_sensor_data, 
                                                                tp_detections_imu_acceleration, tp_detections_imu_rotation,
                                                                mode='TPD')
            X_fpd, columns = self._extract_features_of_signal(threshold, fp_detections_sensor_data, 
                                                                fp_detections_imu_acceleration, fp_detections_imu_rotation,
                                                                mode='FPD')
        if feat == 'tsfel':
            X_tpd, columns = self._extract_features_tsfel(tp_detections_sensor_data, 
                                                            tp_detections_imu_acceleration, tp_detections_imu_rotation,
                                                            mode='TPD')
            X_fpd, columns = self._extract_features_tsfel(fp_detections_sensor_data, 
                                                            fp_detections_imu_acceleration, fp_detections_imu_rotation,
                                                            mode='FPD')
        X_extracted = np.vstack([X_tpd, X_fpd])
        y_extracted = np.zeros((X_tpd.shape[0] + X_fpd.shape[0], 1))
        y_extracted[:X_tpd.shape[0], 0] = 1  # From here on, label 1 means TPD and label 0 means FPD
        y_extracted = np.ravel(y_extracted)
        det_indices = np.hstack([tp_detections_indices, fp_detections_indices])

        return {
            'features': X_extracted,
            'labels': y_extracted,
            'det_indices': det_indices,
            'columns': columns
        }
 
    def _calc_time_domain_feats(self, signal_below_thresh, signal_above_thresh):
        """Calculates the time domain features for a signal"""

        columns = ['max_blw_thrsh', 'mean_blw_thrsh', 'sd_blw_thrsh', 'iq_range', 'skew', 'kurtosis',
                   'duration_abv_thrsh', 'mean_abv_thrsh', 'engy_abv_thrsh']
        arr = [
            # Max value
            np.max(signal_below_thresh),
            # Mean value
            np.mean(signal_below_thresh),
            # Energy
            np.sum(signal_below_thresh ** 2),
            # Standard deviation
            np.std(signal_below_thresh),
            # Interquartile range
            np.percentile(signal_below_thresh, 75) - np.percentile(signal_below_thresh, 25),
            # Skewness
            skew(signal_below_thresh),
            # Kurtosis
            kurtosis(signal_below_thresh),
            # Duration above threshold
            len(signal_above_thresh),
            # Mean above threshold
            0 if len(signal_above_thresh) == 0 else np.mean(signal_above_thresh),
            # Energy above threshold
            0 if len(signal_above_thresh) == 0 else np.sum(signal_above_thresh ** 2)
        ]
        return arr, columns

    
    def _calc_freq_domain_feats(self, signal):
        """Calculates the frequency domain features for a signal"""

        columns = []
        arr = [
            get_frequency_mode(signal, self.sensor_freq, self.freq_mode_threshold)[-1]
        ]
        columns.append(f"freq_mode_{self.freq_mode_threshold}")
        for [min_freq, max_freq] in self.freq_bands:
            arr.append(get_band_energy(signal, self.sensor_freq, min_freq, max_freq))
            columns.append(f"band_{min_freq}_{max_freq}")
        return arr, columns
    
    def _calc_instantenous_feats(self, signal, signal_below_thresh):
        """Calculates the instantaneous amplitude and frequency features for a signal"""

        columns = ['mean_inst_amp', 'mean_inst_freq', 'std_inst_amp', 'std_inst_freq',
                   'max_inst_amp', 'max_inst_freq']
        arr = [
            np.mean(get_inst_amplitude(signal_below_thresh)),
            np.mean(get_inst_frequency(signal, self.sensor_freq)),
            np.std(get_inst_amplitude(signal_below_thresh)),
            np.std(get_inst_frequency(signal, self.sensor_freq)),
            np.max(get_inst_amplitude(signal_below_thresh)) - np.min(get_inst_amplitude(signal_below_thresh)),
            np.max(get_inst_frequency(signal, self.sensor_freq) - np.min(get_inst_frequency(signal, self.sensor_freq)))
        ]
        return arr, columns

    def _calc_morph_feats(self, signal_below_thresh):
        """Calculates the morphological features of a signal"""

        columns = ['abs_area', 'rel_area', 'abs_area_diff']
        absolute_area, relative_area, absolute_area_differential = get_morphology(signal_below_thresh,
                                                                                  self.sensor_freq)
        return [absolute_area, relative_area, absolute_area_differential], columns

    def decorator(method):
        """Decorator that checks the `inference` flag and calls the appropriate method."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            inference = kwargs.pop('inference', False)
            if inference:
                return self._extract_features_for_inference(*args, **kwargs)
            else:
                return self._extract_features_for_train(*args, **kwargs)
        return wrapper
    
    @decorator
    def transform(self):
        # Placeholder method for decorator function
        pass


        