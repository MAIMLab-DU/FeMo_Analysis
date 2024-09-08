import time
import logging
import numpy as np
from scipy.signal import butter, sosfiltfilt
from .utils import apply_pca


class DataPreprocessor: 
    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def __init__(self,
                 base_dir,
                 sensor_freq: int = 1024) -> None:
        
        self._base_dir = base_dir
        self.sensor_freq = sensor_freq
    
    def preprocess_data(self, loaded_data):
        start = time.time()
        preprocessed_data = loaded_data.copy()
        
        # ---------------------------- Filter design -----------------------------#
        # 3 types of filters are designed-
        # bandpass filter, low-pass filter, and IIR notch filter.

        # TODO: Add settings file
        # Filter setting
        filter_order = 10
        lowCutoff_FM = 1
        highCutoff_FM = 30
        lowCutoff_IMU = 1
        highCutoff_IMU = 10
        highCutoff_force = 10  # noqa: F841
            
        if len(loaded_data['sensor_1']) > self.sensor_freq*5*60:  # If greater than 5 minutes remove last and first 30 seconds
            removal_period = 30  # Removal period in seconds   
        else:  # Else remove just 5 seconds
            removal_period = 5  # Removal period in seconds

        self._logger.debug(f"Filter order: {filter_order:.1f}")
        self._logger.debug(f"IMU band-pass: {lowCutoff_IMU}-{highCutoff_IMU} Hz")
        self._logger.debug(f"FM band-pass: {lowCutoff_FM}-{highCutoff_FM} Hz")
        # self._logger.debug(f'\tForce sensor low-pass: {highCutoff_force} Hz')
        self._logger.debug(f"Removal period: {removal_period} s")



        # TODO: Add descriptive comments to docstrings or wiki
        # ================Bandpass filter design==========================
        #   A band-pass filter with a passband of 1-20 Hz is disigned for the fetal fetal movement data
        #   Another band-pass filer with a passband of 1-10 Hz is designed for the IMU data

        # ========SOS-based design
        # Get second-order sections form
        sos_FM  = butter(filter_order / 2, np.array([lowCutoff_FM, highCutoff_FM]) / (self.sensor_freq / 2), 'bandpass', output='sos')  # filter order for bandpass filter is twice the value of 1st parameter
        sos_IMU = butter(filter_order / 2, np.array([lowCutoff_IMU, highCutoff_IMU]) / (self.sensor_freq / 2), 'bandpass', output='sos')
        
        # ========Zero-Pole-Gain-based design
        # z_FM, p_FM, k_FM = butter(filter_order / 2, np.array([lowCutoff_FM, highCutoff_FM]) / (sensor_freq / 2), 'bandpass', output='zpk')# filter order for bandpass filter is twice the value of 1st parameter
        # sos_FM, g_FM = zpk2sos(z_FM, p_FM, k_FM) #Convert zero-pole-gain filter parameters to second-order sections form
        # z_IMU,p_IMU,k_IMU = butter(filter_order / 2, np.array([lowCutoff_IMU, highCutoff_IMU]) / (sensor_freq / 2), 'bandpass', output='zpk')
        # sos_IMU, g_IMU = zpk2sos(z_IMU,p_IMU,k_IMU)
        
        # ========Transfer function-based design
        # Numerator (b) and denominator (a) polynomials of the IIR filter
        # b_FM,a_FM   = butter(filter_order/2,np.array([lowCutoff_FM, highCutoff_FM])/(sensor_freq/2),'bandpass', output='ba')# filter order for bandpass filter is twice the value of 1st parameter
        # b_IMU,a_IMU = butter(filter_order/2,np.array([lowCutoff_IMU, highCutoff_IMU])/(sensor_freq/2),'bandpass', output='ba')
        

        # -----------------------Data filtering--------------------------------
        
        # Bandpass filtering
        preprocessed_data['sensor_1']                = sosfiltfilt(sos_FM,  preprocessed_data['sensor_1'])
        preprocessed_data['sensor_2']                = sosfiltfilt(sos_FM,  preprocessed_data['sensor_2'])
        preprocessed_data['sensor_3']                = sosfiltfilt(sos_FM,  preprocessed_data['sensor_3'])
        preprocessed_data['sensor_4']                = sosfiltfilt(sos_FM,  preprocessed_data['sensor_4'])
        preprocessed_data['sensor_5']                = sosfiltfilt(sos_FM,  preprocessed_data['sensor_5'])
        preprocessed_data['sensor_6']                = sosfiltfilt(sos_FM,  preprocessed_data['sensor_6'])
        preprocessed_data['imu_acceleration']        = sosfiltfilt(sos_IMU, preprocessed_data['imu_acceleration'])
        preprocessed_data['imu_rotation']['roll']    = sosfiltfilt(sos_IMU, preprocessed_data['imu_rotation']['roll'].values)
        preprocessed_data['imu_rotation']['pitch']   = sosfiltfilt(sos_IMU, preprocessed_data['imu_rotation']['pitch'].values)
        preprocessed_data['imu_rotation']['yaw']     = sosfiltfilt(sos_IMU, preprocessed_data['imu_rotation']['yaw'].values)

        # -----------------------Data trimming---------------------------------
        
        # Trimming of raw data
        start_index = removal_period * self.sensor_freq
        end_index = -removal_period * self.sensor_freq
        try:
            preprocessed_data['sensation_data'] = preprocessed_data['sensation_data'][start_index:end_index]
        except IndexError:
            preprocessed_data['sensation_data'] = preprocessed_data['sensation_data']

        # Trimming of filtered data
        preprocessed_data['sensor_1']                = preprocessed_data['sensor_1'][start_index:end_index]
        preprocessed_data['sensor_2']                = preprocessed_data['sensor_2'][start_index:end_index]
        preprocessed_data['sensor_3']                = preprocessed_data['sensor_3'][start_index:end_index]
        preprocessed_data['sensor_4']                = preprocessed_data['sensor_4'][start_index:end_index]
        preprocessed_data['sensor_5']                = preprocessed_data['sensor_5'][start_index:end_index]
        preprocessed_data['sensor_6']                = preprocessed_data['sensor_6'][start_index:end_index]
        preprocessed_data['imu_acceleration']        = preprocessed_data['imu_acceleration'][start_index:end_index]
        preprocessed_data['imu_rotation']            = preprocessed_data['imu_rotation'].iloc[start_index:end_index]           

        preprocessed_data['imu_rotation_1D'] = apply_pca(preprocessed_data['imu_rotation'])
        self._logger.debug(f"Data preprocessed in {(time.time()-start)*1000} ms")

        return preprocessed_data