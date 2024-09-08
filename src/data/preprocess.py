import time
import logging
import pathlib
import boto3
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, sosfiltfilt
from .femo import FeMo
from .utils import apply_pca


class DataPreprocessor: 
    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def __init__(self, base_dir, 
                 sensor_freq = 1024) -> None:
        self._base_dir = base_dir
        self.sensor_freq = sensor_freq

    def download_file(self, bucket, key):
        pathlib.Path(f"{self._base_dir}/data").mkdir(parents=True, exist_ok=True)

        self._logger.info(f"Downloading data from bucket: {bucket}, key: {key}")
        fn = f"{self._base_dir}/data/{key}.dat"
        s3 = boto3.resource("s3")
        s3.Bucket(bucket).download_file(key, fn)

        return fn
    
    def load_data_file(self, filename):

        start = time.time()
        keys = [
            'sensor_1',
            'sensor_2',
            'sensor_3',
            'sensor_4',
            'sensor_5',
            'sensor_6',
            'imu_acceleration',
            'imu_rotation',
            'sensation_data'
        ]
        loaded_data = {
            k:[] for k in keys
        }  
                
        read_data = FeMo(filename)
        all_sensor_df = (read_data.dataframes["piezos"]
                        .join(read_data.dataframes["accelerometers"])
                        .join(read_data.dataframes["imu"])
                        .join(read_data.dataframes["force"])
                        .join(read_data.dataframes["push_button"])
                        .join(read_data.dataframes["timestamp"]))

        # Resample accelerometer data using linear interpolation

        ### Accelerometer 1
        all_sensor_df['x1'] = pd.Series(all_sensor_df['x1']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['y1'] = pd.Series(all_sensor_df['y1']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['z1'] = pd.Series(all_sensor_df['z1']).interpolate(method='linear', limit_direction='both')

        ### Accelerometer 2
        all_sensor_df['x2'] = pd.Series(all_sensor_df['x2']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['y2'] = pd.Series(all_sensor_df['y2']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['z2'] = pd.Series(all_sensor_df['z2']).interpolate(method='linear', limit_direction='both')

        ### IMU_data
        all_sensor_df['rotation_r'] = pd.Series(all_sensor_df['rotation_r']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['rotation_i'] = pd.Series(all_sensor_df['rotation_i']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['rotation_j'] = pd.Series(all_sensor_df['rotation_j']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['rotation_k'] = pd.Series(all_sensor_df['rotation_k']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['accel_x']    = pd.Series(all_sensor_df['accel_x']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['accel_y']    = pd.Series(all_sensor_df['accel_y']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['accel_z']    = pd.Series(all_sensor_df['accel_z']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['magnet_x']   = pd.Series(all_sensor_df['magnet_x']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['magnet_y']   = pd.Series(all_sensor_df['magnet_y']).interpolate(method='linear', limit_direction='both')
        all_sensor_df['magnet_z']   = pd.Series(all_sensor_df['magnet_z']).interpolate(method='linear', limit_direction='both')

        selected_data_columns = ['p1', 'p2', 'p3', 'p4', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'rotation_r', 'rotation_i', 'rotation_j', 'rotation_k', 'magnet_x','magnet_y','magnet_z', 'accel_x', 'accel_y', 'accel_z', 'button']
        selected_sensor_data = all_sensor_df[selected_data_columns]
        
        FM_sensor_columns = ['p1', 'p2', 'p3', 'p4', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
        
        # Convert them to voltage value
        # All sensor is 16 bit data and max voltage is 3.3v
        # Voltage = (Raw data / 2^ADC resolution) * max_voltage

        max_sensor_value = 2**16 - 1  
        max_voltage = 3.3  

        selected_sensor_data = selected_sensor_data.copy()
        for column in selected_sensor_data.columns:
            if column in FM_sensor_columns:
                selected_sensor_data.loc[:, column] = (selected_sensor_data[column] / max_sensor_value) * max_voltage

        # •	contains the rotation vector, which is the most accurate position (based on magnetometer, accelerometer and gyroscope)
        # •	rotation vector is currently in quaternion format
        # •	rotation vector originally is a float variable, it is stored as int16 with the following conversion: 
        #       rounded (original_float_value x 10000)
        # •	Magnetic vector is originally in Microtesla
        #       Magnetic vector is stored as int16: rounded (original_float x 100)
        # •	Linear acceleration is originally in m/s^2 (gravity excluded)
        #       Linear acceleration is stored as int16: rounded (original_float_value x 1000)
        
        IMU_all = selected_sensor_data[['rotation_r','rotation_i','rotation_j', 'rotation_k',
                                            'magnet_x','magnet_y','magnet_z',
                                            'accel_x','accel_y','accel_z']]
        
        IMU_aclm_single_file    = (IMU_all[['accel_x', 'accel_y', 'accel_z']]/1000)
        IMU_mag_single_file     = (IMU_all[['magnet_x', 'magnet_y', 'magnet_z']]/100)  # noqa: F841
        IMU_rotation_quat       = (IMU_all[['rotation_i', 'rotation_j', 'rotation_k', 'rotation_r']]/10000)  
        
        # --------- Quaternion to euler conversion is not possible with zero-magnitude rows -----
        # --------- Converting zero-magnitude rows with the first nonzero-magnitude row values --
        # Find the first non-zero orientation row
        non_zero_row = IMU_rotation_quat[(IMU_rotation_quat != 0).any(axis=1)].iloc[0].tolist()
        # Replace rows with all zeros with first valid nonzero-magnitude row
        IMU_rotation_quat.loc[(IMU_rotation_quat == 0).all(axis=1)] = non_zero_row
        
        IMU_rotation_quat = IMU_rotation_quat.values  # convert to numpy array        
        
        rotation = R.from_quat(IMU_rotation_quat)
        IMU_rotation_rpy = rotation.as_euler('xyz', degrees=True)                
        IMU_rotation_rpy = pd.DataFrame(IMU_rotation_rpy, columns=['roll', 'pitch', 'yaw'])  

        # Calculate magnitude values for FM accelerometer data
        loaded_data['sensor_1'] = np.linalg.norm(selected_sensor_data[['x1', 'y1', 'z1' ]], axis=1)
        loaded_data['sensor_2'] = np.linalg.norm(selected_sensor_data[['x2', 'y2', 'z2' ]], axis=1)
        
        # Calculate magnitude values for FM piezoelectric data
        loaded_data['sensor_3'] = np.array(selected_sensor_data['p1'])
        loaded_data['sensor_4'] = np.array(selected_sensor_data['p2'])
        loaded_data['sensor_5'] = np.array(selected_sensor_data['p3'])
        loaded_data['sensor_6'] = np.array(selected_sensor_data['p4'])

        # Calculate magnitude values for IMU accelerometers
        loaded_data['imu_acceleration'] = np.linalg.norm(IMU_aclm_single_file[['accel_x', 'accel_y', 'accel_z']], axis=1)
        loaded_data['imu_rotation'] = IMU_rotation_rpy # Rotation data is not combined
         
        # New data do not have flexi data, so we have passed a blank array for flexi data to keep the same format of return values
        # loaded_data['flexi_data'] = np.zeros_like(loaded_data['sensor_3'])

        #Get maternal sensation 
        loaded_data['sensation_data'] = np.array(selected_sensor_data['button'])
        self._logger.debug(f"Loaded data file {filename} in {(time.time()-start)*1000} ms")

        return loaded_data
    
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
        preprocessed_data['sensation_data'] = preprocessed_data['sensation_data'][start_index:end_index]

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