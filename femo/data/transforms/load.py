import time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from .base import BaseTransform, FeMo


class DataLoader(BaseTransform): 

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def transform(self, filename):

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

        selected_data_columns = ['p1', 'p2', 'p3', 'p4', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'rotation_r', 'rotation_i', 
                                 'rotation_j', 'rotation_k', 'magnet_x','magnet_y','magnet_z', 'accel_x', 'accel_y', 'accel_z', 'button']
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

        try:
            # Get maternal sensation (for training)
            loaded_data['sensation_data'] = np.array(selected_sensor_data['button'])
        except KeyError:
            loaded_data['sensation_data'] = np.array([])

        self.logger.debug(f"Loaded data file {filename} in {(time.time()-start)*1000} ms")

        return loaded_data

    