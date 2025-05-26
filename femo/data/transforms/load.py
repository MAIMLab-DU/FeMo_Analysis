import time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from .base import BaseTransform, FeMoData


class DataLoader(BaseTransform): 

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def transform(self, filename):
        start = time.time()
        self.logger.debug(f"Started Loading from file: {filename}")

        keys = [
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
            'sensor_5', 'sensor_6', 'imu_acceleration',
            'imu_rotation', 'sensation_data'
        ]
        loaded_data = {k: [] for k in keys}

        read_data = FeMoData(filename)
        self.logger.debug(f"DataFrames loaded from file: {list(read_data.dataframes.keys())}")

        all_sensor_df = (
            read_data.dataframes["piezos"]
            .join(read_data.dataframes["accelerometers"])
            .join(read_data.dataframes["imu"])
            .join(read_data.dataframes["force"])
            .join(read_data.dataframes["push_button"])
            .join(read_data.dataframes["timestamp"])
        )
        self.logger.debug(f"Combined sensor DataFrame shape: {all_sensor_df.shape}")

        # Interpolate missing values
        interp_columns = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2',
                        'rotation_r', 'rotation_i', 'rotation_j', 'rotation_k',
                        'accel_x', 'accel_y', 'accel_z',
                        'magnet_x', 'magnet_y', 'magnet_z']
        for col in interp_columns:
            all_sensor_df[col] = all_sensor_df[col].interpolate(method='linear', limit_direction='both')
        self.logger.debug(f"Interpolated columns: {interp_columns}")

        selected_data_columns = interp_columns + ['p1', 'p2', 'p3', 'p4', 'button']
        selected_sensor_data = all_sensor_df[selected_data_columns].copy()
        self.logger.debug(f"Selected sensor columns: {selected_data_columns}")
        self.logger.debug(f"Selected sensor data shape: {selected_sensor_data.shape}")

        # Convert raw sensor values to voltages
        max_sensor_value = 2**16 - 1
        max_voltage = 3.3
        for col in ['p1', 'p2', 'p3', 'p4', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']:
            selected_sensor_data[col] = (
                (selected_sensor_data[col].astype(np.float32) / max_sensor_value) * max_voltage
            )
        self.logger.debug(f"Sample converted voltage values: {selected_sensor_data[['p1', 'x1']].iloc[0].to_dict()}")

        IMU_all = selected_sensor_data[['rotation_r','rotation_i','rotation_j', 'rotation_k',
                                        'magnet_x','magnet_y','magnet_z',
                                        'accel_x','accel_y','accel_z']]

        IMU_aclm = IMU_all[['accel_x', 'accel_y', 'accel_z']] / 1000
        IMU_rotation_quat = IMU_all[['rotation_i', 'rotation_j', 'rotation_k', 'rotation_r']] / 10000

        # Normalize quaternions
        imu_rot_norm = np.linalg.norm(IMU_rotation_quat, axis=1, keepdims=True)
        imu_rot_norm[imu_rot_norm == 0] = 1
        IMU_rotation_quat /= imu_rot_norm
        self.logger.debug("Quaternion normalization completed")

        non_zero_row = IMU_rotation_quat[(IMU_rotation_quat != 0).any(axis=1)].iloc[0].tolist()
        IMU_rotation_quat.loc[(IMU_rotation_quat == 0).all(axis=1)] = non_zero_row

        rotation = R.from_quat(IMU_rotation_quat.values)
        IMU_rotation_rpy = pd.DataFrame(rotation.as_euler('xyz', degrees=True), columns=['roll', 'pitch', 'yaw'])
        self.logger.debug(f"Euler angle conversion done. Sample: {IMU_rotation_rpy.iloc[0].to_dict()}")
        
        loaded_data['sensor_1'] = np.linalg.norm(selected_sensor_data[['x1', 'y1', 'z1']], axis=1)
        loaded_data['sensor_2'] = np.linalg.norm(selected_sensor_data[['x2', 'y2', 'z2']], axis=1)
        loaded_data['sensor_3'] = selected_sensor_data['p1'].to_numpy()
        loaded_data['sensor_4'] = selected_sensor_data['p2'].to_numpy()
        loaded_data['sensor_5'] = selected_sensor_data['p3'].to_numpy()
        loaded_data['sensor_6'] = selected_sensor_data['p4'].to_numpy()
        loaded_data['imu_acceleration'] = np.linalg.norm(IMU_aclm, axis=1)
        loaded_data['imu_rotation'] = IMU_rotation_rpy
        loaded_data['fm_acceleration'] = {
            'sensor_1': pd.DataFrame(
                selected_sensor_data[['x1', 'y1', 'z1']].to_numpy(),
                columns=['x', 'y', 'z']
            ),
            'sensor_2': pd.DataFrame(
                selected_sensor_data[['x2', 'y2', 'z2']].to_numpy(),
                columns=['x', 'y', 'z']
            )
        }
        self.logger.debug("Sensor magnitudes computed for all FM channels and IMU acceleration.")

        try:
            loaded_data['sensation_data'] = selected_sensor_data['button'].to_numpy()
            self.logger.debug(f"Sensation data loaded. Length: {len(loaded_data['sensation_data'])}")
        except KeyError:
            loaded_data['sensation_data'] = np.array([])
            self.logger.warning("Button column not found in dataset. Using empty sensation_data array.")

        duration_ms = (time.time() - start) * 1000
        self.logger.info(f"Data file '{filename}' loaded in {duration_ms:.2f} ms")

        return loaded_data

    