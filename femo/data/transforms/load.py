import time
import numpy as np
import pandas as pd
from dataclasses import asdict
from ._utils import timestamp_to_iso
from scipy.spatial.transform import Rotation as R
from .base import BaseTransform, FeMoData


class DataLoader(BaseTransform): 

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def transform(self, filename):
        """
        Load a .dat file and return a dict with:
        - sensor_1…sensor_6: FM channel magnitudes or voltages (float32)
        - imu_acceleration:    interpolated & exact L2‐norm (float32)
        - imu_rotation:        interpolated & exact Euler angles DataFrame
        - sensation_data:      button presses (int8)
        """

        start = time.time()
        self.logger.debug(f"Started loading from file: {filename}")

        # 1) Read raw arrays and header
        fe  = FeMoData(filename)
        raw = fe._arrays
        header = asdict(fe.header)
        header['start_time'] = timestamp_to_iso(header['start_time'])
        header['end_time'] = timestamp_to_iso(header['end_time'])  

        # 2) Pre‐allocate outputs
        N   = raw['button'].shape[0]
        sensor_1         = np.zeros(N, dtype=np.float32)
        sensor_2         = np.zeros(N, dtype=np.float32)
        sensor_3         = np.zeros(N, dtype=np.float32)
        sensor_4         = np.zeros(N, dtype=np.float32)
        sensor_5         = np.zeros(N, dtype=np.float32)
        sensor_6         = np.zeros(N, dtype=np.float32)
        imu_acceleration = np.zeros(N, dtype=np.float32)
        imu_rotation_arr = np.zeros((N, 3), dtype=np.float32)
        sensation_data   = raw['button'][:, 1].astype(np.int8)

        # 3) Scaling factors
        scale_voltage = 3.3 / (2**16 - 1)
        scale_accel   = 1.0 / 1000.0
        scale_quat    = 1.0 / 10000.0

        # 4) Sensor magnitudes & voltages
        idx_acc    = raw['accel'][:, 0].astype(int)
        accel_vals = raw['accel'][:, 1:].astype(np.float32) * scale_voltage
        sensor_1[idx_acc] = np.linalg.norm(accel_vals[:, 0:3], axis=1)
        sensor_2[idx_acc] = np.linalg.norm(accel_vals[:, 3:6], axis=1)

        idx_pz  = raw['piezo'][:, 0].astype(int)
        pz_vals = raw['piezo'][:, 1:].astype(np.float32) * scale_voltage
        sensor_3[idx_pz] = pz_vals[:, 0]
        sensor_4[idx_pz] = pz_vals[:, 1]
        sensor_5[idx_pz] = pz_vals[:, 2]
        sensor_6[idx_pz] = pz_vals[:, 3]

        # 5) Prepare IMU raw data for interpolation (use float64 for accuracy)
        idx_imu  = raw['imu'][:, 0].astype(int)
        imu_vals = raw['imu'][:, 1:].astype(np.float64)

        # 6) Per‐axis interpolation, then norm
        t_full = np.arange(N)
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        az = np.zeros(N, dtype=np.float64)
        # accel_x,y,z are in cols 7,8,9
        ax[idx_imu] = imu_vals[:, 7] * scale_accel
        ay[idx_imu] = imu_vals[:, 8] * scale_accel
        az[idx_imu] = imu_vals[:, 9] * scale_accel

        ax = np.interp(t_full, idx_imu, ax[idx_imu])
        ay = np.interp(t_full, idx_imu, ay[idx_imu])
        az = np.interp(t_full, idx_imu, az[idx_imu])

        imu_acc = np.sqrt(ax*ax + ay*ay + az*az)
        imu_acceleration[:] = imu_acc.astype(np.float32)

        # 7) Per‐component quaternion interpolation, then normalize → Euler
        qi = np.zeros(N, dtype=np.float64)
        qj = np.zeros(N, dtype=np.float64)
        qk = np.zeros(N, dtype=np.float64)
        qr = np.zeros(N, dtype=np.float64)
        # quaternion in cols [1,2,3,0]
        qi[idx_imu] = imu_vals[:, 1] * scale_quat
        qj[idx_imu] = imu_vals[:, 2] * scale_quat
        qk[idx_imu] = imu_vals[:, 3] * scale_quat
        qr[idx_imu] = imu_vals[:, 0] * scale_quat

        qi = np.interp(t_full, idx_imu, qi[idx_imu])
        qj = np.interp(t_full, idx_imu, qj[idx_imu])
        qk = np.interp(t_full, idx_imu, qk[idx_imu])
        qr = np.interp(t_full, idx_imu, qr[idx_imu])

        quat_full = np.stack([qi, qj, qk, qr], axis=1)
        norms = np.linalg.norm(quat_full, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        quat_full /= norms

        zero_mask = np.all(quat_full == 0, axis=1)
        if zero_mask.any():
            first_valid = quat_full[~zero_mask][0]
            quat_full[zero_mask] = first_valid

        euler = R.from_quat(quat_full).as_euler('xyz', degrees=True)
        imu_rotation_arr[:] = euler.astype(np.float32)

        # 8) Forward‐fill any initial zeros in rotation
        valid = np.any(imu_rotation_arr != 0, axis=1)
        if valid.any():
            first = np.argmax(valid)
            imu_rotation_arr[:first, :] = imu_rotation_arr[first, :]
            for i in range(first + 1, N):
                if not valid[i]:
                    imu_rotation_arr[i] = imu_rotation_arr[i - 1]

        imu_rotation_df = pd.DataFrame(imu_rotation_arr, columns=['roll', 'pitch', 'yaw'])

        # 9) Bundle outputs
        loaded_data = {
            'sensor_1':         sensor_1,
            'sensor_2':         sensor_2,
            'sensor_3':         sensor_3,
            'sensor_4':         sensor_4,
            'sensor_5':         sensor_5,
            'sensor_6':         sensor_6,
            'imu_acceleration': imu_acceleration,
            'imu_rotation':     imu_rotation_df,
            'sensation_data':   sensation_data,
            'header':           header
        }

        duration_ms = (time.time() - start) * 1e3
        self.logger.info(f"Loaded '{filename}' in {duration_ms:.2f} ms")
        return loaded_data
