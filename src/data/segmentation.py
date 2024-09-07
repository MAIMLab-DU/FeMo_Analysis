import time
import numpy as np
import logging
import pandas as pd
import concurrent.futures
from skimage.measure import label
from functools import reduce
from .config import SENSOR_MAP
from .utils import (
    custom_binary_dilation
)


class DataSegmentor:
    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def __init__(self, base_dir,  
                 sensor_freq: int = 1024,
                 imu_acceleration_threshold: float = 0.2,
                 imu_rotation_threshold: int = 4,
                 imu_dilation: int = 4,
                 fm_dilation: int = 3,
                 fm_min_sn: list = 40,
                 fm_signal_cutoff: list = 0.0001,
                 sensor_selection: list = ['accelerometer', 
                                           'piezoelectric_small', 
                                           'piezoelectric_large']) -> None:
        
        self._base_dir = base_dir
        self.sensor_freq = sensor_freq
        self.imu_acceleration_threshold = imu_acceleration_threshold
        self.imu_rotation_threshold = imu_rotation_threshold
        self.imu_dilation = imu_dilation
        self.fm_dilation = fm_dilation
        self.sensor_selection = sensor_selection
        
        self.sensors = [item for s in self.sensor_selection for item in SENSOR_MAP[s]]
        self.num_sensors = len(self.sensors)
        self.fm_min_sn = [fm_min_sn for _ in range(self.num_sensors)]
        self.segmentation_signal_cutoff = [fm_signal_cutoff for _ in range(self.num_sensors)]
        # Dilation length in seconds
        self.imu_dilation_size = round(imu_dilation * sensor_freq)  # Dilation length in sample number 
        self.fm_dilation_size = round(fm_dilation * sensor_freq)  # Dilation length in sample number
    
    def _create_imu_accleration_map(self, imu_accleration):
    
        imu_acceleration_map = np.abs(imu_accleration) >= self.imu_acceleration_threshold

        # ----------------------Dilation of IMU data-------------------------------
        # Dilate or expand the ROI's(points with value = 1) by dilation_size (half above and half below), as defined by SE
        imu_acceleration_map = custom_binary_dilation(imu_acceleration_map, self.imu_dilation_size)

        return imu_acceleration_map

    def _create_imu_rotation_map(self, imu_rotation):

        # Take the absolute values of the extracted Euler angles
        imu_rotR_preproc = np.abs(imu_rotation['roll'].values)
        imu_rotP_preproc = np.abs(imu_rotation['pitch'].values)
        imu_rotY_preproc = np.abs(imu_rotation['yaw'].values)
    
        # Initialize the IMU map with zeros (False)
        imu_rotation_map = np.zeros_like(imu_rotP_preproc, dtype=bool)
    
        # Apply new conditions to modify the IMU map
        for i in range(len(imu_rotP_preproc)):
            if imu_rotP_preproc[i] < 1.5:
                if imu_rotR_preproc[i] < 4 and imu_rotY_preproc[i] < 4:
                    imu_rotation_map[i] = False  
                else:
                    imu_rotation_map[i] = True
            else:  # IMU_rotP_preproc[i] >= 1
                if imu_rotR_preproc[i] > 4 or imu_rotY_preproc[i] > 4:
                    imu_rotation_map[i] = True  
    
        # Dilation of IMU data
        imu_rotation_map = custom_binary_dilation(imu_rotation_map, self.imu_dilation_size)
    
        return imu_rotation_map
    
    def create_imu_map(self, preprocessed_data: dict):

        tic = time.time()
        map_added = self._create_imu_accleration_map(preprocessed_data['imu_acceleration']).astype(int) \
                    + self._create_imu_rotation_map(preprocessed_data['imu_rotation']).astype(int)
    
        map_added[0] = 0
        map_added[-1] = 0
        
        # Find where changes from non-zero to zero or zero to non-zero occur
        changes = np.where((map_added[:-1] == 0) != (map_added[1:] == 0))  [0] + 1


        # Create tuples of every two values
        windows = []
        for i in range(0, len(changes), 2):
            if i < len(changes) - 1:
                windows.append((changes[i], changes[i+1]))
            else:
                windows.append((changes[i],))
        
        map_final = np.zeros(len(map_added))
        
        for window in windows:
            start = window[0]
            
            if len(window) == 2:
                end = window[1]
                if np.any(map_added[start:end] == 2):
                    map_final[start:end] = 1
            else:
                map_final[start:] = 1        

        self._logger.info(f"IMU rotation map created in {(time.time()-tic)*1000:.2f} ms")                
        return map_final.astype(dtype=bool)
    
    def _get_segmented_sensor_data(self, preprocessed_sensor_data, imu_map):  

        low_signal_quantile = 0.25
        SE = np.ones((self.fm_dilation_size + 1))  # linear element necessary for dilation operation

        h = np.zeros(self.num_sensors)  # Variable for threshold
        segmented_sensor_data = [None] * self.num_sensors
        
        def getBinaryDialation(index):
            # Determining the threshold
            s = np.abs(preprocessed_sensor_data[index])
            LQ = np.quantile(s, low_signal_quantile)  # Returns the quantile value for low 25% (= low_signal_quantile) of the signal
            e = s[s <= LQ]  # Signal noise
            h[index] = self.fm_min_sn[index] * np.median(e)  # Threshold value. Each row will contain the threshold value for each data file

            if np.isnan(h[index]):  # Check if h = NaN. This happens when e=[], as median(e)= NaN for that case!!!
                h[index] = np.inf
            if h[index] < self.segmentation_signal_cutoff[index]:
                h[index] = np.inf  # Precaution against too noisy signal

            # Thresholding
            each_sensor_data_sgmntd = (s >= h[index]).astype(int)  # Considering the signals that are above the threshold value; data point for noise signal becomes zero and signal becomes 1

            # Exclusion of body movement data
            each_sensor_data_sgmntd *= (1 - imu_map)  # Exclusion of body movement data            
            
            # return  (binary_dilation(each_sensor_data_sgmntd, structure=SE), index)
            return (custom_binary_dilation(each_sensor_data_sgmntd, self.fm_dilation_size + 1), index)
        
        # Create a ThreadPoolExecutor with a specified number of threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_sensors) as executor:
            # Submit the tasks to the executor and store the Future objects
            futures = [executor.submit(getBinaryDialation, j) for j in range(self.num_sensors)]        
            # Retrieve the results as they become available
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
        for r in results:
            segmented_sensor_data[r[1]] = r[0]

        return segmented_sensor_data, h
    
    def create_fm_map(self, preprocessed_data: dict, imu_map=None):
        
        if imu_map is None:
            imu_map = self.create_imu_map(preprocessed_data)
        preprocessed_sensor_data = [preprocessed_data[key] for key in self.sensors]
        
        segmented_sensor_data, threshold = self._get_segmented_sensor_data(preprocessed_sensor_data, 
                                                                           imu_map)
        print(f"{self.num_sensors = }, {len(segmented_sensor_data) = }")

        for i in range(self.num_sensors):
            map_added = imu_map.astype(int) + segmented_sensor_data[i].astype(int)
    
            map_added[0] = 0
            map_added[-1] = 0
            
            # Find where changes from non-zero to zero or zero to non-zero occur
            changes = np.where((map_added[:-1] == 0) != (map_added[1:] == 0))  [0] + 1

            # Create tuples of every two values
            windows = []
            for j in range(0, len(changes), 2):
                if j < len(changes) - 1:
                    windows.append((changes[j], changes[j+1]))
                else:
                    windows.append((changes[j],))
            
            map_final = np.copy(segmented_sensor_data[i])
            
            for window in windows:
                start = window[0]
                
                if len(window) == 2:
                    end = window[1]
                    if np.any(map_added[start:end] == 2):
                        map_final[start:end] = 0
                else:
                    map_final[start:] = 1
                    
            map_final[0] = 0     
            map_final[-1] = 0

            segmented_sensor_data[i] = map_final.astype(int)
            
        combined_fm_map = reduce(lambda x, y: x | y, (data for data in segmented_sensor_data)) 

        return {
            'fm_map': combined_fm_map,  # sensor_data_sgmntd_cmbd_all_sensors
            'fm_threshold': threshold,  # h
            'fm_segmented': segmented_sensor_data  # sensor_data_sgmntd
        }





