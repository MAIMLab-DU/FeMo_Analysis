import logging
import numpy as np
from functools import wraps
from .config import SENSOR_MAP


class DetectionExtractor:
    @property
    def _logger(self):
        return logging.getLogger(__name__)
    
    def __init__(self, base_dir,
                 inferece: bool = True,
                 sensor_selection: list = ['accelerometer', 
                                           'piezoelectric_small', 
                                           'piezoelectric_large']) -> None:
        
        self._base_dir = base_dir
        self.inference = inferece
        self.sensor_selection = sensor_selection
        self.sensors = [item for s in self.sensor_selection for item in SENSOR_MAP[s]]
        self.num_sensors = len(self.sensors)

    def _extract_detections_for_inference(self, 
                           preprocessed_data: dict,
                           scheme_dict: dict):
        
        num_labels = scheme_dict['num_labels']
        labeled_user_scheme = scheme_dict['labeled_user_scheme']
        preprocessed_sensor_data = [preprocessed_data[key] for key in self.sensors]

        extracted_sensor_data = []
        extracted_imu_acceleration = []
        extracted_imu_rotation = []
        
        for i in range(1, num_labels + 1):
            label_start = np.where(labeled_user_scheme == i)[0][0]  # Sample no. corresponding to the start of the label
            label_end = np.where(labeled_user_scheme == i)[0][-1] + 1  # Sample no. corresponding to the end of the label
            
            extracted_data = np.zeros((label_end - label_start, self.num_sensors))
            
            for j in range(self.num_sensors):
                extracted_data[:, j] = preprocessed_sensor_data[j][label_start:label_end]
            
            extracted_sensor_data.append(extracted_data)
            extracted_imu_acceleration.append(preprocessed_data['imu_acceleration'][label_start:label_end])
            extracted_imu_rotation.append(preprocessed_data['imu_rotation_1D'][label_start:label_end])            
                
        return {
            'extracted_sensor_data': extracted_sensor_data,
            'extracted_imu_acceleration': extracted_imu_acceleration, 
            'extracted_imu_rotation': extracted_imu_rotation
        }
    
    # TODO: implement functionality
    def _extract_detections_for_train(self,
                                      preprocessed_data: dict,
                                      scheme_dict: dict):
        ...

    def decorator(method):
        """Decorator that checks the `inference` flag and calls the appropriate method."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self.inference:
                return self._extract_detections_for_inference(*args, **kwargs)
            else:
                return self._extract_detections_for_train(*args, **kwargs)
        return wrapper
    
    @decorator
    def extract_detections(self):
        pass


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
        
        extracted_sensor_data = extracted_detections['extracted_sensor_data']
        extracted_imu_acceleration = extracted_detections['extracted_imu_acceleration']
        extracted_imu_rotation = extracted_detections['extracted_imu_rotation']

        threshold = fm_dict['fm_threshold']
    
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