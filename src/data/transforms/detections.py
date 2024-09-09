import numpy as np
from functools import wraps
from .base import BaseTransform


class DetectionExtractor(BaseTransform):
    
    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)

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
            inference = kwargs.pop('inference', False)
            if inference:
                return self._extract_detections_for_inference(*args, **kwargs)
            else:
                return self._extract_detections_for_train(*args, **kwargs)
        return wrapper
    
    @decorator
    def transform(self):
        pass


