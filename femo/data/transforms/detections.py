import numpy as np
from functools import wraps
from .base import BaseTransform


class DetectionExtractor(BaseTransform):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _extract_detections_for_inference(self,
                                          preprocessed_data: dict,
                                          scheme_dict: dict):
        
        num_dets = scheme_dict['num_labels']
        labeled_user_scheme = scheme_dict['labeled_user_scheme']
        preprocessed_sensor_data = [preprocessed_data[key] for key in self.sensors]

        detections_sensor_data = []
        detections_imu_acceleration = []
        detections_imu_rotation = []
        
        for i in range(1, num_dets + 1):
            label_start = np.where(labeled_user_scheme == i)[0][0]  # Sample no. corresponding to the start of the label
            label_end = np.where(labeled_user_scheme == i)[0][-1] + 1  # Sample no. corresponding to the end of the label
            
            detections_data = np.zeros((label_end - label_start, self.num_sensors))
            
            for j in range(self.num_sensors):
                detections_data[:, j] = preprocessed_sensor_data[j][label_start:label_end]
            
            detections_sensor_data.append(detections_data)
            detections_imu_acceleration.append(preprocessed_data['imu_acceleration'][label_start:label_end])
            detections_imu_rotation.append(preprocessed_data['imu_rotation_1D'][label_start:label_end])            
                
        return {
            'detections_sensor_data': detections_sensor_data,
            'detections_imu_acceleration': detections_imu_acceleration, 
            'detections_imu_rotation': detections_imu_rotation
        }

    def _extract_detections_for_train(self,
                                      preprocessed_data: dict,
                                      scheme_dict: dict,
                                      sensation_map: np.ndarray):
        
        num_dets = scheme_dict['num_labels']
        labeled_user_scheme = scheme_dict['labeled_user_scheme']
        num_tp_dets = len(np.unique(labeled_user_scheme * sensation_map)) - 1
        num_dets - num_tp_dets
        preprocessed_sensor_data = [preprocessed_data[key] for key in self.sensors]

        tp_detections_indices = []
        tp_detections_sensor_data = []
        tp_detections_imu_acceleration = []
        tp_detections_imu_rotation = []

        fp_detections_indices = []
        fp_detections_sensor_data = []
        fp_detections_imu_acceleration = []
        fp_detections_imu_rotation = []
        
        for i in range(1, num_dets + 1):
            label_start = np.where(labeled_user_scheme == i)[0][0]  # Sample no. corresponding to the start of the label
            label_end = np.where(labeled_user_scheme == i)[0][-1] + 1  # Sample no. corresponding to the end of the label

            indv_window = np.zeros(len(sensation_map))
            indv_window[label_start:label_end] = 1

            # Used to calculate 'weightage', unused in legacy code
            if self.use_all_sensors:
                pass

            detections_data = np.zeros((label_end - label_start, self.num_sensors))
            for j in range(self.num_sensors):
                detections_data[:, j] = preprocessed_sensor_data[j][label_start:label_end]

            # Check overlap with sensation map, non-zero values indicate overlap
            intersection = np.sum(indv_window * sensation_map)
            if intersection:
                tp_detections_indices.append(i)  # Index of the detected label in the labeled_user_scheme array
                tp_detections_sensor_data.append(detections_data)
                tp_detections_imu_acceleration.append(preprocessed_data['imu_acceleration'][label_start:label_end])
                tp_detections_imu_rotation.append(preprocessed_data['imu_rotation_1D'][label_start:label_end])
            else:
                fp_detections_indices.append(i)  # Index of the detected label in the labeled_user_scheme array
                fp_detections_sensor_data.append(detections_data)
                fp_detections_imu_acceleration.append(preprocessed_data['imu_acceleration'][label_start:label_end])
                fp_detections_imu_rotation.append(preprocessed_data['imu_rotation_1D'][label_start:label_end])

        return {
            'tp_detections_indices': tp_detections_indices,
            'tp_detections_sensor_data': tp_detections_sensor_data,
            'tp_detections_imu_acceleration': tp_detections_imu_acceleration,
            'tp_detections_imu_rotation': tp_detections_imu_rotation,
            'fp_detections_indices': fp_detections_indices,
            'fp_detections_sensor_data': fp_detections_sensor_data,
            'fp_detections_imu_acceleration': fp_detections_imu_acceleration,
            'fp_detections_imu_rotation': fp_detections_imu_rotation
        }

    def decorator(method):
        """Decorator that checks the `inference` flag and calls the appropriate method."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            inference = kwargs.pop('inference', False)
            if inference:
                kwargs.pop('sensation_map', None)
                return self._extract_detections_for_inference(*args, **kwargs)
            else:
                return self._extract_detections_for_train(*args, **kwargs)
        return wrapper
    
    @decorator
    def transform(self):
        pass


