# TODO: implement functionality
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)


class FeMoMetrics(object):

    @property
    def sensors(self) -> list:
        return sorted([item for s in self.sensor_selection for item in self.sensor_map[s]])

    @property
    def num_sensors(self) -> int:
        return len(self.sensors)
    
    def __init__(self,
                 maternal_dilation_forward: int = 2,
                 maternal_dilation_backward: int = 5,
                 fm_dilation: int = 3,
                 sensor_freq: int = 1024,
                 sensation_freq: int = 1024,
                 sensor_selection: list = ['accelerometer', 
                                           'piezoelectric_small', 
                                           'piezoelectric_large']) -> None:
        
        self.maternal_dilation_forward = maternal_dilation_forward
        self.maternal_dilation_backward = maternal_dilation_backward
        self.fm_dilation = fm_dilation
        self._sensor_selection = sensor_selection
        self._sensor_freq = sensor_freq
        self._sensation_freq = sensation_freq

        self.accuracy_scores = defaultdict()

    def calc_accuracy_scores(self, preds: list[np.ndarray], labels: list[np.ndarray]):
        accuracy_scores = []
        for label, pred in zip(labels, preds):
            acc = defaultdict()
            acc['test'] = accuracy_score(label, pred)
            acc['test_tpd'] = accuracy_score(label[label == 1], pred[pred == 1])
            acc['test_fpd'] = accuracy_score(label[label == 0], pred[pred == 0])
            accuracy_scores.append(acc)

        self.accuracy_scores['test_avg'] = np.mean(accuracy_scores['test'])
        self.accuracy_scores['test_tpd_avg'] = np.mean(accuracy_scores['test_tpd'])
        self.accuracy_scores['test_fpd_avg'] = np.mean(accuracy_scores['test_fpd'])
        self.accuracy_scores['test_sd'] = np.std(accuracy_scores['test'])
        self.accuracy_scores['test_tpd_sd'] = np.std(accuracy_scores['test_tpd'])
        self.accuracy_scores['test_fpd_sd'] = np.std(accuracy_scores['test_fpd'])

    def get_ml_detection_map(self,
                             preds: list[np.ndarray],
                             scheme_dict: dict,
                             sensation_map: np.ndarray):
        
        preds_tpd = np.ndarray([p[p == 1] for p in preds])
        preds_fpd = np.ndarray([p[p == 0] for p in preds])
        
        num_labels: int = scheme_dict['num_labels']
        labeled_user_scheme: np.ndarray = scheme_dict['labeled_user_scheme']

        segmented_sensor_data_ml = np.zeros(labeled_user_scheme.shape)
        
        tpd_match_idx, fpd_match_idx = 0, 0
        if num_labels:  # When there is a detection by the sensor system
            for k in range(1, num_labels + 1):
                label_start = np.where(labeled_user_scheme == k)[0][0]  # start of the label
                label_end = np.where(labeled_user_scheme == k)[0][-1] + 1  # end of the label
                indv_window = np.zeros(len(sensation_map))
                indv_window[label_start:label_end] = 1
                overlap = np.sum(indv_window * sensation_map)  # Checks the overlap with the maternal sensation

                if overlap:
                    # This is a TPD
                    if  preds_tpd[tpd_match_idx] == 1:  # Checks the detection from the classifier
                        segmented_sensor_data_ml[label_start:label_end] = 1
                    tpd_match_idx += 1
                else:
                    # This is an FPD
                    if preds_fpd[fpd_match_idx] == 1:  # Checks the detection from the classifier
                        segmented_sensor_data_ml[label_start:label_end] = 1
                    fpd_match_idx += 1

        return segmented_sensor_data_ml
    
    def match_with_sensation_map(self,
                                 ml_detection_map: np.ndarray,
                                 preprocessed_dict: dict,
                                 sensation_map: np.ndarray):
        
        pass


    