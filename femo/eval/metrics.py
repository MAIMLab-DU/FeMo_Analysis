import os
import joblib
import tarfile
import numpy as np
import pandas as pd
from ..logger import LOGGER
from skimage.measure import label
from sklearn.metrics import accuracy_score


class FeMoMetrics(object):

    @property
    def logger(self):
        return LOGGER

    @property
    def sensor_map(self):
        return {
            'accelerometer': ['sensor_1', 'sensor_2'],
            'piezoelectric_large': ['sensor_3', 'sensor_6'],
            'piezoelectric_small': ['sensor_4', 'sensor_5']
        }

    @property
    def sensors(self) -> list:
        return sorted([item for s in self._sensor_selection for item in self.sensor_map[s]])

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
        # Dilation length in sample number 
        self.fm_dilation_size = round(fm_dilation * sensor_freq)
        self.extension_forward = round(maternal_dilation_forward * sensor_freq)
        self.extension_backward = round(maternal_dilation_backward * sensor_freq)

    def calc_accuracy_scores(self, preds: list[np.ndarray], labels: list[np.ndarray]):

        acc, acc_tpd, acc_fpd = [], [], []
        num_folds = len(preds)
        for i in range(num_folds):
            label, pred = labels[i], preds[i]
            acc.append(accuracy_score(label, pred))
            acc_tpd.append(accuracy_score(label[label == 1], pred[pred == 1]))
            acc_fpd.append(accuracy_score(label[label == 0], pred[pred == 0]))

        return {
            'test_avg': np.mean(acc),
            'test_sd': np.std(acc),
            'test_tpd_avg': np.mean(acc_tpd),
            'test_tpd_sd': np.std(acc_tpd),
            'test_fpd_avg': np.mean(acc_fpd),
            'test_fpd_sd': np.std(acc_fpd),
        }
    
    def calc_tpfp(self,
                preprocessed_data: dict,
                imu_map: np.ndarray,
                sensation_map: np.ndarray,
                scheme_dict: dict,
                pred_results: pd.DataFrame):
        
        # window size is equal to the window size used to create the maternal sensation map
        matching_window_size = self.maternal_dilation_forward + self.maternal_dilation_backward 
        # Minimum overlap in second
        min_overlap_time = self.fm_dilation / 2
        sensation_data = preprocessed_data['sensation_data']
    
        # Get user scheme data
        user_scheme = scheme_dict['user_scheme']
        num_labels = scheme_dict['num_labels']
        
        # Initialize ML prediction map with same length as other maps
        ml_prediction_map = np.zeros(len(user_scheme))        
        # Populate ML prediction map using pred_results DataFrame
        if not pred_results.empty:
            for idx, row in pred_results.iterrows():
                start_idx = int(row['start_indices'])
                end_idx = int(row['end_indices'])
                prediction = int(row['predictions'])
                
                # Only mark as 1 if prediction is positive (1) and indices are valid
                if prediction == 1 and 0 <= start_idx < len(ml_prediction_map) and 0 <= end_idx <= len(ml_prediction_map):
                    ml_prediction_map[start_idx:end_idx] = 1
        
        # Get sensation data for ground truth
        num_ml_detections = len(np.unique(label(ml_prediction_map))) - 1
        ml_detection_map = np.copy(ml_prediction_map)
        
        # Variable declaration
        true_pos = 0  # True positive ML detection
        false_neg = 0  # False negative detection
        true_neg = 0  # True negative detection
        false_pos = 0  # False positive detection

        # Labeling sensation data and determining the number of maternal sensation detection
        labeled_sensation_data = label(sensation_data)
        num_maternal_sensed = len(np.unique(labeled_sensation_data)) - 1

        # ------------------ Determination of True Positive Prediction and False Negative Prediction ----------------%    
        if num_maternal_sensed:  # When there is a detection by the mother
            for k in range(1, num_maternal_sensed + 1):
                L_min = np.where(labeled_sensation_data == k)[0][0]  # Sample no. corresponding to the start of the label
                L_max = np.where(labeled_sensation_data == k)[0][-1] # Sample no. corresponding to the end of the label

                # sample no. for the starting point of this sensation in the map
                L1 = L_min * round(self._sensor_freq / self._sensation_freq) - self.extension_backward
                L1 = max(L1, 0)  # Just a check so that L1 remains higher than 1st data sample

                # sample no. for the ending point of this sensation in the map
                L2 = L_max * round(self._sensor_freq / self._sensation_freq) + self.extension_forward
                L2 = min(L2, len(ml_detection_map))  # Just a check so that L2 remains lower than the last data sample

                indv_sensation_map = np.zeros(len(ml_detection_map))  # Need to be initialized before every detection matching
                indv_sensation_map[L1:L2+1] = 1  # mapping individual sensation data

                # this is non-zero if there is a coincidence with maternal body movement
                overlap = np.sum(indv_sensation_map * imu_map)

                if not overlap:  # true when there is no coincidence, meaning FM
                    # TPP and FNP calculation
                    # Non-zero value gives the matching
                    Y = np.sum(ml_detection_map * indv_sensation_map)
                    if Y:  # true if there is a coincidence
                        true_pos += 1  # TPP incremented
                    else:
                        false_neg += 1  # FNP incremented

            # ------------------- Determination of True Negative Prediction and False Positive Prediction  ------------------%    
            # Removal of the TPD and FND parts from the individual sensor data
            labeled_ml_detection = label(ml_detection_map)
            # Non-zero elements give the matching. In sensation_map multiple windows can overlap, which was not the case in the sensation_data
            curnt_matched_vector = labeled_ml_detection * sensation_map
            # Gives the label of the matched sensor data segments
            curnt_matched_label = np.unique(curnt_matched_vector)
            arb_value = 4  # An arbitrary value
            
            
            if len(curnt_matched_label) > 1:
                curnt_matched_label = curnt_matched_label[1:]  # Removes the first element, which is 0
                for m in range(len(curnt_matched_label)):
                    ml_detection_map[labeled_ml_detection == curnt_matched_label[m]] = arb_value
                    # Assigns an arbitrary value to the TPD segments of the segmented signal

            # Assigns an arbitrary value to the area under the M_sntn_Map
            ml_detection_map[sensation_map == 1] = arb_value
            # Removes all the elements with value = arb_value from the segmented data
            removed_ml_detection = ml_detection_map[ml_detection_map != arb_value]

            # Calculation of TND and FPD for individual sensors
            L_removed = len(removed_ml_detection)
            index_window_start = 0
            index_window_end = int(min(index_window_start+self._sensor_freq*matching_window_size, L_removed))
            while index_window_start < L_removed:
                indv_window = removed_ml_detection[index_window_start: index_window_end]
                index_non_zero = np.where(indv_window)[0]

                if len(index_non_zero) >= (min_overlap_time*self._sensor_freq):
                    false_pos += 1
                else:
                    true_neg += 1

                index_window_start = index_window_end + 1
                index_window_end = int(min(index_window_start+self._sensor_freq*matching_window_size, L_removed))
                
        # Else no ground truth, only True Negative Prediction and False Positive Prediction calculation allowed
        else:
            # ------------------- Determination of TND and FPD  ------------------%    
            # Removal of the TPD and FND parts from the individual sensor data
            labeled_ml_detection = label(ml_detection_map)
            # Non-zero elements give the matching. In sensation_map multiple windows can overlap, which was not the case in the sensation_data
            curnt_matched_vector = labeled_ml_detection * sensation_map
            # Gives the label of the matched sensor data segments
            curnt_matched_label = np.unique(curnt_matched_vector)
            arb_value = 4  # An arbitrary value
            
            
            if len(curnt_matched_label) > 1:
                curnt_matched_label = curnt_matched_label[1:]  # Removes the first element, which is 0
                for m in range(len(curnt_matched_label)):
                    ml_detection_map[labeled_ml_detection == curnt_matched_label[m]] = arb_value
                    # Assigns an arbitrary value to the TPD segments of the segmented signal

            # Assigns an arbitrary value to the area under the M_sntn_Map
            ml_detection_map[sensation_map == 1] = arb_value
            # Removes all the elements with value = arb_value from the segmented data
            removed_ml_detection = ml_detection_map[ml_detection_map != arb_value]

            # Calculation of TND and FPD for individual sensors
            L_removed = len(removed_ml_detection)
            index_window_start = 0
            index_window_end = int(min(index_window_start+self._sensor_freq*matching_window_size, L_removed))
            while index_window_start < L_removed:
                indv_window = removed_ml_detection[index_window_start: index_window_end]
                index_non_zero = np.where(indv_window)[0]

                if len(index_non_zero) >= (min_overlap_time*self._sensor_freq):
                    false_pos += 1
                else:
                    true_neg += 1

                index_window_start = index_window_end + 1
                index_window_end = int(min(index_window_start+self._sensor_freq*matching_window_size, L_removed))
        
        return {
            'true_positive': true_pos,
            'false_positive': false_pos, 
            'true_negative': true_neg,
            'false_negative': false_neg,
            'num_maternal_sensed': num_maternal_sensed,
            'num_sensor_detections': num_labels,
            'num_ml_detections': num_ml_detections
        }
    
    def calc_metrics(self,
                     tpfp_dict: dict):
        
        metric_dict = {
            'accuracy': 0,
            'sensitivity': 0,
            'specificity': 0,
            'precision/ppv': 0,
            'f1-score': 0,
            'fp_rate': 0,
            'PABAK': 0
        }

        tp = tpfp_dict['true_positive']
        fp = tpfp_dict['false_positive']
        tn = tpfp_dict['true_negative']
        fn = tpfp_dict['false_negative']

        accuracy = (tp + tn) / (tp + fp + tn + fn)
        metric_dict['accuracy'] = accuracy

        sensitivity = tp / (tp + fn)
        metric_dict['sensitivity'] = sensitivity
        
        precision = tp / (tp + fp)
        metric_dict['precision/ppv'] = precision

        specificity = tn / (tn + fp)
        metric_dict['specificity'] = specificity
        
        fp_rate = fp / (fp + tn)
        metric_dict['fp_rate'] = fp_rate

        beta = 1
        f1_score = (1+beta**2)*(precision*sensitivity) / (beta**2*precision+sensitivity)
        metric_dict['f1-score'] = f1_score

        pabak = 2*accuracy - 1
        metric_dict['PABAK'] = pabak

        return metric_dict
    
    def save(self, file_path):
        """Save the metrics to a joblib file

        Args:
            file_path (str): Path to directory for saving the metrics
        """
        
        joblib.dump(self, os.path.join(file_path, "metrics.joblib"))
        tar = tarfile.open(os.path.join(file_path, "metrics.tar.gz"), "w:gz")
        tar.add(os.path.join(file_path, "metrics.joblib"), arcname="metrics.joblib")
        tar.close()
        self.logger.debug(f"Metrics saved to {file_path}")
    