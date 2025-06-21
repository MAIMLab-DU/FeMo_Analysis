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
              sensation_map: np.ndarray,
              scheme_dict: dict,
              pred_results: pd.DataFrame):
    
        # Get user scheme data
        user_scheme = scheme_dict['user_scheme']
        labeled_user_scheme = scheme_dict['labeled_user_scheme'] 
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
        num_maternal_sensed = len(np.unique(label(sensation_map))) - 1
        num_ml_detections = len(np.unique(label(ml_prediction_map))) - 1
        
        # Variable declaration for confusion matrix
        true_pos = 0   # True FM detections
        false_neg = 0  # Missed FM detections  
        true_neg = 0   # True Non-FM detections
        false_pos = 0  # False FM detections
        
        # ------------------ Calculate TP and FN (FM Detection Performance) ------------------
        if num_maternal_sensed > 0:  # When there are maternal sensations
            # ------------------ Simple Binary Map Matching ------------------
            # Process each segment in the user_scheme
            if num_labels > 0:
                for k in range(1, num_labels + 1):
                    # Get segment boundaries
                    segment_indices = np.where(labeled_user_scheme == k)[0]
                    segment_start = segment_indices[0]
                    segment_end = segment_indices[-1] + 1
                    
                    # Check overlap with each map for this segment
                    ml_overlap = np.sum(ml_prediction_map[segment_start:segment_end])
                    sensation_overlap = np.sum(sensation_map[segment_start:segment_end])
                    
                    # Convert overlaps to binary (any overlap = 1, no overlap = 0)
                    ml_present = 1 if ml_overlap > 0 else 0
                    sensation_present = 1 if sensation_overlap > 0 else 0
                    
                    # Calculate TP, FP, TN, FN based on the three binary maps
                    if sensation_present and ml_present:
                        # Ground truth FM + ML detected FM
                        true_pos += 1
                    elif sensation_present and not ml_present:
                        # Ground truth FM + ML missed FM
                        false_neg += 1
                    elif not sensation_present and ml_present:
                        # No ground truth FM + ML detected FM
                        false_pos += 1
                    elif not sensation_present and not ml_present:
                        # No ground truth FM + ML correctly identified no FM
                        true_neg += 1
            
            else:
                # No segments detected by user_scheme but maternal sensations exist
                # This means sensor system missed all maternal sensations
                false_neg = num_maternal_sensed

        else:
            # No maternal sensations - only calculate TN and FP
            if num_labels > 0:
                for k in range(1, num_labels + 1):
                    # Get segment boundaries
                    segment_indices = np.where(labeled_user_scheme == k)[0]
                    segment_start = segment_indices[0]
                    segment_end = segment_indices[-1] + 1
                    
                    # Check ML prediction for this segment
                    ml_overlap = np.sum(ml_prediction_map[segment_start:segment_end])
                    ml_present = 1 if ml_overlap > 0 else 0
                    
                    if ml_present:
                        # ML detected FM where there was no maternal sensation
                        false_pos += 1
                    else:
                        # ML correctly identified no FM
                        true_neg += 1
        
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
    