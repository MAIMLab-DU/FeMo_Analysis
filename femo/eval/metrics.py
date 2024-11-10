import os
import joblib
import tarfile
import numpy as np
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
    
    def _get_ml_detection_map(self,
                             scheme_dict: dict,
                             sensation_map: np.ndarray,
                             overall_tpd_pred: np.ndarray,
                             overall_fpd_pred: np.ndarray,
                             matching_index_tpd: int,
                             matching_index_fpd: int):
        
        num_labels: int = scheme_dict['num_labels']
        labeled_user_scheme: np.ndarray = scheme_dict['labeled_user_scheme']

        segmented_sensor_data_ml = np.zeros(labeled_user_scheme.shape)
        
        tpd_indices, fpd_indices = [], []
        if num_labels:  # When there is a detection by the sensor system
            for k in range(1, num_labels + 1):
                label_start = np.where(labeled_user_scheme == k)[0][0]  # start of the label
                label_end = np.where(labeled_user_scheme == k)[0][-1] + 1  # end of the label
                indv_window = np.zeros(len(sensation_map))
                indv_window[label_start:label_end] = 1
                overlap = np.sum(indv_window * sensation_map)  # Checks the overlap with the maternal sensation

                if overlap:
                    # This is a TPD
                    if overall_tpd_pred[matching_index_tpd] == 1:  # Checks the detection from the classifier
                        segmented_sensor_data_ml[label_start:label_end] = 1
                    matching_index_tpd += 1
                    tpd_indices.append(k)
                else:
                    # This is an FPD
                    if overall_fpd_pred[matching_index_fpd] == 1:  # Checks the detection from the classifier
                        segmented_sensor_data_ml[label_start:label_end] = 1
                    matching_index_fpd += 1
                    fpd_indices.append(k)

        return {
            'ml_detection_map': segmented_sensor_data_ml,
            'num_labels': num_labels,
            'matching_index_tpd': matching_index_tpd,
            'matching_index_fpd': matching_index_fpd,
            'tpd_indices': tpd_indices,
            'fpd_indices': fpd_indices
        }
    
    def calc_tpfp(self,
                  preprocessed_data: dict,
                  imu_map: np.ndarray,
                  sensation_map: np.ndarray,
                  ml_dict: dict|None = None,
                  **kwargs):
        
        # window size is equal to the window size used to create the maternal sensation map
        matching_window_size = self.maternal_dilation_forward + self.maternal_dilation_backward 
        # Minimum overlap in second
        min_overlap_time = self.fm_dilation / 2

        sensation_data = preprocessed_data['sensation_data']
        if ml_dict is None:
            ml_dict = self._get_ml_detection_map(sensation_map=sensation_map, **kwargs)
        ml_detection_map = np.copy(ml_dict['ml_detection_map'])

        # Variable declaration
        true_pos = 0  # True positive ML detection
        false_neg = 0  # False negative detection
        true_neg = 0  # True negative detection
        false_pos = 0  # False positive detection

        # Labeling sensation data and determining the number of maternal sensation detection
        labeled_sensation_data = label(sensation_data)
        num_maternal_sensed = len(np.unique(labeled_sensation_data)) - 1

        # ------------------ Determination of TPD and FND ----------------%    
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
                    # TPD and FND calculation
                    # Non-zero value gives the matching
                    Y = np.sum(ml_detection_map * indv_sensation_map)
                    if Y:  # true if there is a coincidence
                        true_pos += 1  # TPD incremented
                    else:
                        false_neg += 1  # FND incremented

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
            'num_sensor_sensed': ml_dict['num_labels'],
            'matching_index_tpd': ml_dict['matching_index_tpd'],
            'matching_index_fpd': ml_dict['matching_index_fpd'],
            'tpd_indices': ml_dict['tpd_indices'],
            'fpd_indices': ml_dict['fpd_indices']
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
    
    def calc_meta_info(self,
                       filename: str,
                       y_pred: np.ndarray,
                       preprocessed_data: dict,
                       fm_dict: dict,
                       scheme_dict: dict):
        
        self.logger.info(f"Calculating meta info for {filename}")

        ones = np.sum(y_pred)
        self.logger.info(f"Number of bouts: {ones}")

        num_labels = scheme_dict['num_labels']
        ml_map = np.zeros_like(scheme_dict['labeled_user_scheme'])

        for k in range(1, num_labels+1):
            L_min = np.argmax(scheme_dict['labeled_user_scheme'] == k)
            L_max = len(scheme_dict['labeled_user_scheme']) - np.argmax(scheme_dict['labeled_user_scheme'][::-1] == k)

            if y_pred[k-1] == 1:
                ml_map[L_min:L_max] = 1
            else:
                ml_map[L_min:L_max] = 0

        self.fm_dilation = 3 # Detections within this s will be considered as the same detection  

        #Now get the reduced detection_map
        reduced_detection_map = ml_map * fm_dict['fm_map']  # Reduced, because of the new dilation length
        reduced_detection_map_labeled = label(reduced_detection_map)

        n_movements = np.max(reduced_detection_map_labeled)

        # Keeps only the detected segments
        detection_only = reduced_detection_map[reduced_detection_map == 1]
        # Dilation length on sides of each detection is removed and converted to minutes
        total_FM_duration = (len(detection_only) / self._sensor_freq - n_movements * self.fm_dilation / 2) / 60

        #If there are no detection
        if n_movements != 0:
            # mean duration in s
            mean_FM_duration = total_FM_duration * 60 / n_movements
        #If no movement then make mean zero
        else:
            mean_FM_duration =  total_FM_duration * 0

        onset_interval = []
        for j in range(1, n_movements):
            onset1 = np.where(reduced_detection_map_labeled == j)[0][0]  # Sample no. corresponding to start of the label
            onset2 = np.where(reduced_detection_map_labeled == j + 1)[0][0]  # Sample no. corresponding to start of the next label
            onset_interval.append( (onset2 - onset1) / self._sensor_freq ) # onset to onset interval in seconds

        # Median onset interval in s
        median_onset_interval = np.median(onset_interval)

        duration_trimmed_data_files = len(preprocessed_data['sensor_1']) / 1024 / 60 # in minutes
        # Time fetus was not moving
        total_nonFM_duration = np.array(duration_trimmed_data_files) - np.array(total_FM_duration)
        active_time = (total_FM_duration / (total_FM_duration + total_nonFM_duration)) *100
        detection_per_hr = (np.array(n_movements) *60)  /  (total_FM_duration + total_nonFM_duration)

        data = {
            "File Name": [os.path.basename(filename)],
            "Number of bouts per hour": [detection_per_hr],
            "Mean duration of fetal movement (seconds)": [mean_FM_duration],
            "Median onset interval (seconds)": [median_onset_interval],
            "Active time of fetal movement (%)": [active_time]
        }

        return data, ml_map
    
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
    