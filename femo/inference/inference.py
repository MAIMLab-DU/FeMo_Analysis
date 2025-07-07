import os
import joblib
import boto3
import tempfile
import numpy as np
from collections import defaultdict
from skimage.measure import label
from dataclasses import dataclass
from .hiccup import HiccupAnalysis
from ..model.base import FeMoBaseClassifier
from ..logger import LOGGER
from ..data.pipeline import Pipeline
from ..eval.metrics import FeMoMetrics
from ..plot.plotter import FeMoPlotter
from ..data.process import Processor
from ..data.transforms import SensorFusion
from ..data.transforms._utils import custom_binary_dilation, custom_binary_erosion


@dataclass
class InferenceMetaInfo:
    """
    Holds the data for inference.
    """

    fileName: str
    numKicks: int
    totalFMDuration: float
    totalNonFMDuration: float
    onsetInterval: list
    matchWithSensationMap: dict = None


class PredictionService(object):
    classifier: FeMoBaseClassifier = None
    pipeline: Pipeline = None
    processor: defaultdict = {"crafted": None, "tsfel": None}
    metrics: FeMoMetrics = None
    plotter = FeMoPlotter()
    default_hiccup_cfg = {
        "std_threshold_percentage": 0.15,
        "hiccup_period_distance": 5,
        "y_shift": 0.5,
        "peak_distance": 800,
        "fusion": "piezo_sup",
        "hiccup_continuous_time": 150,
        "exception_per_minute": 2,
        "tolerance_limit": 6,
        "delta": 2,
        "fm_dilation": 3,
    }

    @property
    def logger(self):
        return LOGGER

    def __init__(
        self,
        classifier_path: str,
        model_path: str,
        pipeline_path: str,
        processor_path: str,
        metrics_path: str,
        pred_cfg: dict,
    ) -> None:
        self.classifier_path = classifier_path
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.processor_path = processor_path
        self.metrics_path = metrics_path
        self.pred_cfg = pred_cfg

    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.classifier is None:
            self.classifier = joblib.load(self.classifier_path)
        self.classifier.load_model(self.model_path)
        return self.classifier

    def get_pipeline(self):
        """Get the pipeline object for this instance, loading it if it's not already loaded."""
        if self.pipeline is None:
            self.pipeline = Pipeline.load(self.pipeline_path)
            self.pipeline.inference = True
        return self.pipeline

    def get_processor(self, feature_set: str = None):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.processor[feature_set] is None and feature_set is not None:
            try:
                self.processor[feature_set] = joblib.load(
                    self.processor_path.replace("processor.joblib", f"{feature_set}_processor.joblib")
                )
            except FileNotFoundError:  # legacy processor (only 'crafted' feature set)
                if os.path.exists(self.processor_path) and feature_set == 'crafted':
                    self.processor[feature_set] = joblib.load(self.processor_path)
                    return self.processor[feature_set]
                return None
        return self.processor[feature_set]

    def get_metrics(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.metrics is None:
            self.metrics = joblib.load(self.metrics_path)
        return self.metrics

    def _calc_meta_info(self, filename: str, pipeline_output: dict, binary_map: np.ndarray):
        self.logger.info(f"Calculating prediction meta info {filename}")

        preprocessed_data = pipeline_output["preprocessed_data"]
        sensor_freq = self.pipeline.get_stage("load").sensor_freq
        fm_dilation = self.pipeline.get_stage("segment").fm_dilation

        labeled_binary_map = label(binary_map)
        n_movements = np.max(labeled_binary_map)

        # Keeps only the detected segments
        detection_only = binary_map[binary_map == 1]
        # Dilation length on sides of each detection is removed and converted to minutes
        total_FM_duration = (len(detection_only) / sensor_freq - n_movements * fm_dilation / 2) / 60

        onset_interval = []
        for j in range(1, n_movements):
            onset1 = np.where(labeled_binary_map == j)[0][0]  # Sample no. corresponding to start of the label
            onset2 = np.where(labeled_binary_map == j + 1)[0][0]  # Sample no. corresponding to start of the next label
            onset_interval.append((onset2 - onset1) / sensor_freq)  # onset to onset interval in seconds

        duration_trimmed_data_files = len(preprocessed_data["sensor_1"]) / 1024 / 60  # in minutes
        # Time fetus was not moving
        total_nonFM_duration = duration_trimmed_data_files - total_FM_duration

        # Match with sensation map if available
        matched_dict = None
        if np.sum(pipeline_output["sensation_map"]):
            sensation_map = pipeline_output["sensation_map"]
            imu_map = pipeline_output["imu_map"]
            matched_dict = self._match_with_sensation_map(
                preprocessed_data=pipeline_output["preprocessed_data"],
                imu_map=imu_map,
                sensation_map=sensation_map,
                ml_detection_map=binary_map,
            )

        data = InferenceMetaInfo(
            fileName=os.path.basename(filename),
            numKicks=n_movements,
            totalFMDuration=total_FM_duration,
            totalNonFMDuration=total_nonFM_duration,
            onsetInterval=onset_interval,
            matchWithSensationMap=matched_dict
        )
        return data

    def _match_with_sensation_map(self,
                                  preprocessed_data: dict,
                                  imu_map: np.ndarray,
                                  sensation_map: np.ndarray,
                                  ml_detection_map: np.ndarray):

        # window size is equal to the window size used to create the maternal sensation map
        matching_window_size = self.metrics.maternal_dilation_forward + self.metrics.maternal_dilation_backward 
        # Minimum overlap in second
        min_overlap_time = self.metrics.fm_dilation / 2

        sensation_data = preprocessed_data['sensation_data']

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
                L1 = L_min * round(self.metrics._sensor_freq / self.metrics._sensation_freq) - self.metrics.extension_backward
                L1 = max(L1, 0)  # Just a check so that L1 remains higher than 1st data sample

                # sample no. for the ending point of this sensation in the map
                L2 = L_max * round(self.metrics._sensor_freq / self.metrics._sensation_freq) + self.metrics.extension_forward
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
            index_window_end = int(min(index_window_start+self.metrics._sensor_freq*matching_window_size, L_removed))
            while index_window_start < L_removed:
                indv_window = removed_ml_detection[index_window_start: index_window_end]
                index_non_zero = np.where(indv_window)[0]

                if len(index_non_zero) >= (min_overlap_time*self.metrics._sensor_freq):
                    false_pos += 1
                else:
                    true_neg += 1

                index_window_start = index_window_end + 1
                index_window_end = int(min(index_window_start+self.metrics._sensor_freq*matching_window_size, L_removed))
        
        return {
            "num_maternally_sensed_kicks": true_pos+false_neg,
            "num_detected_kicks": true_pos+false_pos,
            "num_sensor_events": true_pos + false_neg + true_neg + false_pos,
        }

    def _pre_hiccup_removal(
        self,
        filename: str,
        y_pred: np.ndarray,
        pipeline_output: dict,
    ):
        self.logger.info(f"Generating pre-hiccup removal map for {filename}")
        fm_dict = pipeline_output["fm_dict"]
        scheme_dict = pipeline_output["scheme_dict"]

        ones = np.sum(y_pred)
        self.logger.info(f"Number of bouts: {ones}")

        num_labels = scheme_dict["num_labels"]
        ml_map = np.zeros_like(scheme_dict["labeled_user_scheme"])

        for k in range(1, num_labels + 1):
            L_min = np.argmax(scheme_dict["labeled_user_scheme"] == k)
            L_max = len(scheme_dict["labeled_user_scheme"]) - np.argmax(scheme_dict["labeled_user_scheme"][::-1] == k)

            if y_pred[k - 1] == 1:
                ml_map[L_min:L_max] = 1
            else:
                ml_map[L_min:L_max] = 0

        # Now get the reduced detection_map
        reduced_detection_map = ml_map * fm_dict["fm_map"]  # Reduced, because of the new dilation length
        data = self._calc_meta_info(filename, pipeline_output, reduced_detection_map)

        return data, ml_map

    def _post_hiccup_removal(
        self,
        filename: str,
        hiccup_analyzer: HiccupAnalysis,
        ml_map: np.ndarray,
        pipeline_output: dict,
    ):
        self.logger.info(f"Generating post-hiccup removal map from {filename}")
        preprocessed_data = pipeline_output["preprocessed_data"]
        imu_map = pipeline_output["imu_map"]
        ml_detection_map = np.copy(ml_map)

        if hiccup_analyzer.fm_dilation > self.pipeline.get_stage("segment").fm_dilation:
            dilation_size = int(
                (hiccup_analyzer.fm_dilation - self.pipeline.get_stage("segment").fm_dilation)
                * hiccup_analyzer.Fs_sensor
            )
            ml_detection_map = custom_binary_dilation(ml_detection_map, dilation_size)
        if hiccup_analyzer.fm_dilation < self.pipeline.get_stage("segment").fm_dilation:
            erosion_size = int(
                (self.pipeline.get_stage("segment").fm_dilation - hiccup_analyzer.fm_dilation)
                * hiccup_analyzer.Fs_sensor
            )
            ml_detection_map = custom_binary_erosion(ml_detection_map, erosion_size)

        reduced_detection_map = ml_detection_map * ml_map

        filtered_signals = {
            "Piezo1": preprocessed_data["sensor_3"],
            "Piezo2": preprocessed_data["sensor_6"],
            "Aclm1": preprocessed_data["sensor_1"],
            "Aclm2": preprocessed_data["sensor_2"],
            "Acstc1": preprocessed_data["sensor_4"],
            "Acstc2": preprocessed_data["sensor_5"],
        }

        try:
            (
                dilated_data,
                pre_hiccup_map,
                hiccup_map,
                aclm_1_or_2,
                piezo_1_or_2,
                acstc_1_or_2,
            ) = hiccup_analyzer.detect_hiccups(
                reduced_detection_map=reduced_detection_map,
                signals=filtered_signals,
                imu_map=imu_map,
            )
        except Exception:
            LOGGER.info("Empty hiccup array...")
            dilated_data = custom_binary_dilation(
                reduced_detection_map,
                hiccup_analyzer.period_distance * hiccup_analyzer.Fs_sensor,
            )
            pre_hiccup_map = ml_map.copy()
            hiccup_map = np.zeros_like(ml_map)
            aclm_1_or_2 = np.zeros_like(ml_map)
            piezo_1_or_2 = np.zeros_like(ml_map)
            acstc_1_or_2 = np.zeros_like(ml_map)

        hiccup_removed_ml_map = np.clip(ml_map - hiccup_map, 0, 1)
        data = self._calc_meta_info(filename, pipeline_output, hiccup_removed_ml_map)

        return {
            "data": data,
            "dilated_data": dilated_data,
            "pre_hiccup_map": pre_hiccup_map,
            "hiccup_map": hiccup_map,
            "hiccup_removed_ml_map": hiccup_removed_ml_map,
            "aclm_1_or_2": aclm_1_or_2,
            "piezo_1_or_2": piezo_1_or_2,
            "acstc_1_or_2": acstc_1_or_2,
        }

    def predict(self, filename: str, bucket_name: str = None, remove_hiccups: bool = False):
        """For the input, perform predictions and return results.

        Args:
            filename (str): The path to the file that needs to be processed.
            bucket_name (str, optional): The name of the S3 bucket where the file is located. Defaults to None.
            remove_hiccups(bool, optional): Wether to remove hiccups for analysis. Defaults to False.
        """

        # Helper function to process the file
        def process_file(file_path: str):
            prediction_output = {
                "pre_hiccup_removal": None,
                "post_hiccup_removal": None
            }

            pipeline = self.get_pipeline()
            feature_sets = pipeline.get_stage('extract_feat').feature_sets
            feature_dict, pipeline_output = pipeline.extract_features_batch(filename=file_path, feature_sets=feature_sets)

            feature_arrays = []
            for feature_set in sorted(feature_dict.keys()):  # ['crafted', 'tsfel'] order
                X_extracted = feature_dict[feature_set]["features"]
                processor: Processor = self.get_processor(feature_set)
                if processor is not None:
                    feature_arrays.append(processor.predict(X_extracted))

            prediction_output["pipeline_output"] = pipeline_output

            X_norm_ranked = np.hstack(feature_arrays)

            clf = self.get_model()
            if X_norm_ranked.shape[0] == 0:
                y_pred = np.array([0])
            else:
                y_pred = clf.predict(X_norm_ranked)

            data, ml_map = self._pre_hiccup_removal(
                filename=file_path,
                y_pred=y_pred,
                pipeline_output=pipeline_output,
            )
            prediction_output["pre_hiccup_removal"] = {"data": data, "ml_map": ml_map}

            if remove_hiccups:
                # Initialize hiccup_analyzer now that pipeline is loaded
                hiccup_cfg = self.pred_cfg.get("hiccup_removal", self.default_hiccup_cfg)
                hiccup_analyzer = HiccupAnalysis(Fs_sensor=pipeline.get_stage("load").sensor_freq, **hiccup_cfg)
                self.logger.info(f"{hiccup_analyzer.config = }")
                
                post_hiccup_dict = self._post_hiccup_removal(
                    filename=filename,
                    hiccup_analyzer=hiccup_analyzer,
                    ml_map=ml_map,
                    pipeline_output=pipeline_output,
                )
                prediction_output["post_hiccup_removal"] = post_hiccup_dict

            return prediction_output

        # Check if file is in S3 and download if necessary
        if bucket_name:
            s3_client = boto3.resource("s3")
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                s3_client.Bucket(bucket_name).download_file(filename, temp_file.name)
                LOGGER.info(f"Downloaded {filename} from S3 bucket {bucket_name} to {temp_file.name}")
                result = process_file(temp_file.name)
                os.remove(temp_file.name)
        else:
            result = process_file(filename)

        return result

    def save_pred_plots(
        self,
        pipeline_output: dict,
        ml_map: np.ndarray,
        filename: str,
        det_type: str = "Fetal movement",
    ):
        plt_cfg = {"figsize": [16, 15], "x_unit": "min"}
        fig, axes = self.plotter.create_figure(figsize=plt_cfg["figsize"])
        i = 0
        for sensor_type in self.plotter.sensor_map.keys():
            for j in range(len(self.plotter.sensor_map[sensor_type])):
                sensor_name = f"{sensor_type}_{j+1}"
                axes = self.plotter.plot_sensor_data(
                    axes=axes,
                    axis_idx=i,
                    data=pipeline_output["preprocessed_data"][self.plotter.sensor_map[sensor_type][j]],
                    sensor_name=sensor_name,
                    x_unit=plt_cfg.get("x_unit", "min"),
                )
                i += 1

        # TODO: might be better to have this more configurable
        fusion_stage: SensorFusion = self.pipeline.get_stage("fusion")
        desired_scheme = fusion_stage.desired_scheme

        axes = self.plotter.plot_detections(
            axes=axes,
            axis_idx=i,
            detection_map=pipeline_output["scheme_dict"]["user_scheme"],
            det_type=f"At least {desired_scheme[1]} {desired_scheme[0]} Sensor Events",
            ylabel="Detection",
            xlabel="",
            x_unit=plt_cfg.get("x_unit", "min"),
        )
        axes = self.plotter.plot_detections(
            axes=axes,
            axis_idx=i + 1,
            detection_map=ml_map,
            det_type=det_type,
            ylabel="Detection",
            xlabel=f"Time ({plt_cfg.get('x_unit', 'min')})",
            x_unit=plt_cfg.get("x_unit", "min"),
        )

        self.plotter.save_figure(fig, filename)

    def save_hiccup_analysis_plots(
        self,
        pipeline_output: dict,
        hiccup_output: dict,
        ml_map: np.ndarray,
        filename: str,
    ):
        hiccup_cfg = self.pred_cfg.get("hiccup_removal", self.default_hiccup_cfg)
        hiccup_analyzer = HiccupAnalysis(Fs_sensor=self.pipeline.get_stage("load").sensor_freq, **hiccup_cfg)

        hiccup_analyzer.plot_hiccup_analysis_v3(
            fltdSignal_aclm1=pipeline_output["preprocessed_data"]["sensor_1"],
            fltdSignal_piezo1=pipeline_output["preprocessed_data"]["sensor_3"],
            fltdSignal_acstc1=pipeline_output["preprocessed_data"]["sensor_4"],
            detection_map=ml_map,
            hiccup_map=hiccup_output["hiccup_map"],
            cndtn1_body_map=hiccup_output["dilated_data"],
            accelerometer_1_or_2=hiccup_output["aclm_1_or_2"],
            piezo_1_or_2=hiccup_output["piezo_1_or_2"],
            acoustic_1_or_2=hiccup_output["acstc_1_or_2"],
            x_axis_type="m",
            file_name=filename,
        )
