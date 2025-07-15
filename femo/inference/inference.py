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
from ..data.process import Processor
from ..data.transforms import SensorFusion
from ..data.transforms._utils import custom_binary_dilation, custom_binary_erosion
from .plot_utils import (
    create_figure,
    save_figure,
    plot_imu_rotation,
    plot_imu_acceleration,
    plot_sensor_data,
    plot_sensation_map,
    plot_detections
)


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


class PredictionService(object):
    classifier: FeMoBaseClassifier = None
    pipeline: Pipeline = None
    processor: defaultdict = {"crafted": None, "tsfel": None}
    metrics: FeMoMetrics = None
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

    def _calc_meta_info(self, filename: str, preprocessed_data: dict, binary_map: np.ndarray):
        self.logger.info(f"Calculating prediction meta info {filename}")
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
        data = InferenceMetaInfo(
            fileName=os.path.basename(filename),
            numKicks=n_movements,
            totalFMDuration=total_FM_duration,
            totalNonFMDuration=total_nonFM_duration,
            onsetInterval=onset_interval,
        )
        return data

    def _pre_hiccup_removal(
        self,
        filename: str,
        y_pred: np.ndarray,
        preprocessed_data: dict,
        fm_dict: dict,
        scheme_dict: dict,
    ):
        self.logger.info(f"Generating pre-hiccup removal map for {filename}")

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
        data = self._calc_meta_info(filename, preprocessed_data, reduced_detection_map)

        return data, ml_map

    def _post_hiccup_removal(
        self,
        filename: str,
        hiccup_analyzer: HiccupAnalysis,
        ml_map: np.ndarray,
        preprocessed_data: dict,
        imu_map: np.ndarray,
    ):
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
        data = self._calc_meta_info(filename, preprocessed_data, hiccup_removed_ml_map)

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
            prediction_output = defaultdict()

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
                preprocessed_data=pipeline_output["preprocessed_data"],
                fm_dict=pipeline_output["fm_dict"],
                scheme_dict=pipeline_output["scheme_dict"],
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
                    preprocessed_data=pipeline_output["preprocessed_data"],
                    imu_map=pipeline_output["imu_map"],
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
        detection_map: np.ndarray,
        filename: str,
        det_type: str = "ML Predicted Fetal Movement",
        plot_all_sensors: bool = False,
        plot_preprocessed: bool = False,
    ) -> None:
        """Save prediction plots including IMU data, sensor data, and detection maps.
        
        Args:
            pipeline_output: Dictionary containing preprocessed data and other pipeline outputs
            ml_map: Machine learning detection map
            filename: Filename for saving the plot
            det_type: Type of detection for labeling
        """
        plt_config: dict = {"figsize": [16, 15]}
        plot_key: str = "preprocessed_data" if plot_preprocessed else "loaded_data"
        
        # Calculate total number of subplots needed by checking actual data availability
        num_subplots: int = 0
        plot_types = ["imu_rotation", "imu_acceleration"]
        if plot_all_sensors:
            sensors = self.pipeline.get_stage("load").sensor_map.values()
            plot_types.extend([item for values in sensors for item in values])
        else:
            plot_types.extend(self.pipeline.get_stage("load").sensors)
        
        num_subplots += len(plot_types)  # Add number of sensor plots        
        # Add sensation map if present and has data
        if "sensation_map" in pipeline_output and np.sum(pipeline_output["sensation_map"]) > 0:
            num_subplots += 1        
        # Add 2 for detection plots
        num_subplots += 2
        self.logger.info(f"Total number of subplots to create: {num_subplots}")
        
        # Create figure with correct number of subplots
        fig, axes = create_figure(figsize=plt_config["figsize"], num_subplots=num_subplots)
        current_axis_index: int = 0
        
        # Get sensor frequency for time axis calculations
        sensor_freq: int = self.pipeline.get_stage("load").sensor_freq
        
        # Plot IMU rotation data (roll, pitch, yaw) in the first subplot
        if "imu_rotation" in pipeline_output[plot_key]:
            imu_rotation_data: np.ndarray = pipeline_output[plot_key]["imu_rotation"]
            self.logger.info(f"Plotting IMU rotation data with shape {imu_rotation_data.shape}")
            axes = plot_imu_rotation(
                axes=axes,
                axis_idx=current_axis_index,
                imu_rotation_data=imu_rotation_data,
                sensor_freq=sensor_freq,
                preprocessed=plot_preprocessed
            )
            current_axis_index += 1
        
        # Plot IMU acceleration data in the second subplot
        if "imu_acceleration" in pipeline_output[plot_key]:
            imu_acceleration_data: np.ndarray = pipeline_output[plot_key]["imu_acceleration"]
            self.logger.info(f"Plotting IMU acceleration data with shape {imu_acceleration_data.shape}")
            axes = plot_imu_acceleration(
                axes=axes,
                axis_idx=current_axis_index,
                imu_acceleration_data=imu_acceleration_data,
                sensor_freq=sensor_freq,
                preprocessed=plot_preprocessed
            )
            current_axis_index += 1
        
        # Plot sensor data - only plot those with actual data
        for sensor in plot_types[2:]:  # Skip IMU rotation and acceleration
            if sensor in pipeline_output[plot_key]:
                sensor_data = pipeline_output[plot_key][sensor]
                if np.sum(sensor_data) == 0:
                    self.logger.warning(f"No data for sensor {sensor}, skipping plot.")
                    continue
                self.logger.info(f"Plotting sensor data for {sensor} with shape {sensor_data.shape}")
                axes = plot_sensor_data(
                    axes=axes,
                    axis_idx=current_axis_index,
                    sensor_data=sensor_data,
                    sensor_name=sensor,
                    sensor_freq=sensor_freq,
                    preprocessed=plot_preprocessed
                )
                current_axis_index += 1
        
        # Plot sensation map if present and has data
        if "sensation_map" in pipeline_output and np.sum(pipeline_output["sensation_map"]) > 0:
            self.logger.info("Plotting sensation map.")
            axes = plot_sensation_map(
                axes=axes,
                axis_idx=current_axis_index,
                sensation_data=pipeline_output["sensation_map"],
                sensor_freq=sensor_freq
            )
            current_axis_index += 1

        # Plot detection maps
        fusion_stage: SensorFusion = self.pipeline.get_stage("fusion")
        desired_scheme_tuple = fusion_stage.desired_scheme

        # Plot detection map from desired scheme
        self.logger.info(f"Plotting detection map for scheme: {desired_scheme_tuple}")
        axes = plot_detections(
            axes=axes,
            axis_idx=current_axis_index,
            detection_map=pipeline_output["scheme_dict"]["user_scheme"],
            det_type=f"At least {desired_scheme_tuple[1]} {desired_scheme_tuple[0]} Sensor Events",
            ylabel="Detection",
            xlabel="",
            sensor_freq=sensor_freq
        )
        current_axis_index += 1
        
        # Plot detection map from ML predictions
        self.logger.info(f"Plotting detection map for {det_type}")
        axes = plot_detections(
            axes=axes,
            axis_idx=current_axis_index,
            detection_map=detection_map,
            det_type=det_type,
            ylabel="Detection",
            xlabel="",
            sensor_freq=sensor_freq
        )

        save_figure(fig, filename)
        self.logger.info(f"Saved figure to {os.path.abspath(filename)}")

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
