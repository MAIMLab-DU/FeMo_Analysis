
import os
import joblib
import boto3
import tempfile
import numpy as np
from collections import defaultdict
from skimage.measure import label
from dataclasses import dataclass
from ..model.base import FeMoBaseClassifier
from ..logger import LOGGER
from ..data.pipeline import Pipeline
from ..eval.metrics import FeMoMetrics
from ..plot.plotter import FeMoPlotter
from ..data.process import Processor
from ..data.transforms import SensorFusion


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
    processor: Processor = None
    metrics: FeMoMetrics = None
    plotter = FeMoPlotter()

    @property
    def logger(self):
        return LOGGER

    def __init__(self,
                 classifier_path: str,
                 model_path: str,
                 pipeline_path: str,
                 processor_path: str,
                 metrics_path: str) -> None:
        
        self.classifier_path = classifier_path
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.processor_path = processor_path
        self.metrics_path = metrics_path

    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.classifier is None:
            self.classifier = joblib.load(self.classifier_path)
        self.classifier.load_model(self.model_path)
        return self.classifier
    
    def get_pipeline(self):
        """Get the pipeline object for this instance, loading it if it's not already loaded."""
        if self.pipeline is None:
            self.pipeline = joblib.load(self.pipeline_path)
            self.pipeline.inference = True
        return self.pipeline
    
    def get_processor(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.processor is None:
            self.processor = joblib.load(self.processor_path)
        return self.processor
    
    def get_metrics(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.metrics is None:
            self.metrics = joblib.load(self.metrics_path)
        return self.metrics
    
    def _pre_hiccup_removal(self,
                       filename: str,
                       y_pred: np.ndarray,
                       preprocessed_data: dict,
                       fm_dict: dict,
                       scheme_dict: dict):
        
        self.logger.info(f"Calculating meta info for {filename}")
        sensor_freq = self.pipeline.stages[0].sensor_freq
        fm_dilation = self.pipeline.stages[2].fm_dilation

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

        #Now get the reduced detection_map
        reduced_detection_map = ml_map * fm_dict['fm_map']  # Reduced, because of the new dilation length
        reduced_detection_map_labeled = label(reduced_detection_map)

        n_movements = np.max(reduced_detection_map_labeled)

        # Keeps only the detected segments
        detection_only = reduced_detection_map[reduced_detection_map == 1]
        # Dilation length on sides of each detection is removed and converted to minutes
        total_FM_duration = (len(detection_only) / sensor_freq - n_movements * fm_dilation / 2) / 60

        onset_interval = []
        for j in range(1, n_movements):
            onset1 = np.where(reduced_detection_map_labeled == j)[0][0]  # Sample no. corresponding to start of the label
            onset2 = np.where(reduced_detection_map_labeled == j + 1)[0][0]  # Sample no. corresponding to start of the next label
            onset_interval.append( (onset2 - onset1) / sensor_freq ) # onset to onset interval in seconds

        duration_trimmed_data_files = len(preprocessed_data['sensor_1']) / 1024 / 60 # in minutes
        # Time fetus was not moving
        total_nonFM_duration = duration_trimmed_data_files - total_FM_duration
        data = InferenceMetaInfo(
            fileName=os.path.basename(filename),
            numKicks=n_movements,
            totalFMDuration=total_FM_duration,
            totalNonFMDuration=total_nonFM_duration,
            onsetInterval=onset_interval
        )

        return data, ml_map

    def predict(self, filename: str, bucket_name: str = None, remove_hiccups: bool = False):
        """For the input, perform predictions and return results.

            Args:
                filename (str): The path to the file that needs to be processed.
                bucket_name (str, optional): The name of the S3 bucket where the file is located. Defaults to None.
        """
    
        # Helper function to process the file
        def process_file(file_path: str):
            inference_output = defaultdict()

            pipeline = self.get_pipeline()
            pipeline_output = pipeline.process(filename=file_path)
            
            processor = self.get_processor()
            X_extracted = pipeline_output['extracted_features']['features']
            X_norm_ranked = processor.predict(X_extracted)
            
            clf = self.get_model()
            y_pred = clf.predict(X_norm_ranked)

            data, ml_map = self._pre_hiccup_removal(
                filename=file_path,
                y_pred=y_pred,
                preprocessed_data=pipeline_output['preprocessed_data'],
                fm_dict=pipeline_output['fm_dict'],
                scheme_dict=pipeline_output['scheme_dict']
            )
            inference_output['pre_hiccup_removal'] = {
                'data': data,
                'pipeline_output': pipeline_output,
                'ml_map': ml_map
            }

            if remove_hiccups:
                ...
            
            return data, pipeline_output, ml_map
        
        # Check if file is in S3 and download if necessary
        if bucket_name:
            s3_client = boto3.resource('s3')
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                s3_client.Bucket(bucket_name).download_file(filename, temp_file.name)
                LOGGER.info(f"Downloaded {filename} from S3 bucket {bucket_name} to {temp_file.name}")
                result = process_file(temp_file.name)
                os.remove(temp_file.name)
        else:
            result = process_file(filename)
        
        return result  
    
    
    def save_pred_plots(self, pipeline_output: dict, ml_map: np.ndarray, filename: str):
        plt_cfg = {
            'figsize': [16, 15],
            'x_unit': 'min'
        }
        fig, axes = self.plotter.create_figure(
            figsize=plt_cfg['figsize']
        )
        i = 0
        for sensor_type in self.plotter.sensor_map.keys():
            for j in range(len(self.plotter.sensor_map[sensor_type])):
                sensor_name = f"{sensor_type}_{j+1}"
                axes = self.plotter.plot_sensor_data(
                    axes=axes,
                    axis_idx=i,
                    data=pipeline_output['preprocessed_data'][self.plotter.sensor_map[sensor_type][j]],
                    sensor_name=sensor_name,
                    x_unit=plt_cfg.get('x_unit', 'min')
                )
                i += 1

        # TODO: might be better to have this more configurable
        fusion_stage: SensorFusion = self.pipeline.stages[3]
        desired_scheme = fusion_stage.desired_scheme

        axes = self.plotter.plot_detections(
            axes=axes,
            axis_idx=i,
            detection_map=pipeline_output['scheme_dict']['user_scheme'],
            det_type=f"At least {desired_scheme[1]} {desired_scheme[0]} Sensor Events",
            ylabel='Detection',
            xlabel='',
            x_unit=plt_cfg.get('x_unit', 'min')
        )
        axes = self.plotter.plot_detections(
            axes=axes,
            axis_idx=i+1,
            detection_map=ml_map,
            det_type='Fetal movement',
            ylabel='Detection',
            xlabel=f"Time ({plt_cfg.get('x_unit', 'min')})",
            x_unit=plt_cfg.get('x_unit', 'min')
        )

        self.plotter.save_figure(fig, filename)