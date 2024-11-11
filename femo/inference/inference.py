
import os
import joblib
import boto3
import tempfile
import numpy as np
from ..model.base import FeMoBaseClassifier
from ..logger import LOGGER
from ..data.pipeline import Pipeline
from ..eval.metrics import FeMoMetrics
from ..plot.plotter import FeMoPlotter
from ..data.process import Processor
from ..data.transforms import SensorFusion


class PredictionService(object):
    classifier: FeMoBaseClassifier = None
    pipeline: Pipeline = None
    processor: Processor = None
    metrics: FeMoMetrics = None
    plotter = FeMoPlotter()

    def __init__(self,
                 model_path: str,
                 pipeline_path: str,
                 processor_path: str,
                 metrics_path: str) -> None:
        
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.processor_path = processor_path
        self.metrics_path = metrics_path

    
    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.classifier is None:
            self.classifier = joblib.load(self.model_path)
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
    

    
    def predict(self, filename: str, bucket_name: str = None):
        """For the input, perform predictions and return results.

            Args:
                filename (str): The path to the file that needs to be processed.
                bucket_name (str, optional): The name of the S3 bucket where the file is located. Defaults to None.
        """
    
        # Helper function to process the file
        def process_file(file_path: str):
            pipeline = self.get_pipeline()
            pipeline_output = pipeline.process(filename=file_path)
            
            processor = self.get_processor()
            X_extracted = pipeline_output['extracted_features']['features']
            X_norm_ranked = processor.predict(X_extracted)
            
            clf = self.get_model()
            y_pred = clf.predict(X_norm_ranked)
            
            metrics = self.get_metrics()
            data, ml_map = metrics.calc_meta_info(
                filename=file_path,
                y_pred=y_pred,
                preprocessed_data=pipeline_output['preprocessed_data'],
                fm_dict=pipeline_output['fm_dict'],
                scheme_dict=pipeline_output['scheme_dict']
            )
            
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