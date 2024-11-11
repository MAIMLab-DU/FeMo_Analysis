from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import boto3
import tempfile
import joblib
import zipfile
import numpy as np
import pandas as pd
from femo.model.base import FeMoBaseClassifier
from femo.logger import LOGGER
from femo.data.pipeline import Pipeline
from femo.eval.metrics import FeMoMetrics
from femo.plot.plotter import FeMoPlotter
from femo.data.process import Processor
from femo.data.transforms import SensorFusion
from urllib.parse import urlparse
import os

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

app = FastAPI()

# Define the input data structure for the request body
class RequestData(BaseModel):
    s3_path: str


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class PredictionService(object):
    classifier: FeMoBaseClassifier = None
    pipeline: Pipeline = None
    processor: Processor = None
    metrics: FeMoMetrics = None
    plotter = FeMoPlotter()

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.classifier is None:
            cls.classifier = joblib.load(os.path.join(model_path, 'model.joblib'))
        return cls.classifier
    
    @classmethod
    def get_pipeline(cls):
        """Get the pipeline object for this instance, loading it if it's not already loaded."""
        if cls.pipeline is None:
            cls.pipeline = joblib.load(os.path.join(model_path, 'pipeline.joblib'))
            cls.pipeline.inference = True
        return cls.pipeline
    
    @classmethod
    def get_processor(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.processor is None:
            cls.processor = joblib.load(os.path.join(model_path, 'processor.joblib'))
        return cls.processor
    
    @classmethod
    def get_metrics(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.metrics is None:
            cls.metrics = joblib.load(os.path.join(model_path, 'metrics.joblib'))
        return cls.metrics
    

    @classmethod
    def predict(cls, input_s3_path):
        """For the input, do the predictions and return them.

        Args:
            input_s3_path (str): The S3 path to the file that needs to be processed.
        """
        # Initialize S3 client
        s3_client = boto3.resource('s3')

        # Parse the S3 URI to get bucket name and file key
        parsed_url = urlparse(input_s3_path)
        bucket_name = parsed_url.netloc
        file_key = parsed_url.path.lstrip('/')

        # Download the file from S3 to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Download the S3 file to the temp file
            s3_client.Bucket(bucket_name).download_file(file_key, temp_file.name)
            LOGGER.info(f"Downloaded {file_key} from S3 bucket {bucket_name} to {temp_file.name}")

            # Now use the downloaded file path for processing
            pipeline = cls.get_pipeline()
            pipeline_output = pipeline.process(filename=temp_file.name)
            X_extracted = pipeline_output['extracted_features']['features']

            processor = cls.get_processor()
            X_norm_ranked = processor.predict(X_extracted)
            
            clf = cls.get_model()
            y_pred = clf.predict(X_norm_ranked)

            metrics = cls.get_metrics()
            metainfo_dict, ml_map = metrics.calc_meta_info(
                filename=temp_file.name,
                y_pred=y_pred,
                preprocessed_data=pipeline_output['preprocessed_data'],
                fm_dict=pipeline_output['fm_dict'],
                scheme_dict=pipeline_output['scheme_dict']
            )
            metainfo_dict['File Name'] = [file_key]

            # Clean up the temp file after processing
            os.remove(temp_file.name)

        return pd.DataFrame(metainfo_dict), pipeline_output, ml_map    
    
    @classmethod
    def plot_predictions(cls, pipeline_output: dict, ml_map: np.ndarray):
        plt_cfg = {
            'figsize': [16, 15],
            'x_unit': 'min'
        }
        cls.plotter.create_figure(
            figsize=plt_cfg['figsize']
        )
        i = 0
        for sensor_type in cls.plotter.sensor_map.keys():
            for j in range(len(cls.plotter.sensor_map[sensor_type])):
                sensor_name = f"{sensor_type}_{j+1}"
                cls.plotter.plot_sensor_data(
                    axis_idx=i,
                    data=pipeline_output['preprocessed_data'][cls.plotter.sensor_map[sensor_type][j]],
                    sensor_name=sensor_name,
                    x_unit=plt_cfg.get('x_unit', 'min')
                )
                i += 1

        # TODO: might be better to have this more configurable
        fusion_stage: SensorFusion = cls.pipeline.stages[3]
        desired_scheme = fusion_stage.desired_scheme

        cls.plotter.plot_detections(
            axis_idx=i,
            detection_map=pipeline_output['scheme_dict']['user_scheme'],
            det_type=f"At least {desired_scheme[1]} {desired_scheme[0]} Sensor Events",
            ylabel='Detection',
            xlabel='',
            x_unit=plt_cfg.get('x_unit', 'min')
        )
        cls.plotter.plot_detections(
            axis_idx=i+1,
            detection_map=ml_map,
            det_type='Fetal movement',
            ylabel='Detection',
            xlabel=f"Time ({plt_cfg.get('x_unit', 'min')})",
            x_unit=plt_cfg.get('x_unit', 'min')
        )

        return cls.plotter.fig


@app.get("/ping")
async def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = all([
        PredictionService.get_model() is not None,
        PredictionService.get_pipeline() is not None,
        PredictionService.get_processor() is not None,
        PredictionService.get_metrics() is not None
    ])

    status = 200 if health else 404
    return {"status": status}


@app.post("/invocations")
async def transformation(data: RequestData):
    """Do an inference on a single batch of data, save the CSV and image to temp files, zip them, and send as StreamingResponse."""
    input_s3_path = data.s3_path

    if not input_s3_path:
        raise HTTPException(status_code=400, detail="S3 path to .dat file should be provided")

    LOGGER.info(f'Performing prediction on {input_s3_path}')
    prediction, pipeline_output, ml_map = PredictionService.predict(input_s3_path)

    # Save the prediction as CSV in a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix='.csv') as csv_tempfile:
        prediction.to_csv(csv_tempfile, index=False)
        csv_tempfile_path = csv_tempfile.name

    # Generate the image and save it as a temporary file
    image_output = PredictionService.plot_predictions(pipeline_output, ml_map)
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.png') as image_tempfile:
        image_output.savefig(image_tempfile, format='PNG')
        image_tempfile_path = image_tempfile.name

    # Create a zip file containing the CSV and image files
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as temp_zip:
        # Add the CSV and image file
        temp_zip.write(csv_tempfile_path, 'prediction.csv') 
        temp_zip.write(image_tempfile_path, 'plot.png')

    zip_io.seek(0)  # Go back to the start of the BytesIO object

    # Clean up the temporary files
    os.remove(csv_tempfile_path)
    os.remove(image_tempfile_path)

    # Return the zip file as a StreamingResponse
    return StreamingResponse(zip_io, media_type="application/x-zip-compressed", status_code=200,
                             headers={"Content-Disposition": "attachment; filename=predictions.zip"})
