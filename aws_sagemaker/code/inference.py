import os
import boto3
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from femo.logger import LOGGER
from typing import Optional
from femo.inference import PredictionService
from utils import (
    convert_floats_to_decimal,
    extract_s3_details,
    generate_uuid_and_timestamp
)

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

app = FastAPI()

# Define the input data structure for the request body
class RequestData(BaseModel):
    s3_path: str
    job_id: Optional[str] = None


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
pred_service = PredictionService(
    os.path.join(model_path, 'model.joblib'),
    os.path.join(model_path, 'pipeline.joblib'),
    os.path.join(model_path, 'processor.joblib'),
    os.path.join(model_path, 'metrics.joblib')
)




def run_inference_job(job_id: str, timestamp: str, s3_path: str):
    """_summary_

    Args:
        job_id (str): Unique inference job id
        s3_path (str): S3 path to .dat file
    """
    bucket_name, filename = extract_s3_details(s3_path)

    data, _, _ = pred_service.predict(filename, bucket_name)

    prediction = {
        "numBoutsPerHour": (data.numKicks*60) / (data.totalFMDuration+data.totalNonFMDuration) if data.numKicks > 0 else 0,
        "meanDurationFM_s": data.totalFMDuration*60/data.numKicks if data.numKicks > 0 else 0,
        "medianOnsetInterval_s": np.median(data.onsetInterval) if data.numKicks > 0 else 0,
        "activeTimeFM_%": (data.totalFMDuration/(data.totalFMDuration+data.totalNonFMDuration))*100 if data.numKicks > 0 else 0,
    }

    prediction = convert_floats_to_decimal(prediction)
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    table = dynamodb.Table(os.getenv('DYNAMODB_TABLE')) 
    # Store the result in DynamoDB or S3 after completion
    table.put_item(Item={"job_id": job_id, "fileName": filename, "timeStamp": timestamp, "result": prediction})
    LOGGER.info(f"DB insertion success for {job_id = }")


@app.get("/ping")
async def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = all([
        pred_service.get_model() is not None,
        pred_service.get_pipeline() is not None,
        pred_service.get_processor() is not None,
        pred_service.get_metrics() is not None
    ])
    status = 200 if health else 404
    return {"status": status}


@app.post("/invocations")
async def transformation(data: RequestData, background_tasks: BackgroundTasks):
    # Validate and extract S3 path
    if not data.s3_path:
        raise HTTPException(status_code=400, detail="S3 path to .dat file must be provided")

    dynamodb = boto3.resource('dynamodb', region_name=os.getenv('AWS_DEFAULT_REGION', 'eu-north-1'))
    table = dynamodb.Table(os.getenv('DYNAMODB_TABLE'))

    # Case 1: No job_id provided - initiate a new job
    if data.job_id is None:
        job_id, timestamp = generate_uuid_and_timestamp()
        background_tasks.add_task(run_inference_job, job_id, timestamp, data.s3_path)
        LOGGER.info(f"Started new job with job_id: {job_id}")
        return {"message": "Job started, completion in ~120 seconds", "job_id": job_id}

    # Case 2: job_id provided - fetch the result from DynamoDB
    try:
        response = table.get_item(Key={'job_id': data.job_id}, ConsistentRead=True)
        LOGGER.debug(f"DynamoDB response for job_id {data.job_id}: {response}")
    except Exception as e:
        LOGGER.error(f"Error fetching from DynamoDB for job_id {data.job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # Check if item exists in the response
    if 'Item' in response:
        LOGGER.info(f"Result found for job_id: {data.job_id}")
        item = response['Item']
        return {"message": "Job completed", "job_id": data.job_id, "fileName": item.get('fileName'), "result": item.get('result')}
    
    # If no item found, return 404
    LOGGER.warning(f"No result found for job_id: {data.job_id}")
    raise HTTPException(status_code=404, detail=f"Result not found for job_id: {data.job_id}")
