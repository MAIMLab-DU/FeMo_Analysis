import os
import boto3
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from femo.logger import LOGGER
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
        "numBoutsPerHour": (data.numKicks*60) / (data.totalFMDuration+data.totalNonFMDuration),
        "meanDurationFM_s": data.totalFMDuration*60/data.numKicks if data.numKicks > 0 else 0,
        "medianOnsetInterval_s": np.median(data.onsetInterval),
        "activeTimeFM_%": (data.totalFMDuration/(data.totalFMDuration+data.totalNonFMDuration))*100
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


@app.get("/results/{job_id}")
async def get_result(job_id: str):
    print(f"Received request for job_id: {job_id}")
    
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    table = dynamodb.Table(os.getenv('DYNAMODB_TABLE'))
    
    try:
        response = table.get_item(Key={'job_id': job_id}, ConsistentRead=True)
        LOGGER.debug(f"Response from DynamoDB: {response}")
    except Exception as e:
        LOGGER.error(f"Error fetching from DynamoDB: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    if "Item" in response:
        return {"message": "Job completed", "result": response['Item']['result']}
    else:
        LOGGER.warning(f"No item found for job_id: {job_id}")
        raise HTTPException(status_code=404, detail="Result not found for job_id: {}".format(job_id))


@app.post("/invocations")
async def transformation(data: RequestData, background_tasks: BackgroundTasks):
    # Generate a unique job ID
    job_id, timestamp = generate_uuid_and_timestamp()
    
    # Extract the input data
    input_s3_path = data.s3_path
    if not input_s3_path:
        raise HTTPException(status_code=400, detail="S3 path to .dat file should be provided")

    # Schedule the background task
    background_tasks.add_task(run_inference_job, job_id, timestamp, input_s3_path)

    # Immediately return job ID to the client
    return {"message": "Job started", "job_id": job_id}
