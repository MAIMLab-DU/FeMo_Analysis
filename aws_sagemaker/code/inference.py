import os
import boto3
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from femo.logger import LOGGER
from femo.inference import PredictionService, InferenceMetaInfo
from utils import (
    convert_floats_to_decimal,
    extract_s3_details,
)

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

app = FastAPI()

# Define the input data structure for the request body
class RequestData(BaseModel):
    s3Key: str
    jobId: str
    timeStamp: str


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
pred_service = PredictionService(
    os.path.join(model_path, 'classifier.joblib'),
    os.path.join(model_path, 'model.joblib'),
    os.path.join(model_path, 'pipeline.joblib'),
    os.path.join(model_path, 'processor.joblib'),
    os.path.join(model_path, 'metrics.joblib'),
    {}  # empty dict for pred_cfg, couldn't care less for a better approach
)

dynamodb = boto3.resource('dynamodb', region_name=os.getenv('AWS_DEFAULT_REGION', 'eu-north-1'))
table = dynamodb.Table(os.getenv('DYNAMODB_TABLE'))


def update_item_in_dynamodb(table, s3_key, time_stamp, update_data):
    """
    Update an item in a DynamoDB table with specific s3Key and timeStamp.

    Args:
        table_name (str): The name of the DynamoDB table.
        s3_key (str): The s3Key value to identify the item.
        time_stamp (str): The timeStamp value to identify the item.
        update_data (dict): The attributes and their new values to update.

    Returns:
        dict: Response from the update_item operation.
    """

    # Build the update expression
    update_expression = "SET " + ", ".join(f"#{k} = :{k}" for k in update_data.keys())
    expression_attribute_names = {f"#{k}": k for k in update_data.keys()}
    expression_attribute_values = {f":{k}": v for k, v in update_data.items()}

    try:
        # Update the item
        response = table.update_item(
            Key={
                "s3Key": s3_key,
                "timeStamp": time_stamp
            },
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values,
            ReturnValues="UPDATED_NEW"
        )
        return response
    except Exception as e:
        LOGGER.error(f"Error updating item: {e}")
        raise


def run_inference_job(job_id: str, time_stamp: str, s3_path: str):
    """_summary_

    Args:
        job_id (str): Unique inference job id
        s3_path (str): S3 path to .dat file
    """
    bucket_name, file_name = extract_s3_details(s3_path)

    try:
        pred_output = pred_service.predict(file_name, bucket_name, True)
        pre_removal_data: InferenceMetaInfo = pred_output['pre_hiccup_removal']['data']
        post_removal_data: InferenceMetaInfo = pred_output['post_hiccup_removal']['data']
        result = {
            "pre_hiccup": {                
                "numKicks": int(pre_removal_data.numKicks),
                "totalFMDuration": float(pre_removal_data.totalFMDuration),
                "totalNonFMDuration": float(pre_removal_data.totalNonFMDuration),
                "medianOnsetInterval_s": float(np.median(pre_removal_data.onsetInterval) if pre_removal_data.numKicks > 0 else 0),
            },
            "post_hiccup": {
                "numKicks": int(post_removal_data.numKicks),
                "totalFMDuration": float(post_removal_data.totalFMDuration),
                "totalNonFMDuration": float(post_removal_data.totalNonFMDuration),
                "medianOnsetInterval_s": float(np.median(post_removal_data.onsetInterval) if post_removal_data.numKicks > 0 else 0),
            }
        }
        result = {k: convert_floats_to_decimal(result[k]) for k in result.keys()}
    
    except Exception as e:
        LOGGER.error(f"Error running inference job: {e}")
        update_item_in_dynamodb(table, s3_path, time_stamp, {"status": "Failed", "result": str(e)})
        return
    
    # Store the result in DynamoDB or S3 after completion
    update_item_in_dynamodb(table, s3_path, time_stamp, {"status": "Completed", "result": result})
    LOGGER.info(f"DB insertion success for {job_id = }")


@app.get("/ping")
async def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = all([
        pred_service.get_model() is not None,
        pred_service.get_pipeline() is not None,
        pred_service.get_metrics() is not None
    ])
    status = 200 if health else 404
    return {"status": status}


@app.post("/invocations")
async def transformation(data: RequestData, background_tasks: BackgroundTasks):
    # Validate and extract S3 path
    if not data.s3Key:
        raise HTTPException(status_code=400, detail="S3 path to .dat file must be provided")
    
    table.put_item(Item={"s3Key": data.s3Key, "timeStamp": data.timeStamp, "jobId": data.jobId, "status": "InProgress"})
    background_tasks.add_task(run_inference_job, data.jobId, data.timeStamp, data.s3Key)
    LOGGER.info(f"Started new job with job_id: {data.jobId}")
    return {"message": "Job started, completion in ~120 seconds",
            "s3Key":data.s3Key, "timeStamp": data.timeStamp, "jobId": data.jobId}
