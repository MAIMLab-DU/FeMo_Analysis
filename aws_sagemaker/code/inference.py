import os
import boto3
import numpy as np
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from femo.logger import LOGGER
from femo.inference import PredictionService, InferenceMetaInfo
from utils import (
    convert_floats_to_decimal,
    extract_s3_details,
)

prefix: str = '/opt/ml/'
model_path: str = os.path.join(prefix, 'model')

app = FastAPI()

# Define the input data structure for the request body
class RequestData(BaseModel):
    s3Key: str
    jobId: str
    timeStamp: str
    removeHiccups: Optional[bool] = False


def validate_model_files(model_path: str) -> bool:
    """Validate that all required model files exist and are readable."""
    required_files = [
        'classifier.joblib',
        'model.joblib', 
        'pipeline.joblib',
        'tsfel_processor.joblib',
        'crafted_processor.joblib',
        'metrics.joblib'
    ]
    
    for file_name in required_files:
        file_path = os.path.join(model_path, file_name)
        if not os.path.exists(file_path):
            LOGGER.error(f"Missing model file: {file_path}")
            return False
        if not os.access(file_path, os.R_OK):
            LOGGER.error(f"Cannot read model file: {file_path}")
            return False
        LOGGER.info(f"Found model file: {file_path}")
    
    return True


def create_prediction_service(model_path: str) -> Optional[PredictionService]:
    """Create and validate the PredictionService instance."""
    try:
        if not validate_model_files(model_path):
            return None
            
        # Try to load the PredictionService with better error handling
        pred_service = PredictionService(
            os.path.join(model_path, 'classifier.joblib'),
            os.path.join(model_path, 'model.joblib'),
            os.path.join(model_path, 'pipeline.joblib'),
            os.path.join(model_path, 'processor.joblib'),
            os.path.join(model_path, 'metrics.joblib'),
            {}  # empty dict for pred_cfg
        )
        
        # Validate that all components can be loaded successfully
        try:
            model = pred_service.get_model()
            if model is None:
                LOGGER.error("Model failed to load")
                return None
        except Exception as e:
            LOGGER.error(f"Error loading model: {e}")
            return None
            
        try:
            pipeline = pred_service.get_pipeline()
            if pipeline is None:
                LOGGER.error("Pipeline failed to load")
                return None
        except Exception as e:
            LOGGER.error(f"Error loading pipeline: {e}")
            return None
            
        try:
            metrics = pred_service.get_metrics()
            if metrics is None:
                LOGGER.error("Metrics failed to load")
                return None
        except Exception as e:
            LOGGER.error(f"Error loading metrics: {e}")
            return None
            
        LOGGER.info("PredictionService initialized successfully")
        return pred_service
        
    except Exception as e:
        LOGGER.error(f"Error creating PredictionService: {e}")
        return None


# Initialize the prediction service with validation
pred_service: Optional[PredictionService] = create_prediction_service(model_path)

if pred_service is None:
    LOGGER.error("Failed to initialize PredictionService")

dynamodb = boto3.resource('dynamodb', region_name=os.getenv('AWS_DEFAULT_REGION', 'eu-north-1'))
table = dynamodb.Table(os.getenv('DYNAMODB_TABLE'))


def update_item_in_dynamodb(table, s3_key: str, time_stamp: str, update_data: Dict[str, Any]) -> Optional[Dict]:
    """
    Update an item in a DynamoDB table with specific s3Key and timeStamp.

    Args:
        table: DynamoDB table resource
        s3_key (str): The s3Key value to identify the item.
        time_stamp (str): The timeStamp value to identify the item.
        update_data (dict): The attributes and their new values to update.

    Returns:
        dict: Response from the update_item operation.
    """

    # Build the update expression
    update_expression: str = "SET " + ", ".join(f"#{k} = :{k}" for k in update_data.keys())
    expression_attribute_names: Dict[str, str] = {f"#{k}": k for k in update_data.keys()}
    expression_attribute_values: Dict[str, Any] = {f":{k}": v for k, v in update_data.items()}

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


def run_inference_job(job_id: str, time_stamp: str, s3_path: str, remove_hiccups: bool = False) -> None:
    """Run inference job with better error handling.

    Args:
        job_id (str): Unique inference job id
        time_stamp (str): Timestamp for the job
        s3_path (str): S3 path to .dat file
        remove_hiccups (bool): Whether to perform hiccup removal analysis
    """
    if pred_service is None:
        error_msg = "PredictionService not initialized"
        LOGGER.error(error_msg)
        update_item_in_dynamodb(table, s3_path, time_stamp, {"status": "Failed", "result": error_msg})
        return

    try:
        bucket_name, file_name = extract_s3_details(s3_path)
        LOGGER.info(f"Starting inference for bucket: {bucket_name}, file: {file_name}, remove_hiccups: {remove_hiccups}")
        
        pred_output = pred_service.predict(file_name, bucket_name, remove_hiccups)
        pre_removal_data: InferenceMetaInfo = pred_output['pre_hiccup_removal']['data']
        
        result = {
            "pre_hiccup": {                
                "numKicks": int(pre_removal_data.numKicks),
                "totalFMDuration": float(pre_removal_data.totalFMDuration),
                "totalNonFMDuration": float(pre_removal_data.totalNonFMDuration),
                "medianOnsetInterval_s": float(np.median(pre_removal_data.onsetInterval) if pre_removal_data.numKicks > 0 else 0),
            }
        }
        
        # Only include post-hiccup data if hiccup removal was performed
        if remove_hiccups and 'post_hiccup_removal' in pred_output:
            post_removal_data: InferenceMetaInfo = pred_output['post_hiccup_removal']['data']
            result["post_hiccup"] = {
                "numKicks": int(post_removal_data.numKicks),
                "totalFMDuration": float(post_removal_data.totalFMDuration),
                "totalNonFMDuration": float(post_removal_data.totalNonFMDuration),
                "medianOnsetInterval_s": float(np.median(post_removal_data.onsetInterval) if post_removal_data.numKicks > 0 else 0),
            }
        
        result = {k: convert_floats_to_decimal(result[k]) for k in result.keys()}
    
    except Exception as e:
        LOGGER.error(f"Error running inference job: {e}", exc_info=True)
        update_item_in_dynamodb(table, s3_path, time_stamp, {"status": "Failed", "result": str(e)})
        return
    
    # Store the result in DynamoDB or S3 after completion
    update_item_in_dynamodb(table, s3_path, time_stamp, {"status": "Completed", "result": result})
    LOGGER.info(f"DB insertion success for {job_id = }")


@app.get("/ping")
async def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    if pred_service is None:
        return {"status": 404}
    
    try:
        health = all([
            pred_service.get_model() is not None,
            pred_service.get_pipeline() is not None,
            pred_service.get_metrics() is not None
        ])
        status = 200 if health else 404
        return {"status": status}
    except Exception as e:
        LOGGER.error(f"Health check failed: {e}")
        return {"status": 404}


@app.post("/invocations")
async def transformation(data: RequestData, background_tasks: BackgroundTasks):
    # Validate and extract S3 path
    if not data.s3Key:
        raise HTTPException(status_code=400, detail="S3 path to .dat file must be provided")
    
    # Validate removeHiccups parameter
    remove_hiccups = data.removeHiccups if data.removeHiccups is not None else True
    
    table.put_item(Item={
        "s3Key": data.s3Key, 
        "timeStamp": data.timeStamp, 
        "jobId": data.jobId, 
        "status": "InProgress"
    })
    
    background_tasks.add_task(run_inference_job, data.jobId, data.timeStamp, data.s3Key, remove_hiccups)
    LOGGER.info(f"Started new job with job_id: {data.jobId}, remove_hiccups: {remove_hiccups}")
    
    return {
        "message": "Job started, completion in ~120 seconds",
        "s3Key": data.s3Key, 
        "timeStamp": data.timeStamp, 
        "jobId": data.jobId
    }
