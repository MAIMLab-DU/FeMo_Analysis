import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Request, Response, status, HTTPException
from pydantic import BaseModel, Field
from femo.logger import LOGGER
from femo.inference import PredictionService, InferenceMetaInfo
from utils import (
    extract_s3_details,
    extract_events
)

# SageMaker model paths
prefix: str = os.getenv('MODEL_PREFIX', '/opt/ml/')
model_path: str = os.path.join(prefix, 'model')

app = FastAPI()


class InferenceRequest(BaseModel):
    """Request model for inference requests."""
    jobId: str = Field(..., description="Unique identifier for the inference job")
    requestTime: str = Field(..., description="Timestamp when the request was created (ISO format)")
    s3Key: str = Field(..., description="S3 key path to the input file")
    removeHiccups: bool = Field(False, description="Whether to perform hiccup removal")
    includeFullIntervals: bool = Field(False, description="Whether to include full interval arrays in metadata")
    extractEvents: bool = Field(True, description="Whether to extract events (True) or return metadata (False)")
    
    class Config:
        schema_extra = {
            "example": {
                "jobId": "job-123e4567-e89b-12d3-a456-426614174000",
                "requestTime": "2025-06-04T14:30:00.000Z",
                "s3Key": "input-data/recording-2024-06-04.dat",
                "removeHiccups": False,
                "includeFullIntervals": False,
                "extractEvents": True
            }
        }


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
            
        pred_service = PredictionService(
            os.path.join(model_path, 'classifier.joblib'),
            os.path.join(model_path, 'model.joblib'),
            os.path.join(model_path, 'pipeline.joblib'),
            os.path.join(model_path, 'processor.joblib'),
            os.path.join(model_path, 'metrics.joblib'),
            {}  # empty dict for pred_cfg
        )
        
        # Validate components
        if not all([
            pred_service.get_model() is not None,
            pred_service.get_pipeline() is not None,
            pred_service.get_metrics() is not None
        ]):
            return None
            
        LOGGER.info("PredictionService initialized successfully")
        return pred_service
        
    except Exception as e:
        LOGGER.error(f"Error creating PredictionService: {e}")
        return None


# Initialize the prediction service
pred_service: Optional[PredictionService] = create_prediction_service(model_path)
if pred_service is None:
    LOGGER.error("Failed to initialize PredictionService")


def build_inference_metainfo_data(pred_output: dict, remove_hiccups: bool, include_full_intervals: bool = False) -> dict:
    """Build inference meta info data from prediction output."""
    pre_removal_data: InferenceMetaInfo = pred_output['pre_hiccup_removal']['data']
    
    # Calculate summary statistics for onsetInterval
    onset_intervals = pre_removal_data.onsetInterval
    onset_stats = {
        "count": len(onset_intervals),
        "mean": float(np.mean(onset_intervals)) if onset_intervals else 0.0,
        "std": float(np.std(onset_intervals)) if onset_intervals else 0.0,
        "min": float(np.min(onset_intervals)) if onset_intervals else 0.0,
        "max": float(np.max(onset_intervals)) if onset_intervals else 0.0,
        "median": float(np.median(onset_intervals)) if onset_intervals else 0.0,
        "p25": float(np.percentile(onset_intervals, 25)) if onset_intervals else 0.0,
        "p75": float(np.percentile(onset_intervals, 75)) if onset_intervals else 0.0
    }
    
    result = {
        "pre_hiccup": {                
            "numKicks": int(pre_removal_data.numKicks),
            "total_duration_mins": float(pre_removal_data.totalFMDuration + pre_removal_data.totalNonFMDuration),
            "totalFMDuration_mins": float(pre_removal_data.totalFMDuration),
            "totalNonFMDuration_mins": float(pre_removal_data.totalNonFMDuration),
            "onsetInterval_sec_stats": onset_stats,
        }
    }
    
    # Include full array only if requested
    if include_full_intervals:
        result["pre_hiccup"]["onsetInterval_sec"] = [float(x) for x in pre_removal_data.onsetInterval]
    
    # Only include post-hiccup data if hiccup removal was performed
    if remove_hiccups and 'post_hiccup_removal' in pred_output:
        post_removal_data: InferenceMetaInfo = pred_output['post_hiccup_removal']['data']
        
        post_onset_intervals = post_removal_data.onsetInterval
        post_onset_stats = {
            "count": len(post_onset_intervals),
            "mean": float(np.mean(post_onset_intervals)) if post_onset_intervals else 0.0,
            "std": float(np.std(post_onset_intervals)) if post_onset_intervals else 0.0,
            "min": float(np.min(post_onset_intervals)) if post_onset_intervals else 0.0,
            "max": float(np.max(post_onset_intervals)) if post_onset_intervals else 0.0,
            "median": float(np.median(post_onset_intervals)) if post_onset_intervals else 0.0,
            "p25": float(np.percentile(post_onset_intervals, 25)) if post_onset_intervals else 0.0,
            "p75": float(np.percentile(post_onset_intervals, 75)) if post_onset_intervals else 0.0
        }
        
        result["post_hiccup"] = {
            "numKicks": int(post_removal_data.numKicks),
            "total_duration_mins": float(post_removal_data.totalFMDuration + post_removal_data.totalNonFMDuration),
            "totalFMDuration_mins": float(post_removal_data.totalFMDuration),
            "totalNonFMDuration_mins": float(post_removal_data.totalNonFMDuration),
            "onsetInterval_sec_stats": post_onset_stats,
        }
        
        if include_full_intervals:
            result["post_hiccup"]["onsetInterval_sec"] = [float(x) for x in post_removal_data.onsetInterval]
    
    return result


def build_inference_events_data(pred_output: dict, remove_hiccups: bool, sensor_freq: int = 1024) -> list[dict]:
    """Build inference events data from prediction output."""
    try:
        # Extract required data with validation
        pipeline_output = pred_output.get('pipeline_output', {})
        loaded_data = pipeline_output.get('loaded_data', {})
        header = loaded_data.get('header', {})
        start_time = header.get('start_time')
        
        if not start_time:
            raise ValueError("Missing start_time in pipeline output header")
        
        pre_hiccup = pred_output.get('pre_hiccup_removal', {})
        if not pre_hiccup:
            raise ValueError("Missing pre_hiccup_removal data in prediction output")
            
        time_per_sample = 1.0 / sensor_freq
        events = []
        
        # Define event types with validation
        event_types = {}
        
        # Maternal body movement events
        imu_map = pipeline_output.get('imu_map')
        if imu_map is not None:
            event_types['maternal_body_movement'] = np.array(imu_map)
        
        # Inferred kick events
        ml_map = pre_hiccup.get('ml_map')
        if ml_map is not None:
            event_types['inferred_kick'] = np.array(ml_map)
        
        # Hiccup/other events (only if hiccup removal was performed)
        if remove_hiccups:
            post_hiccup = pred_output.get('post_hiccup_removal', {})
            hiccup_map = post_hiccup.get('hiccup_map')
            if hiccup_map is not None:
                event_types['other'] = np.array(hiccup_map)
        
        # Add maternally sensed kicks if available
        sensation_map = pipeline_output.get('sensation_map')
        if sensation_map is not None and np.any(sensation_map):
            event_types['maternally_sensed_kick'] = np.array(sensation_map)
        
        # Extract events for each type
        for event_type, event_data in event_types.items():
            if event_data.size > 0:  # Only process non-empty arrays
                extracted_events = extract_events(event_data, start_time, time_per_sample, event_type)
                LOGGER.info(f"Extracted {len(extracted_events)} events for type '{event_type}'")
                events.extend(extracted_events)

        return events
        
    except Exception as e:
        LOGGER.error(f"Error building inference events data: {e}")
        raise ValueError(f"Failed to build inference events data: {e}")


def process_inference_request(request_data: InferenceRequest) -> str:
    """Core inference processing logic."""
    if pred_service is None:
        raise ValueError("PredictionService not initialized")

    start_time = time.time()
    bucket_name, file_name = extract_s3_details(request_data.s3Key)
    
    LOGGER.info(f"Starting inference for bucket: {bucket_name}, file: {file_name}, "
               f"jobId: {request_data.jobId}, remove_hiccups: {request_data.removeHiccups}, "
               f"extract_events: {request_data.extractEvents}")
    
    # Run prediction
    pred_output = pred_service.predict(file_name, bucket_name, request_data.removeHiccups)
    sensor_freq = pred_service.pipeline.get_stage('load').sensor_freq
    len_file = len(pred_output['pipeline_output']['loaded_data']) / sensor_freq
    
    # Build response based on request type
    if not request_data.extractEvents:
        # Return metadata
        inference_result = build_inference_metainfo_data(
            pred_output, 
            request_data.removeHiccups, 
            request_data.includeFullIntervals
        )
        output_type = "metadata"
    else:
        # Extract events
        events = build_inference_events_data(
            pred_output, 
            request_data.removeHiccups,
            sensor_freq=sensor_freq
        )
        
        inference_result = {
            "event_count": len(events),
            "events": events
        }
        output_type = "events"
    
    # Prepare complete result
    complete_result = {
        "jobId": request_data.jobId,
        "requestTime": request_data.requestTime,
        "s3Key": request_data.s3Key,
        "parameters": {
            "removeHiccups": request_data.removeHiccups,
            "includeFullIntervals": request_data.includeFullIntervals,
            "extractEvents": request_data.extractEvents
        },
        "fileDurationSeconds": round(len_file, 3),
        "processingTimeSeconds": round(time.time() - start_time, 2),
        "outputType": output_type,
        "result": inference_result,
        "status": "Completed",
        "responseTime": datetime.utcnow().isoformat() + "Z"
    }
    
    LOGGER.info(f"Inference completed for jobId: {request_data.jobId}, "
                f"processing_time: {complete_result['processingTimeSeconds']}s")
    
    return json.dumps(complete_result, default=str)


async def model(request: Request) -> str:
    """Process the inference request and return JSON result as string."""
    if pred_service is None:
        raise HTTPException(status_code=503, detail="PredictionService not initialized")

    try:
        # Parse request body
        request_body = await request.body()
        request_json = json.loads(request_body.decode('utf-8'))
        
        # Validate request data using Pydantic model
        try:
            request_data = InferenceRequest(**request_json)
        except Exception as validation_error:
            LOGGER.error(f"Request validation failed: {validation_error}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid request format: {validation_error}"
            )
        
        # Use the synchronous processing function
        return process_inference_request(request_data)

    except HTTPException:
        # Re-raise HTTP exceptions (like validation errors)
        raise
    except Exception as e:
        LOGGER.error(f"Error in inference: {e}", exc_info=True)
        
        # Extract jobId for error response if possible
        job_id = "unknown"
        time_stamp = datetime.utcnow().isoformat() + "Z"
        s3_key = "unknown"
        
        try:
            request_body = await request.body()
            request_json = json.loads(request_body.decode('utf-8'))
            job_id = request_json.get('jobId', 'unknown')
            time_stamp = request_json.get('requestTime', time_stamp)
            s3_key = request_json.get('s3Key', 'unknown')
        except Exception:
            pass  # Use defaults if we can't parse the request
        
        # Return error result as JSON string
        error_result = {
            "jobId": job_id,
            "requestTime": time_stamp,
            "s3Key": s3_key,
            "status": "Failed",
            "error": str(e),
            "responseTime": datetime.utcnow().isoformat() + "Z"
        }
        
        return json.dumps(error_result, default=str)


@app.get("/ping")
async def ping():
    """Health check endpoint - SageMaker compatible."""
    if pred_service is None:
        return {"status": 404}
    
    try:
        health = all([
            pred_service.get_model() is not None,
            pred_service.get_pipeline() is not None,
            pred_service.get_metrics() is not None
        ])
        return {"status": 200 if health else 404}
        
    except Exception as e:
        LOGGER.error(f"Health check failed: {e}")
        return {"status": 404}


@app.post("/invocations")
async def invocations(request: Request):
    """
    SageMaker inference endpoint for both realtime and async inference.
    """
    model_resp = await model(request)
    
    return Response(
        content=model_resp,
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )