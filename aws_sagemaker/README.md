# AWS SageMaker FeMo Inference

AWS SageMaker implementation for FeMo (Fetal Movement) analysis with support for both synchronous and asynchronous inference.

## Overview

This module provides:
- **Training Pipeline**: Automated ML pipeline for training FeMo models
- **Async Inference Endpoint**: Scalable inference service for processing fetal movement data
- **Docker Container**: Production-ready containerized inference service

## Architecture

```
Data → Feature Extract → Preprocess → Train → Evaluate → Model Registry
                                                     ↓
                                            Async Inference Endpoint
```

## Quick Start

### 1. Build and Deploy Pipeline

```bash
# Run training pipeline
./run_pipeline.sh -m manifest.json

# Force feature re-extraction
./run_pipeline.sh -f -m manifest.json

# Local development mode
./run_pipeline.sh -l -m manifest.json
```

### 2. Async Inference Usage

**Request Format:**
```json
{
  "jobId": "unique-job-id",
  "requestTime": "2025-06-04T14:30:00.000Z",
  "s3Key": "input-data/recording.dat",
  "removeHiccups": false,
  "includeFullIntervals": false,
  "extractEvents": true
}
```

**Response Format:**
```json
{
  "jobId": "unique-job-id",
  "status": "Completed",
  "processing_time_seconds": 45.2,
  "output_type": "events",
  "result": {
    "event_count": 12,
    "events": [...]
  },
  "completedTimeStamp": "2025-06-04T14:31:00.000Z"
}
```

## Async Inference Technical Details

### Endpoint Configuration
- **Auto-scaling**: Automatically scales instance count based on queue backlog
- **Scale-out cooldown**: Prevents rapid scaling up, allows time for container initialization
- **Scale-in cooldown**: Prevents premature scale-down during long-running jobs
- **Max payload**: Maximum input file size supported by async inference
- **Max processing time**: Maximum duration allowed for a single inference job

### Request Flow
1. Upload input file to S3
2. Send inference request with S3 URI
3. SageMaker queues request
4. Instance processes data and saves output to S3
5. Notification sent to SNS topic on completion/failure (optional)
6. Retrieve results from output S3 location

### Output Types
- **Events**: Extracted fetal movement events with timestamps
- **Metadata**: Statistical summaries (kick counts, intervals, durations)

### Scaling Behavior
- **Cold start**: Initial delay when scaling from zero instances due to container initialization
- **Warm scaling**: Time required to provision additional instances when demand increases
- **Scale-down**: Conservative cooldown period prevents job interruption during long-running inference tasks

## File Structure

```
aws_sagemaker/
├── pipeline.py          # SageMaker ML pipeline definition
├── run_pipeline.py      # Pipeline execution script
├── run_pipeline.sh      # Shell wrapper with environment setup
├── Dockerfile           # Container image for training/inference
└── code/
    ├── inference.py     # FastAPI inference service
    ├── train           # Training script
    ├── serve           # Production server startup
    └── utils.py        # Helper functions
```

## Environment Variables

- `SAGEMAKER_PIPELINE_ROLE_ARN`: IAM role for pipeline execution
- `SAGEMAKER_ARTIFACT_BUCKET`: S3 bucket for artifacts
- `SAGEMAKER_PROJECT_NAME`: Project identifier
- `IMAGE_URI`: Docker image URI for processing
- `MODEL_SERVER_TIMEOUT`: Inference timeout (default: 3600s)

## Requirements

- Python 3.10+
- AWS CLI configured
- SageMaker execution role with S3, SNS permissions
- Docker for container builds