"""Pipeline script for FeMo training pipeline.

                                                        . -RegisterModel
                                                        .
    Feature_Extract -> Preprocess -> Train -> Evaluate -> Condition .
                                                        .
                                                        . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from steps import (
    get_feature_extraction_step
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="AbaloneModelPackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )

    # Feature extraction step
    step_extract = get_feature_extraction_step(
        image_uri=os.getenv("PROCESSING_IMAGE_URI"),
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=sagemaker_session,
        base_job_prefix=base_job_prefix,
        role=role
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count
        ],
        steps=[step_extract],
        sagemaker_session=sagemaker_session,
    )
    return pipeline