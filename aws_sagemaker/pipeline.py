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

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = "/opt/ml/processing/"


def get_session(region, default_bucket, local_mode=False):
    """Gets the SageMaker session based on the region and mode.

    Args:
        region: The AWS region to start the session.
        default_bucket: The bucket to use for storing artifacts.
        local_mode: Boolean flag to indicate whether to use local mode.

    Returns:
        A `sagemaker.session.Session` instance or `sagemaker.local.LocalSession` if local_mode=True.
    """

    boto_session = boto3.Session(region_name=region)

    if local_mode:
        # Create a LocalSession for running SageMaker jobs locally
        sagemaker_session = LocalPipelineSession(
            boto_session=boto_session,
            default_bucket=default_bucket
        )
        # Set the default bucket in local mode (if needed)
        # sagemaker_session.config = {'local': {'local_code': True}}
    else:
        # Create a regular SageMaker session
        sagemaker_client = boto_session.client("sagemaker")
        runtime_client = boto_session.client("sagemaker-runtime")
        sagemaker_session = sagemaker.session.Session(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            sagemaker_runtime_client=runtime_client,
            default_bucket=default_bucket,
        )

    return sagemaker_session


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="FeMoModelPackageGroup",
    pipeline_name="FeMoPipeline",
    base_job_prefix="FeMo",
    local_mode=False
):
    """Gets a SageMaker ML Pipeline instance working with on femo data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """

    sagemaker_session = get_session(region, default_bucket, local_mode)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )


    # ===== Feature Extraction Step =====
    feat_processor = ScriptProcessor(
        image_uri=os.getenv("PROCESSING_IMAGE_URI"),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/feature-extract",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    manifest_path = os.path.join(OUT_DIR, "input", "dataManifest/dataManifest.json")
    feat_args = ["--data-manifest", manifest_path,
                 "--work-dir", os.path.join(OUT_DIR, "features"),
                 "--config-path", os.path.join(OUT_DIR, "input", "config/dataset-cfg.yaml")]
    if os.getenv("FORCE_EXTRACT_FEATURES", False):
        feat_args.append("--extract")

    step_extract = ProcessingStep(
        name="ExtractFeatures",
        processor=feat_processor,
        inputs=[
            ProcessingInput(input_name="config",
                            source=os.path.join(BASE_DIR, "..", "configs/dataset-cfg.yaml"),
                            destination=os.path.join(OUT_DIR, "input", "config")),
            ProcessingInput(input_name="dataManifest",
                            source=os.path.join(BASE_DIR, "..", "configs/dataManifest.json"),
                            destination=os.path.join(OUT_DIR, "input", "dataManifest")),                
        ],
        outputs=[
            ProcessingOutput(output_name="features", source=os.path.join(OUT_DIR, "features"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ExtractFeatures", "output"]) if local_mode else None),
        ],
        code=os.path.join(BASE_DIR, "..", "scripts", "extract.py"),
        job_arguments=feat_args,
    )


    # ===== Data Processing Step =====
    data_processor = ScriptProcessor(
        image_uri=os.getenv("PROCESSING_IMAGE_URI"),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/data-process",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    features_dir = os.path.join(OUT_DIR, "input", "features")
    preproc_args = ["--features-dir", features_dir,
                    "--work-dir", os.path.join(OUT_DIR, "dataset"),
                    "--config-path", os.path.join(OUT_DIR, "input", "config/preprocess-cfg.yaml")]

    step_preprocess = ProcessingStep(
        name="ProcessData",
        processor=data_processor,
        inputs=[
            ProcessingInput(input_name="features",
                            source=step_extract.properties.ProcessingOutputConfig.Outputs[
                                "features"
                            ].S3Output.S3Uri,
                            destination=features_dir),
            ProcessingInput(input_name="config",
                            source=os.path.join(BASE_DIR, "..", "configs/preprocess-cfg.yaml"),
                            destination=os.path.join(OUT_DIR, "input", "config")),                
        ],
        outputs=[
            ProcessingOutput(output_name="dataset", source=os.path.join(OUT_DIR, "dataset"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ProcessData", "output"]) if local_mode else None),
            ProcessingOutput(output_name="processor", source=os.path.join(OUT_DIR, "dataset/processor"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ProcessData", "output"]) if local_mode else None)
        ],
        code=os.path.join(BASE_DIR, "..", "scripts", "process.py"),
        job_arguments=preproc_args,
    )


    # ===== Model Training Step =====
    """
    /opt/ml/model – Your algorithm should write all final model artifacts to this directory.
    SageMaker copies this data as a single object in compressed tar format to the S3 location that you specified in the CreateTrainingJob request.
    If multiple containers in a single training job write to this directory they should ensure no file/directory names clash.
    SageMaker aggregates the result in a TAR file and uploads to S3 at the end of the training job.

    /opt/ml/output/data – Your algorithm should write artifacts you want to store other than the final model to this directory.
    SageMaker copies this data as a single object in compressed tar format to the S3 location that you specified in the CreateTrainingJob request.
    If multiple containers in a single training job write to this directory they should ensure no file/directory names clash.
    SageMaker aggregates the result in a TAR file and uploads to S3 at the end of the training job.
    """

    

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count
        ],
        steps=[step_extract, step_preprocess],
        sagemaker_session=sagemaker_session,
    )
    return pipeline