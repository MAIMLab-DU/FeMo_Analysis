"""Pipeline script for FeMo training pipeline.

                                                        . -RegisterModel
                                                        .
    Feature_Extract -> Preprocess -> Train -> Evaluate -> Condition .
                                                        .
                                                        . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import yaml
import boto3
import sagemaker
import sagemaker.local
import sagemaker.session
from sagemaker.inputs import (
    TrainingInput
)
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
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep
)
from sagemaker.estimator import Estimator
from utils import yaml2json

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PROC_DIR = "/opt/ml/processing/"


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
        sagemaker_session = sagemaker.local.LocalSession(
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
        name="ProcessingInstanceType", default_value="ml.m5.4xlarge"
    )
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.4xlarge"
    )
    # model_approval_status = ParameterString(
    #     name="ModelApprovalStatus", default_value="Approved"
    # )


    # ===== Feature Extraction Step =====
    script_extract = ScriptProcessor(
        image_uri=os.getenv("PROCESSING_IMAGE_URI"),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/feature-extract",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    manifest_path = os.path.join(PROC_DIR, "input", "dataManifest/dataManifest.json")
    feat_args = ["--data-manifest", manifest_path,
                 "--work-dir", os.path.join(PROC_DIR, "output"),
                 "--config-path", os.path.join(PROC_DIR, "input", "config/dataset-cfg.yaml")]
    if os.getenv("FORCE_EXTRACT_FEATURES", False):
        feat_args.append("--extract")

    step_extract = ProcessingStep(
        name="ExtractFeatures",
        processor=script_extract,
        inputs=[
            ProcessingInput(input_name="config",
                            source=os.path.join(BASE_DIR, "..", "configs/dataset-cfg.yaml"),
                            destination=os.path.join(PROC_DIR, "input", "config")),
            ProcessingInput(input_name="dataManifest",
                            source=os.path.join(BASE_DIR, "..", "configs/dataManifest.json"),
                            destination=os.path.join(PROC_DIR, "input", "dataManifest")),                
        ],
        outputs=[
            ProcessingOutput(output_name="features", source=os.path.join(PROC_DIR, "output"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ExtractFeatures", "output", "features"]) if local_mode else None)
        ],
        code=os.path.join(BASE_DIR, "..", "scripts", "extract.py"),
        job_arguments=feat_args,
    )


    # ===== Data Processing Step =====
    script_process = ScriptProcessor(
        image_uri=os.getenv("PROCESSING_IMAGE_URI"),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/data-process",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    features_dir = os.path.join(PROC_DIR, "input", "features")
    preproc_args = ["--features-dir", features_dir,
                    "--work-dir", os.path.join(PROC_DIR, "output"),
                    "--config-path", os.path.join(PROC_DIR, "input", "config/preprocess-cfg.yaml"),
                    "--train-config-path", os.path.join(PROC_DIR, "input", "train_config_in/train-cfg.json")]
    train_cfg_path = os.path.join(BASE_DIR, "..", "configs/train-cfg.yaml")

    step_process = ProcessingStep(
        name="ProcessData",
        processor=script_process,
        inputs=[
            ProcessingInput(input_name="features",
                            source=step_extract.properties.ProcessingOutputConfig.Outputs[
                                "features"
                            ].S3Output.S3Uri,
                            destination=features_dir),
            ProcessingInput(input_name="config",
                            source=os.path.join(BASE_DIR, "..", "configs/preprocess-cfg.yaml"),
                            destination=os.path.join(PROC_DIR, "input", "config")),
            ProcessingInput(input_name="train_config_in",
                            source=yaml2json(train_cfg_path),
                            destination=os.path.join(PROC_DIR, "input", "train_config_in"))
        ],
        outputs=[
            ProcessingOutput(output_name="dataset", source=os.path.join(PROC_DIR, "output", "dataset"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ProcessData", "output", "dataset"]) if local_mode else None),
            ProcessingOutput(output_name="processor", source=os.path.join(PROC_DIR, "output", "processor"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ProcessData", "output", "processor"]) if local_mode else None),
            ProcessingOutput(output_name="train_config", source=os.path.join(PROC_DIR, "output", "train_config"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ProcessData", "output", "train_config"]) if local_mode else None),
        ],
        code=os.path.join(BASE_DIR, "..", "scripts", "process.py"),
        job_arguments=preproc_args,
    )


    # ===== Model Training Step =====
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/train"
    estimator = Estimator(
        image_uri=os.getenv("TRAINING_IMAGE_URI"),
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/train",
        sagemaker_session=sagemaker_session,
        role=role,
        metric_definitions=[
            {"Name": "train:avg-acc", "Regex": "- Average training accuracy: ([0-9\\.]+)"},
            {"Name": "test:avg-acc", "Regex": "- Average testing accuracy: ([0-9\\.]+)"},
            {"Name": "test:best-acc", "Regex": "- Best Test Accuracy: ([0-9\\.]+)"}
        ],
        environment={
            "SM_CHANNEL_TRAIN": "/opt/ml/input/data/train",  # Path within container for dataset
            "SM_MODEL_DIR": "/opt/ml/model",  # Model output directory within container
            "SM_OUTPUT_DATA_DIR": "/opt/ml/output/data",
            "SM_TRAIN_CFG": "/opt/ml/input/data/config/train-cfg.json",
            "SM_CKPT_NAME": os.getenv('SM_CKPT_NAME', "null"),  # Name of model checkpoint file within the container
            "SM_TUNE": "true"
        }
    )
    with open(train_cfg_path, 'r') as f:
        estimator.set_hyperparameters(**yaml.safe_load(f)["config"]["hyperparams"])
    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "dataset"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "config": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train_config"
                ].S3Output.S3Uri,
                content_type="application/json"
            )
        }
    )

    # # TODO: Complete implementation
    # # ===== Model Evaluation Step =====
    # script_eval = ScriptProcessor(
    #     image_uri=os.getenv("PROCESSING_IMAGE_URI"),
    #     command=["python3"],
    #     instance_type=processing_instance_type,
    #     instance_count=processing_instance_count,
    #     base_job_name=f"{base_job_prefix}/model-eval",
    #     sagemaker_session=sagemaker_session,
    #     role=role,
    # )
    # evaluation_report = PropertyFile(
    #     name="EvaluationReport",
    #     output_name="evaluation",
    #     path="evaluation.json",
    # )
    # step_eval = ProcessingStep(
    #     name="EvaluateModel",
    #     processor=script_eval,
    #     inputs=[
    #         ProcessingInput(
    #             source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    #             destination="/opt/ml/processing/model",
    #         ),
    #         ProcessingInput(
    #             source=step_process.properties.ProcessingOutputConfig.Outputs[
    #                 "test"
    #             ].S3Output.S3Uri,
    #             destination="/opt/ml/processing/test",
    #         ),
    #     ],
    #     outputs=[
    #         ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
    #     ],
    #     code=os.path.join(BASE_DIR, "..", "scripts", "evaluate.py"),
    #     property_files=[evaluation_report],
    # )



    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count
        ],
        steps=[step_extract, step_process, step_train],
        sagemaker_session=sagemaker_session,
    )
    return pipeline