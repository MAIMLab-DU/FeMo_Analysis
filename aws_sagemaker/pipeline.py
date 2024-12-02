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
from typing import Literal
from sagemaker.inputs import (
    TrainingInput
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep
)
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet
)

from sagemaker.workflow.step_collections import RegisterModel
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
    manifest_file: str,
    role=None,
    default_bucket=None,
    belt_type: Literal["A", "B", "C"] = "A",
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

    pipeline_name = pipeline_name + "-belt" + belt_type
    model_package_group_name = model_package_group_name + '-belt' + belt_type
    assert manifest_file.endswith('.json'), "Must be a path to manifest.json file"

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


    # ===== Feature Extraction Step =====
    script_extract = ScriptProcessor(
        image_uri=os.getenv("IMAGE_URI"),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/feature-extract",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    manifest_path = os.path.join(PROC_DIR, "input", "dataManifest", os.path.basename(manifest_file))
    feat_args = ["--data-manifest", manifest_path,
                 "--work-dir", os.path.join(PROC_DIR, "output"),
                 "--config-path", os.path.join(PROC_DIR, "input", f"config/dataset-cfg_belt{belt_type}.yaml")]
    if os.getenv("FORCE_EXTRACT_FEATURES", False):
        feat_args.append("--extract")

    step_extract = ProcessingStep(
        name="ExtractFeatures",
        processor=script_extract,
        inputs=[
            ProcessingInput(input_name="config",
                            source=os.path.join(BASE_DIR, "..", f"configs/dataset-cfg_belt{belt_type}.yaml"),
                            destination=os.path.join(PROC_DIR, "input", "config")),
            ProcessingInput(input_name="dataManifest",
                            source=manifest_file,
                            destination=os.path.join(PROC_DIR, "input", "dataManifest")),                
        ],
        outputs=[
            ProcessingOutput(output_name="features", source=os.path.join(PROC_DIR, "output", "features"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ExtractFeatures", "output", "features"]) if local_mode else None),
            ProcessingOutput(output_name="pipeline", source=os.path.join(PROC_DIR, "output", "pipeline"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ExtractFeatures", "output", "pipeline"]) if local_mode else None)
        ],
        code=os.path.join(BASE_DIR, "..", "scripts", "extract.py"),
        job_arguments=feat_args,
    )


    # ===== Data Processing Step =====
    script_process = ScriptProcessor(
        image_uri=os.getenv("IMAGE_URI"),
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
        name="ProcessFeatures",
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
                                                              "local_run", "ProcessFeatures", "output", "dataset"]) if local_mode else None),
            ProcessingOutput(output_name="processor", source=os.path.join(PROC_DIR, "output", "processor"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ProcessFeatures", "output", "processor"]) if local_mode else None),
            ProcessingOutput(output_name="train_config", source=os.path.join(PROC_DIR, "output", "train_config"),
                             destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                              "local_run", "ProcessFeatures", "output", "train_config"]) if local_mode else None)
        ],
        code=os.path.join(BASE_DIR, "..", "scripts", "process.py"),
        job_arguments=preproc_args,
    )


    # ===== Model Training Step =====
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/train"
    estimator = Estimator(
        image_uri=os.getenv("IMAGE_URI"),
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/train",
        sagemaker_session=sagemaker_session,
        enable_sagemaker_metrics=True,
        role=role,
        metric_definitions=[
            {"Name": "train:avg-acc", "Regex": "- Average training accuracy: ([0-9\\.]+)"},
            {"Name": "test:avg-acc", "Regex": "- Average testing accuracy: ([0-9\\.]+)"},
            {"Name": "test:best-acc", "Regex": "- Best Test Accuracy: ([0-9\\.]+)"}
        ]
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


    # ===== Model Evaluation Step =====  
    script_eval = ScriptProcessor(
        image_uri=os.getenv("IMAGE_URI"),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/model-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    EVAL_DIR = os.path.join(PROC_DIR, "input", "trainjob/data") if local_mode else os.path.join(PROC_DIR, "input", "trainjob")
    eval_args = ["--data-manifest", manifest_path,
                 "--results-path", os.path.join(EVAL_DIR, "results/results.csv"),
                 "--metadata-path", os.path.join(EVAL_DIR, "metadata/metadata.joblib"),
                 "--config-path", os.path.join(PROC_DIR, "input", f"config/dataset-cfg_belt{belt_type}.yaml"),
                 "--work-dir", PROC_DIR,
                 "--sagemaker", os.path.join(PROC_DIR, "input", "trainjob")]

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                input_name="config",
                source=os.path.join(BASE_DIR, "..", f"configs/dataset-cfg_belt{belt_type}.yaml"),
                destination=os.path.join(PROC_DIR, "input", "config")
            ),
            ProcessingInput(
                input_name="dataManifest",
                source=manifest_file,
                destination=os.path.join(PROC_DIR, "input", "dataManifest")
            ),
            ProcessingInput(
                input_name="trainjob",
                source=Join(
                    on='/',
                    values=[
                        step_train.properties.OutputDataConfig.S3OutputPath,
                        step_train.properties.TrainingJobName,
                        'output',
                        'output.tar.gz'
                    ]
                ),
                destination=os.path.join(PROC_DIR, "input", "trainjob"),
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source=os.path.join(PROC_DIR, "performance"),
                destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                "local_run", "EvaluateModel", "output", "performance"]) if local_mode else None
            ),
            ProcessingOutput(
                output_name="metrics",
                source=os.path.join(PROC_DIR, "metrics"),
                destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                "local_run", "EvaluateModel", "output", "metrics"]) if local_mode else None
            )
        ],
        code=os.path.join(BASE_DIR, "..", "scripts", "evaluate.py"),
        job_arguments=eval_args,
        property_files=[evaluation_report],
    )


    # ===== Model Repack Step =====
    script_repack = ScriptProcessor(
        image_uri=os.getenv("IMAGE_URI"),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/model-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    repack_args = [
        "--model", os.path.join(PROC_DIR, "input", "model", "model.tar.gz"),
        "--pipeline", os.path.join(PROC_DIR, "input", "pipeline", "pipeline.tar.gz"),
        "--processor", os.path.join(PROC_DIR, "input", "processor", "processor.tar.gz"),
        "--metrics", os.path.join(PROC_DIR, "input", "metrics", "metrics.tar.gz"),
        "--work-dir", os.path.join(PROC_DIR, "output")
    ]
    step_repack = ProcessingStep(
        name="RepackModel",
        processor=script_repack,
        inputs=[            
            ProcessingInput(
                input_name="model",
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination=os.path.join(PROC_DIR, "input", "model"),
            ),
            ProcessingInput(
                input_name="pipeline",
                source=step_extract.properties.ProcessingOutputConfig.Outputs[
                                "pipeline"
                            ].S3Output.S3Uri,
                destination=os.path.join(PROC_DIR, "input", "pipeline")
            ),
            ProcessingInput(
                input_name="processor",
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                                "processor"
                            ].S3Output.S3Uri,
                destination=os.path.join(PROC_DIR, "input", "processor")
            ),
            ProcessingInput(
                input_name="metrics",
                source=step_eval.properties.ProcessingOutputConfig.Outputs[
                                "metrics"
                            ].S3Output.S3Uri,
                destination=os.path.join(PROC_DIR, "input", "metrics")
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="repacked-model",
                source=os.path.join(PROC_DIR, "output", "repack"),
                destination=Join(on='/', values=["s3:/", default_bucket, pipeline_name,
                                                "local_run", "RepackModel", "output", "repacked-model"]) if local_mode else None
            )
        ],
        code=os.path.join(BASE_DIR, "..", "scripts", "repack.py"),
        job_arguments=repack_args,
    )


    # ===== Model Repack Step (Conditionally Executed) =====
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_eval.properties.ProcessingOutputConfig.Outputs[
                                "evaluation"
                            ].S3Output.S3Uri,
                    "evaluation.json"
                ]
            ),
            content_type="application/json",
        )
    )
    inference_model = Model(
        image_uri=os.getenv("IMAGE_URI"),
        model_data=Join(
            on="/",
            values=[
                step_repack.properties.ProcessingOutputConfig.Outputs[
                                "repacked-model"
                            ].S3Output.S3Uri,
                "model.tar.gz"
            ]
        ),
        role=role,
        name=f"{pipeline_name}-InferenceModel",
        sagemaker_session=sagemaker_session,
        predictor_cls=Predictor,
        env={
            "MODEL_SERVER_TIMEOUT": "300"
        }
    )

    step_register_inference_model = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.c5.xlarge", "ml.c5.2xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
        model_metrics=model_metrics,
        model=inference_model
    )

    # condition step for evaluating model quality and branching execution
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="classification_metrics.f1-score.value",
        ),
        right=0.55,
    )
    step_cond = ConditionStep(
        name="CheckFScoreEvaluation",
        conditions=[cond_gte],
        if_steps=[step_repack, step_register_inference_model] if not local_mode else [step_repack],
        else_steps=[],
    )


    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count
        ],
        steps=[step_extract, step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline