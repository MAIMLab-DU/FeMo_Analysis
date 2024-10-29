import os
from sagemaker.processing import ProcessingOutput, ProcessingInput
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_feature_extraction_step(image_uri,
                        instance_type,
                        instance_count,
                        sagemaker_session,
                        base_job_prefix,
                        role,
                        force_extract_features=False):

    # feature extraction step
    data_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name=f"{base_job_prefix}/feature-extract",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    out_dir = "/opt/ml/processing/"
    manifest_path = os.path.join(out_dir, "input", "dataManifest/dataManifest.json")
    job_args = ["--data-manifest", manifest_path,
                "--work-dir", os.path.join(out_dir, "dataset"),
                "--config-dir", os.path.join(out_dir, "input", "config")]
    if force_extract_features:
        job_args.append("--extract")

    step_extract = ProcessingStep(
        name="ExtractFeatures",
        processor=data_processor,
        inputs=[
            ProcessingInput(input_name="config",
                            source=os.path.join(BASE_DIR, "..", "..", "configs/dataset-cfg.yaml"),
                            destination=os.path.join(out_dir, "input", "config")),
            ProcessingInput(input_name="dataManifest",
                            source=os.path.join(BASE_DIR, "..", "..", "configs/dataManifest.json"),
                            destination=os.path.join(out_dir, "input", "dataManifest")),                
        ],
        outputs=[
            ProcessingOutput(output_name="dataset", source=os.path.join(out_dir, "dataset"))
        ],
        code=os.path.join(BASE_DIR, "..", "..", "scripts", "extract.py"),
        job_arguments=job_args,
    )

    return step_extract
