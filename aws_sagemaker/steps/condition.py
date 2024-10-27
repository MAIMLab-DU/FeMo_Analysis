import os
from sagemaker.processing import ProcessingOutput
from sagemaker.processing import Processor
from sagemaker.workflow.steps import ProcessingStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_condition_step(image_uri,
             instance_type,
             instance_count,
             sagemaker_session,
             base_job_prefix,
             role,
             force_extract_features=False):

    # data processing step
    data_processor = Processor(
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name=f"{base_job_prefix}/process",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    out_dir = "/opt/ml/processing/"
    manifest_path = os.path.join(BASE_DIR, "..", "..", "configs/dataManifest.json")
    job_args = [manifest_path,
                "--params-filename", os.path.join(out_dir, "params_dict.pkl"),
                "--work-dir", out_dir]
    if force_extract_features:
        job_args.append("--extract")

    step_process = ProcessingStep(
        name="PreprocessData",
        processor=data_processor,
        outputs=[
            ProcessingOutput(output_name="dataset", source=out_dir),
            ProcessingOutput(output_name="params_dict", source=out_dir)
        ],
        code=os.path.join(BASE_DIR, "..", "..", "scripts", "process.py"),
        job_arguments=job_args,
    )

    return step_process
