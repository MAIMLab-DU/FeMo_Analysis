import os
from sagemaker.processing import ProcessingOutput
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_preprocess_step(image_uri,
                        instance_type,
                        instance_count,
                        sagemaker_session,
                        base_job_prefix,
                        role,
                        force_extract_features=False):

    # data processing step
    ...
