#!/bin/bash

set -e

VIRTUAL_ENV=.venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

local_mode=false
manifest_file=""

# Function to display help
function show_help {
  echo "Usage: $0 [-l|--local-mode] -m <manifest_file> [-h|--help]"
  echo
  echo "Options:"
  echo "  -l, --local-mode           Upload the configuration file to S3 before downloading it."
  echo "  -m <manifest_file>, --manifest-file"
  echo "                             Path to the dataManifest file."
  echo "  -h, --help             Show this help message and exit."
  echo
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--local-mode)
      local_mode=true  # Set the upload flag to true
      shift
      ;;
    -m|--manifest-file)
      manifest_file="$2"  # Set CONFIG_S3_PATH to the next argument
      shift
      shift
      ;;
    -h|--help)
      show_help  # Display help and exit
      exit
      ;;
    *)
      echo "Invalid option: $1" >&2
      echo "Use -h or --help for usage information."
      exit 1
      ;;
  esac
done

echo "Local mode: $local_mode"
# Create/Update the SageMaker Pipeline and wait for the execution to be completed
if [[ -z "${manifest_file}" ]]; then
  echo "Data manifest file path is required."
  exit 1
fi

items_count=$(jq '.items | length' "$manifest_file")
if [[ "$items_count" -le 0 ]]; then
    echo "Items array is empty, skipping pipeline run"
    output_file="$SCRIPT_DIR/pipelineExecution.json"
    echo '{"arn": "NotTrained"}' > "$output_file"
    exit 0
fi

# Set up virtual env
virtualenv -p python3 $VIRTUAL_ENV
. $VIRTUAL_ENV/bin/activate 

#Install requirements
pip install $SCRIPT_DIR/../.[sagemaker] -q

echo "Starting Pipeline Execution"
export PYTHONUNBUFFERED=TRUE

python $SCRIPT_DIR/run_pipeline.py --module-name pipeline --manifest-file "$manifest_file" --work-dir "$SCRIPT_DIR" \
        --role-arn "$SAGEMAKER_PIPELINE_ROLE_ARN" --work-dir "$SCRIPT_DIR" \
        --tags "[{\"Key\":\"sagemaker:project-name\", \"Value\":\"${SAGEMAKER_PROJECT_NAME}\"}]" \
        --kwargs "{\"region\":\"${AWS_DEFAULT_REGION}\",\"role\":\"${SAGEMAKER_PIPELINE_ROLE_ARN}\",\"default_bucket\":\"${SAGEMAKER_ARTIFACT_BUCKET}\",\"pipeline_name\":\"${SAGEMAKER_PROJECT_NAME}\",\"model_package_group_name\":\"${SAGEMAKER_PROJECT_NAME}\",\"base_job_prefix\":\"${SAGEMAKER_PROJECT_NAME}\",\"local_mode\":\"${local_mode}\"}"

echo "Create/Update of the SageMaker Pipeline and execution Completed."

# Deactivate virtual envs
deactivate
