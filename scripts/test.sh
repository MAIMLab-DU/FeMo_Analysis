#!/bin/bash

set -e

VIRTUAL_ENV=.venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="tests"
DATA_FILES_DIR="$TEST_DIR/datafiles"
TEST_REPORT_PATH="reports/reports.xml"

S3_BUCKET="femo-analysis-ci"
S3_KEY="tests/dataFiles.zip"

# Function to convert paths for Git Bash
function convert_path {
  if [[ "$(uname -s)" == "MINGW"* || "$(uname -s)" == "MSYS"* ]]; then
    echo "$(cygpath -w "$1")"
  else
    echo "$1"
  fi
}

# Set up virtual environment
if [[ "$(uname -s)" == "MINGW"* || "$(uname -s)" == "MSYS"* ]]; then
  python -m venv "$(convert_path $VIRTUAL_ENV)"
  source "$(convert_path $VIRTUAL_ENV)/Scripts/activate"
else
  virtualenv -p python3.10 $VIRTUAL_ENV
  source $VIRTUAL_ENV/bin/activate
fi

# Install requirements
pip install "$(convert_path "$SCRIPT_DIR/../.[dev]")" -q

# Run Ruff for linting
echo "Running ruff to check for linting issues"
ruff check ./**/*.py

# Install AWS CLI if needed
if ! command -v aws &> /dev/null
then
    echo "AWS CLI not found, installing..."
    pip install awscli
fi

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set default.region "$AWS_DEFAULT_REGION"

# Check if dataFiles directory exists, if not, download and unzip the file
if [ ! -d "$(convert_path $DATA_FILES_DIR)" ]; then
    echo "Directory $DATA_FILES_DIR does not exist. Downloading dataFiles.zip from S3..."
    aws s3 cp "s3://$S3_BUCKET/$S3_KEY" "dataFiles.zip"

    echo "Unzipping dataFiles.zip into $TEST_DIR/ directory..."
    unzip -o "dataFiles.zip" -d "$(convert_path $TEST_DIR/)"
    rm "dataFiles.zip"
else
    echo "Directory $DATA_FILES_DIR exists. Skipping download."
fi

# Run tests
echo "Running tests"
pytest --tb=short --junitxml "$(convert_path $TEST_REPORT_PATH)" "$(convert_path $TEST_DIR)"

# Deactivate virtual environment
deactivate
