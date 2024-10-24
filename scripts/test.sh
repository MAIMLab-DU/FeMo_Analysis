#!/bin/bash

set -e

VIRTUAL_ENV=.venv
TEST_DIR="tests"
DATA_FILES_DIR="$TEST_DIR/datafiles"
TEST_REPORT_PATH=reports/reports.xml

S3_BUCKET="femo-analysis-ci"
S3_KEY="tests/dataFiles.zip"

# Set up virtual env
virtualenv -p python3.10 $VIRTUAL_ENV
. $VIRTUAL_ENV/bin/activate

# Install requirements
pip install -r tests/requirements.txt

# Run Ruff for linting
echo "Running ruff to check for linting issues"
ruff check ./**/*.py

# Install AWS CLI if needed
if ! command -v aws &> /dev/null
then
    echo "AWS CLI not found, installing..."
    pip install awscli
fi

aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY && \
aws configure set default.region $AWS_DEFAULT_REGION

# Check if dataFiles directory exists, if not, download and unzip the file
if [ ! -d "$DATA_FILES_DIR" ]; then
    echo "Directory $DATA_FILES_DIR does not exist. Downloading dataFiles.zip from S3..."
    aws s3 cp s3://$S3_BUCKET/$S3_KEY dataFiles.zip

    echo "Unzipping dataFiles.zip into $TEST_DIR/ directory..."
    unzip -o dataFiles.zip -d $TEST_DIR/
    rm dataFiles.zip
else
    echo "Directory $DATA_FILES_DIR exists. Skipping download."
fi

# Run tests
echo "Running tests"
export PYTHONPATH=./src
pytest --tb=short --junitxml=$TEST_REPORT_PATH ./$TEST_DIR

# Deactivate virtual env
deactivate
