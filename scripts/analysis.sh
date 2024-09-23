#!/bin/bash
set -e

start_time="$(date +%s)"

# Check if the first argument is provided and store it in DATA_MANIFEST
if [ -z "$1"]; then
    echo "Usage: $0 <data_manifest> $1 <output_file>"
    exit 1
else
    DATA_MANIFEST="$1"
fi
if [ -z "$2"]; then
    PERF_FILE="peformance.csv"
else
    PERF_FILE="$2"
fi

SCRIPT_DIR="femo_analysis"
WORK_DIR=${WORK_DIR:-"./work_dir"}

# Create the $WORK_DIR directory if it doesn't exist
mkdir -p "$WORK_DIR"

# Find the highest numbered run directory (run1, run2, etc.)
# The command uses ls to list directories, grep to match the pattern, and sort to find the max number.
LAST_RUN=$(ls -d "$WORK_DIR"/run* 2>/dev/null | grep -o 'run[0-9]\+' | sort -V | tail -n 1 | grep -o '[0-9]\+')

# If no previous runs, start with 1, otherwise increment the last run number
NEXT_RUN=$((LAST_RUN + 1))

# Create the next run directory
RUN_DIR="$WORK_DIR/run10"
mkdir -p "$RUN_DIR"

# Output the created run directory and the DATA_MANIFEST
echo "Created run directory: $RUN_DIR"
echo "Data manifest: $DATA_MANIFEST"
echo "Output file: $PERF_FILE"

# TODO: process.py -> train.py -> evaluate.py pipeline
VIRTUAL_ENV=.venv
# Set up virtual env
virtualenv -p python3.10 $VIRTUAL_ENV
. $VIRTUAL_ENV/bin/activate
#Install requirements
pip install -r ./requirements.txt -q
# Run process.py
python "./$SCRIPT_DIR/process.py" "$DATA_MANIFEST" --work-dir "$RUN_DIR" --data-dir "./data"
# Run train.py
python "./$SCRIPT_DIR/train.py" "$RUN_DIR" --work-dir "$RUN_DIR" --tune
# Run evaluate.py
python "./$SCRIPT_DIR/evaluate.py" "$DATA_MANIFEST" "$RUN_DIR" --work-dir "$RUN_DIR" --outfile "$PERF_FILE"

# Record the end time
end_time="$(date +%s)"
# Calculate the running time
running_time="$((end_time - start_time)) seconds"
echo "Total time taken: $running_time"
