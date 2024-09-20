#!/bin/bash

# Check if the first argument is provided and store it in DATA_MANIFEST
if [ -z "$1" ]; then
    echo "Usage: $0 <data_manifest>"
    exit 1
else
    DATA_MANIFEST="$1"
fi

# Use the $WORK_DIR environment variable, or default to './work_dir'
WORK_DIR=${WORK_DIR:-"./work_dir"}

# Create the $WORK_DIR directory if it doesn't exist
mkdir -p "$WORK_DIR"

# Find the highest numbered run directory (run1, run2, etc.)
# The command uses ls to list directories, grep to match the pattern, and sort to find the max number.
LAST_RUN=$(ls -d "$WORK_DIR"/run* 2>/dev/null | grep -o 'run[0-9]\+' | sort -V | tail -n 1 | grep -o '[0-9]\+')

# If no previous runs, start with 1, otherwise increment the last run number
NEXT_RUN=$((LAST_RUN + 1))

# Create the next run directory
RUN_DIR="$WORK_DIR/run$NEXT_RUN"
mkdir -p "$RUN_DIR"

# Output the created run directory and the DATA_MANIFEST
echo "Created run directory: $RUN_DIR"
echo "Data manifest: $DATA_MANIFEST"

# TODO: process.py -> train.py -> evaluate.py pipeline