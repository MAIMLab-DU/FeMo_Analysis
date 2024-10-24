#!/bin/bash
set -e

start_time="$(date +%s)"
SCRIPT_DIR="femo_analysis"

# Process the positional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        *)
            # Store the positional arguments (DATA_FILENAME, output_file, run_dir, ckpt_name)
            if [ -z "$DATA_FILENAME" ]; then
                DATA_FILENAME="$1"
            elif [ -z "$CKPT_NAME" ]; then
                CKPT_NAME="$1"
            elif [ -z "$PARAMS_DICT" ]; then
                PARAMS_DICT="$1"
            elif [ -z "$PERF_FILE" ]; then
                PERF_FILE="$1"
            elif [ -z "$RUN_DIR" ]; then
                RUN_DIR="$1"
            else
                echo "Unknown option or too many arguments: $1"
                exit 1
            fi
            shift # Remove the argument from the list
            ;;
    esac
done

# Check if the required positional arguments are provided
if [ -z "$DATA_FILENAME" ] || [ -z "$CKPT_NAME" ]; then
    echo "Usage: $0 <data_filename> <ckpt_name> <params_dict> [output_file] [run_dir]"
    exit 1
fi

# Set defaults for optional arguments if not provided
PERF_FILE=${PERF_FILE:-"meta_info.xlsx"}
WORK_DIR=${WORK_DIR:-"./work_dir"}

# Create the $WORK_DIR directory if it doesn't exist
mkdir -p "$WORK_DIR"

# Create the next run directory
if [ -z "$RUN_DIR" ]; then
    RUN_DIR="$WORK_DIR/inference"
fi
mkdir -p "$RUN_DIR"

# Output the created run directory and the DATA_FILENAME and CKPT_NAME
echo "Created run directory: $RUN_DIR"
echo "Data filename: $DATA_FILENAME"
echo "Checkpoint name: $CKPT_NAME"
echo "Parameters dict: $PARAMS_DICT"
echo "Performance file: $PERF_FILE"

VIRTUAL_ENV=.venv
# Set up virtual env
virtualenv -p python3.10 $VIRTUAL_ENV
. $VIRTUAL_ENV/bin/activate

# Install requirements
pip install -r ./requirements.txt -q

# Run inference.py
python "./$SCRIPT_DIR/inference.py" "$DATA_FILENAME" "$CKPT_NAME" "$PARAMS_DICT" --work-dir "$RUN_DIR" --outfile "$PERF_FILE"

# Record the end time
end_time="$(date +%s)"
# Calculate the running time
running_time="$((end_time - start_time)) seconds"
echo "Total time taken: $running_time"
