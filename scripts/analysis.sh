#!/bin/bash
set -e

start_time="$(date +%s)"
SCRIPT_DIR="femo_analysis"

# Process the positional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        *)
            # Store the positional arguments (data_manifest, output_file, run_dir, ckpt_name)
            if [ -z "$DATA_MANIFEST" ]; then
                DATA_MANIFEST="$1"
            elif [ -z "$CKPT_NAME" ]; then
                CKPT_NAME="$1"
            elif [ -z "$PERF_FILE" ]; then
                PERF_FILE="$1"
            elif [ -z "$RUN_DIR" ]; then
                RUN_DIR="$1"
            elif [ -z "$PARAMS_FILE" ]; then
                PARAMS_FILE="$1"
            else
                echo "Unknown option or too many arguments: $1"
                exit 1
            fi
            shift # Remove the argument from the list
            ;;
    esac
done

# Check if the required positional arguments are provided
if [ -z "$DATA_MANIFEST" ] || [ -z "$CKPT_NAME" ]; then
    echo "Usage: $0 <data_manifest> <ckpt_name> [output_file] [run_dir] [params_filename]"
    exit 1
fi

# Set defaults for optional arguments if not provided
PERF_FILE=${PERF_FILE:-"performance.csv"}
WORK_DIR=${WORK_DIR:-"./work_dir"}
PARAMS_FILE=${PARAMS_FILE:-null}

# Create the $WORK_DIR directory if it doesn't exist
mkdir -p "$WORK_DIR"

# Find the highest numbered run directory (run1, run2, etc.)
LAST_RUN=$(ls -d "$WORK_DIR"/run* 2>/dev/null | grep -o 'run[0-9]\+' | sort -V | tail -n 1 | grep -o '[0-9]\+')

# If no previous runs, start with 1, otherwise increment the last run number
NEXT_RUN=$((LAST_RUN + 1))

# Create the next run directory
if [ -z "$RUN_DIR" ]; then
    RUN_DIR="$WORK_DIR/run$NEXT_RUN"
fi
mkdir -p "$RUN_DIR"

# Output the created run directory and the DATA_MANIFEST and CKPT_NAME
echo "Created run directory: $RUN_DIR"
echo "Data manifest: $DATA_MANIFEST"
echo "Performance file: $PERF_FILE"
echo "Checkpoint name: $CKPT_NAME"

VIRTUAL_ENV=.venv
# Set up virtual env
virtualenv -p python3.10 $VIRTUAL_ENV
. $VIRTUAL_ENV/bin/activate

# Install requirements
pip install -r ./requirements.txt -q

# Run process.py
python "./$SCRIPT_DIR/process.py" "$DATA_MANIFEST" --work-dir "$RUN_DIR" --data-dir "./data" --params-filename "$PARAMS_FILE"

# Run train.py
python "./$SCRIPT_DIR/train.py" "$RUN_DIR" "$CKPT_NAME" --work-dir "$RUN_DIR" --tune

# Run evaluate.py
python "./$SCRIPT_DIR/evaluate.py" "$DATA_MANIFEST" "$RUN_DIR" --work-dir "$RUN_DIR" --outfile "$PERF_FILE"

# Record the end time
end_time="$(date +%s)"
# Calculate the running time
running_time="$((end_time - start_time)) seconds"
echo "Total time taken: $running_time"
