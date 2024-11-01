#!/bin/bash
set -e

start_time="$(date +%s)"
SCRIPT_DIR="scripts"

# Process the positional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        *)
            # Store the positional arguments (data_manifest, output_file, run_dir, ckpt_name)
            if [ -z "$DATA_MANIFEST" ]; then
                DATA_MANIFEST="$1"
            elif [ -z "$CKPT_NAME" ]; then
                CKPT_NAME="$1"
            elif [ -z "$RUN_DIR" ]; then
                RUN_DIR="$1"
            elif [ -z "$PERF_FILE" ]; then
                PERF_FILE="$1"
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
    echo "Usage: $0 <data_manifest> <ckpt_name> [run_dir] [performance_filename] [params_filename]"
    exit 1
fi

# Set defaults for optional arguments if not provided
PERF_FILE=${PERF_FILE:-"performance.csv"}
WORK_DIR=${WORK_DIR:-"./work_dir"}

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
pip install -e .

# Run extract.py
python "./$SCRIPT_DIR/extract.py" --data-manifest "$DATA_MANIFEST" --work-dir "$RUN_DIR"

# Run preprocess.py
python "./$SCRIPT_DIR/process.py" --dataset-path "$RUN_DIR/dataset.csv" --work-dir "$RUN_DIR"

# Run train.py
python "./$SCRIPT_DIR/train.py" --dataset-path "$RUN_DIR/preprocessed_dataset.csv" --ckpt-name "$CKPT_NAME" --work-dir "$RUN_DIR" --tune

# Run evaluate.py
python "./$SCRIPT_DIR/evaluate.py" --data-manifest "$DATA_MANIFEST" --results-path "$RUN_DIR/results.csv" --work-dir "$RUN_DIR" --outfile "$PERF_FILE"

# Record the end time
end_time="$(date +%s)"
# Calculate the running time in seconds
running_time=$((end_time - start_time))
# Print the running time with "seconds" appended
echo "Total time taken: ${running_time} seconds"
