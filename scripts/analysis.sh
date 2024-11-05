#!/bin/bash
set -e

# Record the start time
start_time="$(date +%s)"
SCRIPT_DIR="scripts"

# Process positional arguments (data_manifest, ckpt_name, run_dir, perf_file)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        *)
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
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$DATA_MANIFEST" ] || [ -z "$CKPT_NAME" ]; then
    echo "Usage: $0 <data_manifest> <ckpt_name> [run_dir] [performance_filename]"
    exit 1
fi

# Set defaults for optional arguments
PERF_FILE=${PERF_FILE:-"performance.csv"}
WORK_DIR=${WORK_DIR:-"./work_dir"}

# Create the work directory if it doesn't exist
mkdir -p "$WORK_DIR"

# Determine the next run directory
LAST_RUN=$(ls -d "$WORK_DIR"/run* 2>/dev/null | grep -o 'run[0-9]\+' | sort -V | tail -n 1 | grep -o '[0-9]\+') || LAST_RUN=0

NEXT_RUN=$((LAST_RUN + 1))

# Set and create run directory
RUN_DIR=${RUN_DIR:-"$WORK_DIR/run$NEXT_RUN"}
mkdir -p "$RUN_DIR"

# Output the initial configuration
echo "Created run directory: $RUN_DIR"
echo "Data manifest: $DATA_MANIFEST"
echo "Performance file: $PERF_FILE"
echo "Checkpoint name: $CKPT_NAME"

# Set up the virtual environment
VIRTUAL_ENV=.venv
virtualenv -p python3.10 $VIRTUAL_ENV
source $VIRTUAL_ENV/bin/activate

# Install dependencies
pip install -e .

# Execute the scripts
python "$SCRIPT_DIR/extract.py" --data-manifest "$DATA_MANIFEST" --work-dir "$RUN_DIR"
python "$SCRIPT_DIR/process.py" --features-dir "$RUN_DIR" --work-dir "$RUN_DIR"
python "$SCRIPT_DIR/train.py" --train "$RUN_DIR/dataset/" --model-dir "$RUN_DIR/model" --output-data-dir "$RUN_DIR/output" --ckpt-name "$CKPT_NAME" --tune
python "$SCRIPT_DIR/evaluate.py" --data-manifest "$DATA_MANIFEST" --results-path "$RUN_DIR/output/results/results.csv" --metadata-path "$RUN_DIR/output/metadata/metadata.joblib" --work-dir "$RUN_DIR" --out-filename "$PERF_FILE"

# Calculate and display the total running time
end_time="$(date +%s)"
running_time=$((end_time - start_time))
echo "Total time taken: ${running_time} seconds"
