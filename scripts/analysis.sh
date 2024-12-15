#!/bin/bash
set -e

# Record the start time
start_time="$(date +%s)"
VIRTUAL_ENV=.venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset_config="$SCRIPT_DIR/../configs/dataset-cfg.yaml"
preproc_config="$SCRIPT_DIR/../configs/preprocess-cfg.yaml"
train_config="$SCRIPT_DIR/../configs/train-cfg.yaml"
manifest_file=""
perf_filename="performance.csv"
work_dir="$SCRIPT_DIR/../work_dir"
run_name=""
force_extract=false


# Function to display help
function show_help {
  echo "Usage: $0 [-f|--force-extract] -m <manifest_file> [-d|--dataset-cfg] [-c|preprocess-cfg] [-t|--train-cfg] [-w|--work-dir] [-r|--run-name] [-p|--perf-filename] [-h|--help]"
  echo
  echo "Options:"
  echo "  -m <manifest_file>, --manifest-file"
  echo "                             Path to the dataManifest file."
  echo "  -f, --force-extract        Force extract features."
  echo "  -d <dataset_config>, --dataset-cfg" 
  echo "                             Path to dataset config (default 'dataset-cfg.yaml')."
  echo "  -c <preproc_config>, --preprocess-cfg" 
  echo "                             Path to preprocess config (default 'preprocess-cfg.yaml')."
  echo "  -t <train_config>, --train-cfg" 
  echo "                             Path to training config (default 'train-cfg.yaml')."
  echo "  -w <work_dir>, --work-dir" 
  echo "                             Project working directory."
  echo "  -r <run_name>, --run-name"
  echo "                             Name of a specific run."
  echo "  -p <perf_filename>, --perf-filename"
  echo "                             Performance csv filename."
  echo "  -h, --help             Show this help message and exit."
  echo
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--force-extract)
      force_extract=true
      shift
      ;;
    -m|--manifest-file)
      manifest_file="$2"
      shift
      shift
      ;;
    -d|--dataset-cfg)
      dataset_config="$2"
      shift
      shift
      ;;
    -c|--preprocess-cfg)
      preproc_config="$2"
      shift
      shift
      ;;
    -t|--train-cfg)
      train_config="$2"
      shift
      shift
      ;;
    -w|--work-dir)
      work_dir="$2"
      shift
      shift
      ;;
    -r|--run-name)
      run_name="$2"
      shift
      shift
      ;;
    -p|--perf-filename)
      perf_filename="$2"
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

# Check required arguments
if [[ -z "${manifest_file}" ]]; then
    show_help
    exit 1
fi

# Create the work directory if it doesn't exist
work_dir="$(realpath "$work_dir")"
mkdir -p "$work_dir"

# Determine the next run directory
if [[ -z "${run_name}" ]]; then    
    last_run=$(ls -d "$work_dir"/run* 2>/dev/null | grep -o 'run[0-9]\+' | sort -V | tail -n 1 | grep -o '[0-9]\+') || last_run=0
    next_run=$((last_run + 1))
    run_name="run$next_run"
fi

# Set and create run directory
run_dir="$work_dir/$run_name"
mkdir -p "$run_dir"

# Output the initial configuration
echo "Belt type: $belt_type"
echo "Run artifact directory: $run_dir"
echo "Data manifest: $manifest_file"
echo "Performance file: $perf_filename"

# Set up the virtual environment
virtualenv -p python3.10 $VIRTUAL_ENV
source $VIRTUAL_ENV/bin/activate

# Install dependencies
pip install -e .

# Execute the scripts
if [ "$force_extract" = true ]; then
    python "$SCRIPT_DIR/extract.py" --data-manifest "$manifest_file" --config-path "$dataset_config" --work-dir "$run_dir" --extract
else
    python "$SCRIPT_DIR/extract.py" --data-manifest "$manifest_file" --config-path "$dataset_config" --work-dir "$run_dir"
fi
python "$SCRIPT_DIR/process.py" --features-dir "$run_dir/features/" --work-dir "$run_dir" --config-path "$preproc_config"
python "$SCRIPT_DIR/train.py" --train "$run_dir/dataset/" --config-path "$train_config" --model-dir "$run_dir/model" --output-data-dir "$run_dir/output" --tune
python "$SCRIPT_DIR/evaluate.py" --data-manifest "$manifest_file" --config-path "$dataset_config" --results-path "$run_dir/output/results/results.csv" --metadata-path "$run_dir/output/metadata/metadata.joblib" --work-dir "$run_dir" --out-filename "$perf_filename"

# Calculate and display the total running time
end_time="$(date +%s)"
running_time=$((end_time - start_time))
echo "Total time taken: ${running_time} seconds"
