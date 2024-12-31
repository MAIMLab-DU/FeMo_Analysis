#!/bin/bash
set -e

# Record the start time
start_time="$(date +%s)"
VIRTUAL_ENV=.venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_filename=""
repacked_model=""
perf_filename="meta_info.xlsx"
work_dir="$SCRIPT_DIR/../work_dir/inference"

# Function to convert paths for Git Bash
function convert_path {
  if [[ "$(uname -s)" == "MINGW"* || "$(uname -s)" == "MSYS"* ]]; then
    echo "$(cygpath -w "$1")"
  else
    echo "$1"
  fi
}

# Function to display help
function show_help {
  echo "Usage: $0 -d <data_filename> -m <repacked_model> [-w|--work-dir] [-r|--run-name] [-p|--perf-filename] [-h|--help]"
  echo
  echo "Options:"
  echo "  -d <data_filename>, --data-filename"
  echo "                             Path to log data file(s) (.dat or .txt)."
  echo "  -m <repacked_model>, --repacked-model"
  echo "                             Path to repacked model file (.tar.gz)."
  echo "  -p <perf_filename>, --perf-filename"
  echo "                             Performance csv filename."
  echo "  -w <work_dir>, --work-dir" 
  echo "                             Project working directory."
  echo "  -r <run_name>, --run-name"
  echo "                             Name of a specific run."
  echo "  -h, --help             Show this help message and exit."
  echo
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--data-filename)
      data_filename="$2"
      shift
      shift
      ;;
    -m|--repacked-model)
      repacked_model="$2"
      shift
      shift
      ;;
    -p|--perf-filename)
      perf_filename="$2"
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
    -h|--help)
      show_help
      exit
      ;;
    *)
      echo "Invalid option: $1" >&2
      echo "Use -h or --help for usage information."
      exit 1
      ;;
  esac
done

# Check if the required arguments are provided
if [[ -z "$repacked_model" ]]; then
    show_help
    exit 1
fi

# Create the work directory and determine the run directory
work_dir="$(convert_path "$work_dir")"
mkdir -p "$work_dir"

if [[ -z "${run_name}" ]]; then    
    run_dir="$work_dir"
else
    run_dir="$work_dir/$run_name"
fi
mkdir -p "$run_dir"

# Output the run directory
echo "Saving inference results to: $run_dir"

# Create a temporary directory inside run_dir
TEMP_DIR=$(mktemp -d "$run_dir/temp.XXXXXX")
# Ensure the temporary directory is removed upon script exit
trap "rm -rf '$TEMP_DIR';" EXIT

# Set up virtual environment
if [[ "$(uname -s)" == "MINGW"* || "$(uname -s)" == "MSYS"* ]]; then
  # Use Windows-compatible virtual environment creation
  python -m venv "$(convert_path $VIRTUAL_ENV)"
  source "$(convert_path $VIRTUAL_ENV)/Scripts/activate"
else
  virtualenv -p python3.10 $VIRTUAL_ENV
  source $VIRTUAL_ENV/bin/activate
fi

# Install requirements
pip install "$(convert_path "$SCRIPT_DIR/../.")" -q

# Extract repacked model files to the temp directory
tar -xzf "$repacked_model" -C "$TEMP_DIR"
echo "Repacked model files extracted to: $TEMP_DIR"

# Run inference script
python "$(convert_path "$SCRIPT_DIR/inference.py")" \
  --data-file "$(convert_path "$data_filename")" \
  --model "$TEMP_DIR/model.joblib" \
  --pipeline "$TEMP_DIR/pipeline.joblib" \
  --processor "$TEMP_DIR/processor.joblib" \
  --metrics "$TEMP_DIR/metrics.joblib" \
  --work-dir "$(convert_path "$run_dir")" \
  --outfile "$(convert_path "$perf_filename")"

# Record the end time
end_time="$(date +%s)"
running_time="$((end_time - start_time)) seconds"
echo "Total time taken: $running_time"
