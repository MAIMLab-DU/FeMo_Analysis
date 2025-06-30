#!/bin/bash
set -e

# Record the start time
start_time="$(date +%s)"
VIRTUAL_ENV=.venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remove_hiccups=false
plot=false
all_sensors=false
plot_preprocessed=false
stage_params=()
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
  echo "Usage: $0 [-z|--remove-hiccups] [-p|--plot] -d <data_filename> -m <repacked_model> [-w|--work-dir] [-r|--run-name] [-f|--perf-filename] [-h|--help]"
  echo
  echo "Options:"
  echo "  -d <data_filename>, --data-filename"
  echo "                             Path to log data file(s) (.dat or .txt)."
  echo "  -z, --remove-hiccups       Remove hiccups for analysis."
  echo "  -p, --plot                 Generate plots."
  echo "  -a, --all-sensors          Plot all sensors in the output plots."
  echo "  -s, --plot-preprocessed    Plot preprocessed data."
  echo "                             (Default: plot loaded data)."
  echo "  -m <repacked_model>, --repacked-model"
  echo "                             Path to repacked model file (.tar.gz)."
  echo "  -o <perf_filename>, --perf-filename"
  echo "                             Performance csv/xlsx filename."
  echo "  -w <work_dir>, --work-dir"
  echo "                             Project working directory."
  echo "  -r <run_name>, --run-name"
  echo "                             Name of a specific run."
  echo "  --set-stage-param key=value   Override pipeline stage parameter (e.g., segment.fm_dilation=5). Can be used multiple times."
  echo "  -h, --help             Show this help message and exit."
  echo
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -z|--remove-hiccups)
      remove_hiccups=true
      shift
      ;;
    -p|--plot)
      plot=true
      shift
      ;;
    -a|--all-sensors)
      all_sensors=true
      shift
      ;;
    -s|--plot-preprocessed)
      plot_preprocessed=true
      shift
      ;;
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
    -o|--perf-filename)
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
    --set-stage-param)
      stage_params+=("$2")
      shift
      shift
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
if [[ "$(uname -s)" == "MINGW"* || "$(uname -s)" == "MSYS"* ]]; then
    # Use Windows-compatible tar command with proper path conversion
    tar -xzf "$(cygpath -u "$repacked_model")" -C "$(cygpath -u "$TEMP_DIR")"
else
    tar -xzf "$repacked_model" -C "$TEMP_DIR"
fi
echo "Repacked model files extracted to: $TEMP_DIR"

# Run inference script
python "$(convert_path "$SCRIPT_DIR/inference.py")" \
  --data-file "$(convert_path "$data_filename")" \
  --classifier "$(convert_path "$TEMP_DIR/classifier.joblib")" \
  --model "$(convert_path "$(find "$TEMP_DIR" -name 'model.*' -print -quit)")" \
  --pipeline "$(convert_path "$TEMP_DIR/pipeline.joblib")" \
  --processor "$(convert_path "$TEMP_DIR/processor.joblib")" \
  --metrics "$(convert_path "$TEMP_DIR/metrics.joblib")" \
  --work-dir "$(convert_path "$run_dir")" \
  --outfile "$(convert_path "$perf_filename")" \
  --remove-hiccups "$remove_hiccups" \
  --plot "$plot" \
  --plot-all-sensors "$all_sensors" \
  --plot-preprocessed "$plot_preprocessed" \
  $(for param in "${stage_params[@]}"; do echo --set-stage-param "$param"; done)

# Record the end time
end_time="$(date +%s)"
running_time="$((end_time - start_time)) seconds"
echo "Total time taken: $running_time"
