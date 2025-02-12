import os
import sys
import uuid
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from femo.logger import LOGGER
from femo.inference import PredictionService


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, help="Path to data file(s) (.dat or .txt)")
    parser.add_argument("--classifier", type=str, required=True, help="Path to fitted classifier object file (.joblib)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.joblib or .h5)")
    parser.add_argument("--pipeline", type=str, required=True, help="Path to data pipeline object (.joblib)")
    parser.add_argument("--processor", type=str, required=True, help="Path to data processor object (.joblib)")
    parser.add_argument("--metrics", type=str, required=True, help="Path to evaluation metrics object (.joblib)")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--outfile", type=str, default="meta_info.xlsx", help="Metrics output file")
    parser.add_argument("--remove-hiccups", action='store_true', default=False, help="Exclude hiccups from ML detections map")
    parser.add_argument("--plot", action='store_true', default=False, help="Generate and save detection plots")
    args = parser.parse_args()

    return args


def main(args):
    LOGGER.info("Starting inference...")

    os.makedirs(args.work_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.data_file.endswith('.dat'):
        # If it's a .dat file, store it in a list
        filenames = [args.data_file]
    elif args.data_file.endswith('.txt'):
        # If it's a .txt file, read each line and store filenames in a list
        with open(args.data_file, 'r') as file:
            filenames = [line.strip() for line in file if line.strip()]  # Strips newlines and empty lines
            filenames = [line for line in filenames if line.endswith('.dat')]
    else:
        raise ValueError("Unsupported file format. Please provide a .dat or .txt file.")

    try:
        pred_service = PredictionService(
            args.classifier,
            args.model,
            args.pipeline,
            args.processor,
            args.metrics
        )
        _ = all([
            pred_service.get_model() is not None,
            pred_service.get_pipeline() is not None,
            pred_service.get_processor() is not None,
            pred_service.get_metrics() is not None
        ])
    except Exception as e:
        LOGGER.error(f"Error loading fitted objects: {e}")
        sys.exit(1)
        
    for data_filename in tqdm(filenames, desc=f"Peforming inference on {len(filenames)} files..."):
        data, pipeline_output, ml_map = pred_service.predict(
            data_filename
        )
        job_id = str(uuid.uuid4())[:8]

        metainfo_dict = {
            "Timestamp": [timestamp],
            "JobId": [job_id],
            "File Name": [data_filename],
            "Number of bouts per hour": [(data.numKicks*60) / (data.totalFMDuration+data.totalNonFMDuration)],
            "Mean duration of fetal movement (seconds)": [data.totalFMDuration*60/data.numKicks if data.numKicks > 0 else 0],
            "Median onset interval (seconds)": [np.median(data.onsetInterval)],
            "Active time of fetal movement (%)": [(data.totalFMDuration/(data.totalFMDuration+data.totalNonFMDuration))*100]
        }

        # Check if the file exists
        if os.path.exists(os.path.join(args.work_dir, args.outfile)):
            # If file exists, read it and append the new data
            df_existing = pd.read_excel(os.path.join(args.work_dir, args.outfile))
            df_new = pd.DataFrame(metainfo_dict)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            # If file does not exist, create a new DataFrame with titles and new data
            df_combined = pd.DataFrame(metainfo_dict)

        # Write the combined DataFrame to the Excel file
        df_combined.to_excel(os.path.join(args.work_dir, args.outfile), index=False)

        LOGGER.info(f"Meta info saved to {os.path.realpath(os.path.join(args.work_dir, args.outfile))}")
        
        if args.plot:
            LOGGER.info("Plotting the results. It may take some time....")
            pred_service.save_pred_plots(
                pipeline_output,
                ml_map,
                os.path.join(args.work_dir, f"{os.path.basename(data_filename).split('.dat')[0]}_{job_id}.png")
            )
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
