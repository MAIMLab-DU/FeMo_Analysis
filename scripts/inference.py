import os
import gc
import sys
import uuid
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from femo.logger import LOGGER
from femo.inference import PredictionService, InferenceMetaInfo

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, help="Path to data file(s) (.dat or .txt)")
    parser.add_argument("--classifier", type=str, required=True, help="Path to fitted classifier object file (.joblib)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.joblib or .h5)")
    parser.add_argument("--pipeline", type=str, required=True, help="Path to data pipeline object (.joblib)")
    parser.add_argument("--processor", type=str, required=True, help="Path to data processor object (.joblib)")
    parser.add_argument("--metrics", type=str, required=True, help="Path to evaluation metrics object (.joblib)")
    parser.add_argument("--config-path", type=str, default=os.path.join(BASE_DIR, "..", "configs/inference-cfg.yaml"), help="Path to config file")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--outfile", type=str, default="meta_info.xlsx", help="Metrics output file")
    parser.add_argument("--remove-hiccups", type=str, default='False', help="Exclude hiccups from ML detections map")
    parser.add_argument("--plot", type=str, default='False', help="Generate and save detection plots")
    args = parser.parse_args()

    if isinstance(args.remove_hiccups, str) and args.remove_hiccups.lower() in ('false', 'f'):
        args.remove_hiccups = False
    if isinstance(args.plot, str) and args.plot.lower() in ('false', 'f'):
        args.plot = False

    return args


def main(args):
    LOGGER.info("Starting inference...")

    os.makedirs(args.work_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(args.config_path, 'r') as f:
        pred_cfg = yaml.safe_load(f)

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
            args.metrics,
            pred_cfg
        )
        _ = all([
            pred_service.get_model() is not None,
            pred_service.get_pipeline() is not None,
            pred_service.get_metrics() is not None
        ])
    except Exception as e:
        LOGGER.error(f"Error loading fitted objects: {e}")
        sys.exit(1)
        

    for data_filename in tqdm(filenames, desc=f"Performing inference on {len(filenames)} files..."):
        LOGGER.info(f"Performing inference for {data_filename}")

        pred_output = pred_service.predict(
            data_filename,
            remove_hiccups=args.remove_hiccups
        )
        job_id = str(uuid.uuid4())[:8]

        # Prepare meta info
        pre_removal_data: InferenceMetaInfo = pred_output['pre_hiccup_removal']['data']
        metainfo_dict = {
            "Timestamp": [timestamp],
            "JobId": [job_id],
            "File Name": [data_filename],
            "Number of bouts per hour - pre_hiccup": [(pre_removal_data.numKicks * 60) / (pre_removal_data.totalFMDuration + pre_removal_data.totalNonFMDuration)],
            "Mean duration of fetal movement (seconds) - pre_hiccup": [pre_removal_data.totalFMDuration * 60 / pre_removal_data.numKicks if pre_removal_data.numKicks > 0 else 0],
            "Median onset interval (seconds) - pre_hiccup": [np.median(pre_removal_data.onsetInterval)],
            "Active time of fetal movement (%) - pre_hiccup": [(pre_removal_data.totalFMDuration / (pre_removal_data.totalFMDuration + pre_removal_data.totalNonFMDuration)) * 100]
        }

        if args.remove_hiccups:
            post_removal_data: InferenceMetaInfo = pred_output['post_hiccup_removal']['data']
            metainfo_dict.update({
                "Number of bouts per hour - post_hiccup": [(post_removal_data.numKicks * 60) / (post_removal_data.totalFMDuration + post_removal_data.totalNonFMDuration)],
                "Mean duration of fetal movement (seconds) - post_hiccup": [post_removal_data.totalFMDuration * 60 / post_removal_data.numKicks if post_removal_data.numKicks > 0 else 0],
                "Median onset interval (seconds) - post_hiccup": [np.median(post_removal_data.onsetInterval)],
                "Active time of fetal movement (%) - post_hiccup": [(post_removal_data.totalFMDuration / (post_removal_data.totalFMDuration + post_removal_data.totalNonFMDuration)) * 100]
            })
            del post_removal_data  # Free memory

        del pre_removal_data  # Free memory

        # Write meta info to Excel
        output_file = os.path.join(args.work_dir, args.outfile)
        df_new = pd.DataFrame(metainfo_dict)

        if os.path.exists(output_file):
            df_existing = pd.read_excel(output_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
            df_combined.to_excel(writer, index=False)

        LOGGER.info(f"Meta info saved to {os.path.realpath(output_file)}")

        # Delete unused DataFrame variables
        del df_new, df_combined
        if 'df_existing' in locals():
            del df_existing

        # Plot results if required
        if args.plot:
            LOGGER.info("Plotting the results. It may take some time....")
            base_filename = os.path.basename(data_filename).split('.dat')[0]

            pred_service.save_pred_plots(
                pred_output['pipeline_output'],
                pred_output['pre_hiccup_removal']['ml_map'],
                os.path.join(args.work_dir, f"{base_filename}_{job_id}_ml.png")
            )

            if args.remove_hiccups:
                pred_service.save_pred_plots(
                    pred_output['pipeline_output'],
                    pred_output['post_hiccup_removal']['hiccup_map'],
                    os.path.join(args.work_dir, f"{base_filename}_{job_id}_hiccup.png"),
                    det_type='Hiccup regions'
                )
                pred_service.save_pred_plots(
                    pred_output['pipeline_output'],
                    pred_output['post_hiccup_removal']['hiccup_removed_ml_map'],
                    os.path.join(args.work_dir, f"{base_filename}_{job_id}_post-removal.png")
                )
                pred_service.save_hiccup_analysis_plots(
                    pred_output['pipeline_output'],
                    pred_output['post_hiccup_removal'],
                    pred_output['pre_hiccup_removal']['ml_map'],
                    os.path.join(args.work_dir, f"{base_filename}_{job_id}_hiccup-analysis.png")
                )

        # Free memory by deleting variables
        del pred_output
        gc.collect()  # Force garbage collection


if __name__ == "__main__":
    args = parse_args()
    main(args)
