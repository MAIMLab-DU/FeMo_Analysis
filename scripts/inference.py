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
from femo.data.transforms._utils import str2bool
from femo.inference import PredictionService, InferenceMetaInfo

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to data file(s) (.dat or .txt)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="Path to fitted classifier object file (.joblib)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file (.joblib or .h5)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Path to data pipeline object (.joblib)",
    )
    parser.add_argument(
        "--processor",
        type=str,
        required=True,
        help="Path to data processor object (.joblib)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to evaluation metrics object (.joblib)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=os.path.join(BASE_DIR, "..", "configs/inference-cfg.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./work_dir",
        help="Path to save generated artifacts",
    )
    parser.add_argument("--outfile", type=str, default="meta_info.xlsx", help="Metrics output file")
    parser.add_argument(
        "--remove-hiccups",
        type=str,
        default="False",
        help="Exclude hiccups from ML detections map",
    )
    parser.add_argument("--plot", type=str, default="False", help="Generate and save detection plots")
    parser.add_argument(
        "--plot-all-sensors",
        type=str,
        default="False",
        help="Plot all sensors in the output plots",
    )
    parser.add_argument(
        "--plot-preprocessed",
        type=str,
        default="False",
        help="Plot preprocessed data (default: plot loaded data)",
    )
    parser.add_argument(
        "--set-stage-param",
        type=str,
        nargs="*",
        default=[],
        help="Override data pipeline stage parameter(s), e.g. segment.fm_dilation=5",
    )
    args = parser.parse_args()

    args.remove_hiccups = str2bool(args.remove_hiccups)
    args.plot = str2bool(args.plot)
    args.plot_all_sensors = str2bool(args.plot_all_sensors)
    args.plot_preprocessed = str2bool(args.plot_preprocessed)

    return args


def parse_stage_param_overrides(param_list):
    """
    Parses CLI parameters like ['segment.fm_dilation=5', 'fusion.window=3'] into
    a dict of stage → {param: value}
    """
    from collections import defaultdict

    overrides = defaultdict(dict)
    for param in param_list:
        if "=" not in param or "." not in param:
            raise ValueError(f"Invalid format for --set-stage-param: '{param}'. Use 'stage.attr=value'")
        stage_attr, value = param.split("=", 1)
        stage, attr = stage_attr.split(".", 1)
        try:
            parsed_val = eval(value, {"__builtins__": {}}, {})  # Safely parse numbers, bools, etc.
        except Exception:
            parsed_val = value  # fallback to string
        overrides[stage][attr] = parsed_val
    return overrides


def main(args):
    LOGGER.info("Starting inference...")

    os.makedirs(args.work_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(args.config_path, "r") as f:
        pred_cfg = yaml.safe_load(f)

    if not args.data_file.endswith(".txt"):
        # If it's a .dat file, store it in a list
        filenames = [args.data_file]
    elif args.data_file.endswith(".txt"):
        # If it's a .txt file, read each line and store filenames in a list
        with open(args.data_file, "r") as file:
            filenames = [line.strip() for line in file if line.strip()]  # Strips newlines and empty lines
    else:
        raise ValueError("Unsupported file format. Please provide a binary file ('.dat') or '.txt' file.")

    try:
        pred_service = PredictionService(
            args.classifier,
            args.model,
            args.pipeline,
            args.processor,
            args.metrics,
            pred_cfg,
        )
        _ = all(
            [
                pred_service.get_model() is not None,
                pred_service.get_pipeline() is not None,
                pred_service.get_metrics() is not None,
            ]
        )
    except Exception as e:
        LOGGER.error(f"Error loading fitted objects: {e}")
        sys.exit(1)

    # Apply stage-level parameter overrides
    if args.set_stage_param:
        stage_overrides = parse_stage_param_overrides(args.set_stage_param)
        pipeline = pred_service.get_pipeline()
        for stage_name, attrs in stage_overrides.items():
            LOGGER.info(f"Updating stage '{stage_name}' with parameters: {attrs}")
            pipeline.set_stage_params(stage_name, **attrs)

    for data_filename in tqdm(filenames, desc=f"Performing inference on {len(filenames)} files..."):
        try:
            LOGGER.info(f"Performing inference for {data_filename}")

            pred_output = pred_service.predict(data_filename, remove_hiccups=args.remove_hiccups)
            job_id = str(uuid.uuid4())[:8]

            # Prepare meta info
            pre_removal_data: InferenceMetaInfo = pred_output["pre_hiccup_removal"]["data"]
            metainfo_dict = {
                "Timestamp": [timestamp],
                "JobId": [job_id],
                "File Name": [data_filename],
                "Number of bouts per hour - pre_hiccup": [
                    (pre_removal_data.numKicks * 60)
                    / (pre_removal_data.totalFMDuration + pre_removal_data.totalNonFMDuration)
                ],
                "Mean duration of fetal movement (seconds) - pre_hiccup": [
                    (
                        pre_removal_data.totalFMDuration * 60 / pre_removal_data.numKicks
                        if pre_removal_data.numKicks > 0
                        else 0
                    )
                ],
                "Median onset interval (seconds) - pre_hiccup": [np.median(pre_removal_data.onsetInterval)],
                "Active time of fetal movement (%) - pre_hiccup": [
                    (
                        pre_removal_data.totalFMDuration
                        / (pre_removal_data.totalFMDuration + pre_removal_data.totalNonFMDuration)
                    )
                    * 100
                ],
            }
            if pre_removal_data.matchWithSensationMap is not None:
                tp = pre_removal_data.matchWithSensationMap.true_positive
                fp = pre_removal_data.matchWithSensationMap.false_positive
                tn = pre_removal_data.matchWithSensationMap.true_negative
                fn = pre_removal_data.matchWithSensationMap.false_negative
                metainfo_dict.update(
                    {
                        "True positive - pre_hiccup": [
                            tp
                        ],
                        "False positive - pre_hiccup": [
                            fp
                        ],
                        "True negative - pre_hiccup": [
                            tn
                        ],
                        "False negative - pre_hiccup": [
                            fn
                        ],
                        "Num sensation label - pre_hiccup": [
                            pre_removal_data.matchWithSensationMap.num_maternal_sensed
                        ],
                        "Num sensation calc - pre_hiccup": [
                            tp + fn
                        ],
                        "Num sensor label - pre_hiccup": [
                            pre_removal_data.matchWithSensationMap.num_sensor_detections
                        ],
                        "Num sensor calc - pre_hiccup": [
                            tp + fp + tn + fn
                        ],
                        "Num ML label - pre_hiccup": [
                            pre_removal_data.matchWithSensationMap.num_ml_detections
                        ],
                        "Num ML calc - pre_hiccup": [
                            tp + fp
                        ],
                    }
                )

            if args.remove_hiccups:
                post_removal_data: InferenceMetaInfo = pred_output["post_hiccup_removal"]["data"]
                metainfo_dict.update(
                    {
                        "Number of bouts per hour - post_hiccup": [
                            (post_removal_data.numKicks * 60)
                            / (post_removal_data.totalFMDuration + post_removal_data.totalNonFMDuration)
                        ],
                        "Mean duration of fetal movement (seconds) - post_hiccup": [
                            (
                                post_removal_data.totalFMDuration * 60 / post_removal_data.numKicks
                                if post_removal_data.numKicks > 0
                                else 0
                            )
                        ],
                        "Median onset interval (seconds) - post_hiccup": [np.median(post_removal_data.onsetInterval)],
                        "Active time of fetal movement (%) - post_hiccup": [
                            (
                                post_removal_data.totalFMDuration
                                / (post_removal_data.totalFMDuration + post_removal_data.totalNonFMDuration)
                            )
                            * 100
                        ],
                    }
                )
                if post_removal_data.matchWithSensationMap is not None:
                    post_tp = post_removal_data.matchWithSensationMap.true_positive
                    post_fp = post_removal_data.matchWithSensationMap.false_positive
                    post_tn = post_removal_data.matchWithSensationMap.true_negative
                    post_fn = post_removal_data.matchWithSensationMap.false_negative
                    metainfo_dict.update(
                        {
                            "True positive - post_hiccup": [
                                post_tp
                            ],
                            "False positive - post_hiccup": [
                                post_fp
                            ],
                            "True negative - post_hiccup": [
                                post_tn
                            ],
                            "False negative - post_hiccup": [
                                post_fn
                            ],
                            "Num sensation label - post_hiccup": [
                                post_removal_data.matchWithSensationMap.num_maternal_sensed
                            ],
                            "Num sensation calc - post_hiccup": [
                                post_tp + post_fn
                            ],
                            "Num sensor label - post_hiccup": [
                                post_removal_data.matchWithSensationMap.num_sensor_detections
                            ],
                            "Num sensor calc - post_hiccup": [
                                post_tp + post_fp + post_tn + post_fn
                            ],
                            "Num ML label - post_hiccup": [
                                post_removal_data.matchWithSensationMap.num_ml_detections
                            ],
                            "Num ML calc - post_hiccup": [
                                post_tp + post_fp
                            ],
                        }
                    )
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

            with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
                df_combined.to_excel(writer, index=False)

            LOGGER.info(f"Meta info saved to {os.path.realpath(output_file)}")

            # Delete unused DataFrame variables
            del df_new, df_combined
            if "df_existing" in locals():
                del df_existing

            # Plot results if required
            if args.plot:
                LOGGER.info("Plotting the results. It may take some time....")
                base_filename = os.path.basename(data_filename).split(".dat")[0]

                pred_service.save_pred_plots(
                    pred_output["pipeline_output"],
                    pred_output["pre_hiccup_removal"]["ml_map"],
                    os.path.join(args.work_dir, f"{base_filename}_{job_id}_ml.png"),
                    plot_all_sensors=args.plot_all_sensors,
                    plot_preprocessed=args.plot_preprocessed,
                )

                if args.remove_hiccups:
                    pred_service.save_pred_plots(
                        pred_output["pipeline_output"],
                        pred_output["post_hiccup_removal"]["hiccup_map"],
                        os.path.join(args.work_dir, f"{base_filename}_{job_id}_hiccup.png"),
                        det_type="Hiccup regions",
                        plot_all_sensors=args.plot_all_sensors,
                        plot_preprocessed=args.plot_preprocessed,
                    )
                    pred_service.save_pred_plots(
                        pred_output["pipeline_output"],
                        pred_output["post_hiccup_removal"]["hiccup_removed_ml_map"],
                        os.path.join(args.work_dir, f"{base_filename}_{job_id}_post-removal.png"),
                        det_type="Hiccup removed ML Predicted Fetal Movement",
                        plot_all_sensors=args.plot_all_sensors,
                        plot_preprocessed=args.plot_preprocessed,
                    )
                    pred_service.save_hiccup_analysis_plots(
                        pred_output["pipeline_output"],
                        pred_output["post_hiccup_removal"],
                        pred_output["pre_hiccup_removal"]["ml_map"],
                        os.path.join(args.work_dir, f"{base_filename}_{job_id}_hiccup-analysis.png"),
                    )

            # Free memory by deleting variables
            del pred_output
            gc.collect()  # Force garbage collection
        
        except Exception as e:
            LOGGER.error(f"Error processing file {data_filename}: {e}")
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args)
