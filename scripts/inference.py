import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..', 'femo')))
import argparse
from tqdm import tqdm
from logger import LOGGER
from data.pipeline import Pipeline
from data.utils import normalize_features
from eval.metrics import FeMoMetrics
from model import CLASSIFIER_MAP
from model.base import FeMoBaseClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataFilename", type=str, help="Path to data file")
    parser.add_argument("ckptFilename", type=str, help="Name of model checkpoint file")
    parser.add_argument("paramsFilename", type=str,  help="Parameters dict filename")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--outfile", type=str, default="meta_info.xlsx", help="Metrics output file")
    args = parser.parse_args()

    return args


def main():
    LOGGER.info("Starting inference...")
    args = parse_args()

    if args.dataFilename.endswith('.dat'):
        # If it's a .dat file, store it in a list
        filenames = [args.dataFilename]
    elif args.dataFilename.endswith('.txt'):
        # If it's a .txt file, read each line and store filenames in a list
        with open(args.dataFilename, 'r') as file:
            filenames = [line.strip() for line in file if line.strip()]  # Strips newlines and empty lines
            filenames = [line for line in filenames if line.endswith('.dat')]
    else:
        raise ValueError("Unsupported file format. Please provide a .dat or .txt file.")
    LOGGER.info(f"Filenames: {filenames}")

    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    config_files = ['inference-cfg.yaml', 'dataproc-cfg.yaml']
    inf_cfg, dataproc_cfg = [yaml.safe_load(open(os.path.join(config_dir, cfg), 'r')) for cfg in config_files]

    try:
        params_dict = joblib.load(args.paramsFilename)
    except FileNotFoundError:
        LOGGER.error(f"Parameters file not found: {args.paramsFilename}")
        sys.exit(1)
    
    try:
        classifier_type = inf_cfg.get('type')        
        classifier: type[FeMoBaseClassifier] = CLASSIFIER_MAP[classifier_type]()
        classifier.load_model(
            model_filename=args.ckptFilename,
            model_framework='keras' if classifier_type == 'neural-net' else 'sklearn'
        )
    except KeyError:
        LOGGER.error(f"Invalid classifier specified: {inf_cfg.get('classifier')}")
        sys.exit(1)

    metrics_calculator = FeMoMetrics()
    pipeline = Pipeline(
            inference=True,
            cfg=dataproc_cfg.get('data_pipeline')
        )
        
    for data_filename in tqdm(filenames, desc=f"Peforming inference on {len(filenames)} files..."):
        pipeline_output = pipeline.process(
                filename=data_filename
            )
        
        X_extracted = pipeline_output['extracted_features']['features']

        # Skip IMU features if desired
        if inf_cfg.get('skip_imu_features'):
            X_extracted = X_extracted[:, :319]
            mask = np.all(np.isin(X_extracted, [0, 1]), axis=0)
            X_extracted = X_extracted[:, ~mask]

        X_norm, _, _ = normalize_features(X_extracted, params_dict['mu'], params_dict['dev'])
        X_norm_ranked = X_norm[:, params_dict['top_feat_indices']]

        y_pred_labels = classifier.predict(X_norm_ranked)

        metainfo_dict = metrics_calculator.calc_meta_info(
            filename=data_filename,
            y_pred=y_pred_labels,
            preprocessed_data=pipeline_output['preprocessed_data'],
            fm_dict=pipeline_output['fm_dict'],
            scheme_dict=pipeline_output['scheme_dict']
        )

        # Check if the file exists
        if os.path.exists(args.outfile):
            # If file exists, read it and append the new data
            df_existing = pd.read_excel(args.outfile)
            df_new = pd.DataFrame(metainfo_dict)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            # If file does not exist, create a new DataFrame with titles and new data
            df_combined = pd.DataFrame(metainfo_dict)

        # Write the combined DataFrame to the Excel file
        df_combined.to_excel(args.outfile, index=False)

        LOGGER.info(f"Meta info saved to {args.outfile}")

        # TODO: plot results
    

if __name__ == "__main__":
    main()