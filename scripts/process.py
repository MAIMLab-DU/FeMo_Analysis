import os
import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
from femo.logger import LOGGER
from femo.data.process import Processor

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, required=True, help="Directory containing 'features.csv' file")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--config-path", type=str, default=os.path.join(BASE_DIR, "..", "configs/preprocess-cfg.yaml"), help="Path to config file")
    parser.add_argument("--train-config-path", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    return args


def main(args):
    LOGGER.info("Starting data processing...")

    dataset_dir = os.path.join(args.work_dir, "dataset")
    processor_dir = os.path.join(args.work_dir, "processor")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(processor_dir, exist_ok=True)
    LOGGER.info(f"Working directory {os.path.abspath(args.work_dir)}")

    with open(args.config_path, 'r') as f:
        preproc_cfg = yaml.safe_load(f)
    
    feature_sets = preproc_cfg.get('feature_sets', ['crafted'])

    # Sagemaker specific
    if args.train_config_path is not None:
        import shutil
        dump_path = os.path.join(args.work_dir, "train_config", "train-cfg.json")
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        shutil.move(args.train_config_path, dump_path)
        LOGGER.info(f"Training configuration saved to {os.path.abspath(dump_path)}")

    for key in feature_sets:  
        features = pd.read_csv(os.path.join(args.features_dir, f"{key}_features.csv"))
        X, y = features.iloc[:, :-4].to_numpy(), features.iloc[:, -1].to_numpy()  # second to last 4 columns are metadata
        filename_hash, start_indices, end_indices = features.iloc[:, -4].to_numpy(), features.iloc[:, -3].to_numpy(), features.iloc[:, -2].to_numpy()  # metadata columns

        LOGGER.info(f"Processing started for '{key}' features")
        LOGGER.info(f"Number of samples: {X.shape[0]}, Number of features: {X.shape[1]}")
        if os.path.exists(os.path.join(processor_dir, f"{key}_processor.joblib")):
            data_processor: Processor = joblib.load(os.path.join(processor_dir, f'{key}_processor.joblib'))
            X_pre = data_processor.predict(X)
        else:
            data_processor = Processor(preprocess_config=preproc_cfg, feature_set=key)
            X_pre = data_processor.fit(X, y).predict(X)

        dataset = data_processor.convert_to_df(
            np.concatenate([X_pre, filename_hash[:, np.newaxis],
                            start_indices[:, np.newaxis], end_indices[:, np.newaxis], y[:, np.newaxis]], axis=1),
            columns=features.columns
        )
        data_processor.save(processor_dir, key)

        dataset.to_csv(os.path.join(dataset_dir, f'{key}_dataset.csv'), header=True, index=False)

        LOGGER.info(f"Preprocessed dataset saved to {os.path.abspath(dataset_dir)}")
        LOGGER.info(f"Processor saved to {os.path.abspath(processor_dir)}")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)