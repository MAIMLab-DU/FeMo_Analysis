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

    # Sagemaker specific
    if args.train_config_path is not None:
        import shutil
        dump_path = os.path.join(args.work_dir, "train_config", "train-cfg.json")
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        shutil.move(args.train_config_path, dump_path)
        LOGGER.info(f"Training configuration saved to {os.path.abspath(dump_path)}")

    
    features = pd.read_csv(os.path.join(args.features_dir, "features.csv"))
    X, y = features.to_numpy()[:, :-3], features.to_numpy()[:, -1]
    filename_hash, det_indices = features.to_numpy()[:, -3], features.to_numpy()[:, -2]  # metadata columns

    if os.path.exists(os.path.join(processor_dir, 'processor.joblib')):
        data_processor: Processor = joblib.load(os.path.join(args.work_dir, 'processor', 'processor.joblib'))
        X_pre = data_processor.predict(X)
    else:
        data_processor = Processor(preprocess_config=preproc_cfg)
        X_pre = data_processor.fit(X, y).predict(X)

    dataset = data_processor.convert_to_df(
        np.concatenate([X_pre, filename_hash[:, np.newaxis],
                        det_indices[:, np.newaxis], y[:, np.newaxis]], axis=1),
        columns=features.columns
    )
    data_processor.save(processor_dir)

    dataset.to_csv(os.path.join(dataset_dir, 'dataset.csv'), header=True, index=False)

    LOGGER.info(f"Preprocessed dataset saved to {os.path.abspath(dataset_dir)}")
    LOGGER.info(f"Processor saved to {os.path.abspath(processor_dir)}")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)