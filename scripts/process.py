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
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to 'dataset.csv' file")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--config-dir", type=str, default=None, help="Path to configuration directory")
    args = parser.parse_args()

    return args


def main():
    LOGGER.info("Starting data processing...")
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    LOGGER.info(f"Working directory {args.work_dir}")

    config_dir = args.config_dir
    if config_dir is None:
        config_dir = os.path.join(BASE_DIR, '..', 'configs')
    config_files = ['preprocess-cfg.yaml']
    [preproc_cfg] = [yaml.safe_load(open(os.path.join(config_dir, cfg), 'r')) for cfg in config_files]
    
    dataset = pd.read_csv(args.dataset_path)
    X, y = dataset.to_numpy()[:, :-1], dataset.to_numpy()[:, -1]

    if os.path.exists(os.path.join(args.work_dir, 'processor', 'processor.joblib')):
        data_processor: Processor = joblib.load(os.path.join(args.work_dir, 'processor', 'processor.joblib'))
        X_pre = data_processor.predict(X)
    else:
        data_processor = Processor(preprocess_config=preproc_cfg)
        X_pre = data_processor.fit(X, y).predict(X)

    preprocessed_data = data_processor.convert_to_df(
        np.concatenate([X_pre, y[:, np.newaxis]], axis=1),
        columns=dataset.columns
    )
    data_processor.save(os.path.join(args.work_dir, 'processor'))

    preprocessed_data.to_csv(os.path.join(args.work_dir, 'dataset.csv'), header=True, index=False)
    LOGGER.info(f"Preprocessed dataset saved to {os.path.abspath(args.work_dir)}")
    

if __name__ == "__main__":
    main()