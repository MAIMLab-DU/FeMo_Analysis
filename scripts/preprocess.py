import os
import yaml
import joblib
import argparse
import pandas as pd
from femo.logger import LOGGER
from femo.data.preprocess import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to 'dataset.csv' file")
    parser.add_argument("--params-filename", type=str, default=None, help="Parameters dict filename")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    args = parser.parse_args()

    return args


def main():
    LOGGER.info("Starting training...")
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    LOGGER.info(f"Working directory {args.work_dir}")

    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    config_files = ['preprocess-cfg.yaml']
    [preproc_cfg] = [yaml.safe_load(open(os.path.join(config_dir, cfg), 'r')) for cfg in config_files]

    data_preprocessor = Preprocessor(preproc_cfg)

    if args.params_filename is None or args.params_filename == 'null':
        params_dict = None
    else:
        try:
            params_dict = joblib.load(args.params_filename)
        except FileNotFoundError:
            params_dict = None
    dataset = pd.read_csv(args.dataset_path)

    preprocessed_data, params_dict = data_preprocessor.preprocess(dataset, params_dict)
    if args.params_filename is not None and args.params_filename != 'null':
        joblib.dump(params_dict, args.params_filename, compress=True)

    preprocessed_data.to_csv(os.path.join(args.work_dir, 'preprocessed_dataset.csv'), header=True, index=False)
    LOGGER.info(f"Preprocessed dataset saved to {os.path.abspath(args.work_dir)}")
    

if __name__ == "__main__":
    main()