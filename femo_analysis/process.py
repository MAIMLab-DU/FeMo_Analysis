import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..', 'src')))
import yaml
import joblib
import argparse
from logger import LOGGER
from data.dataset import FeMoDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataManifest", type=str, help="Path to data manifest json file")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to directory containing .dat and .csv files")
    parser.add_argument("--params-filename", type=str, default=None, help="Parameters dict filename")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--extract", action='store_true', default=False, help="Extract features ")
    args = parser.parse_args()

    return args


def main():
    LOGGER.info("Starting data processing...")
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'dataproc-cfg.yaml')
    with open(config_path, "r") as f:
        dataproc_cfg = yaml.safe_load(f)

    LOGGER.info("Downloading raw input data")

    dataset = FeMoDataset(args.data_dir,
                          args.dataManifest,
                          False,
                          dataproc_cfg.get('data_pipeline'))

    df = dataset.build(force_extract=args.extract)
    df.to_csv(os.path.join(args.work_dir, 'features.csv'), header=True, index=False)
    LOGGER.info(f"Features saved to {os.path.abspath(args.work_dir)}")

    LOGGER.info("Preprocessing raw input data")
    data_output = dataset.process(input_data=df,
                                  params_filename=args.params_filename)

    LOGGER.info(f"Splitting {len(data_output)} rows of data into train, test datasets.")
    split_dict = dataset.split_data(
        data=data_output,
        **dataproc_cfg.get('split_config', {})
    )

    joblib.dump(split_dict, os.path.join(args.work_dir, 'split_dataset.pkl'), compress=True)
    LOGGER.info(f"Saved datasets to {os.path.abspath(args.work_dir)}")


if __name__ == "__main__":
    main()
