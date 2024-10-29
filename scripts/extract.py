import os
import yaml
import argparse
from femo.logger import LOGGER
from femo.data.dataset import FeMoDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, required=True, help="Path to data manifest json file")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to directory containing .dat and .csv files")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--extract", action='store_true', default=False, help="Extract features ")
    args = parser.parse_args()

    return args


def main():
    LOGGER.info("Starting data processing...")
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    config_files = ['dataset-cfg.yaml']
    [dataset_cfg] = [yaml.safe_load(open(os.path.join(config_dir, cfg), 'r')) for cfg in config_files]

    LOGGER.info("Downloading raw input data")

    dataset = FeMoDataset(args.data_dir,
                          args.data_manifest,
                          False,
                          dataset_cfg.get('data_pipeline'))

    df = dataset.build(force_extract=args.extract)
    df.to_csv(os.path.join(args.work_dir, 'dataset.csv'), header=True, index=False)
    LOGGER.info(f"Dataset saved to {os.path.abspath(args.work_dir)}")


if __name__ == "__main__":
    main()
