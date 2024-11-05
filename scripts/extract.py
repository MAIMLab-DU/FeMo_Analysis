import os
import yaml
import argparse
from femo.logger import LOGGER
from femo.data.dataset import FeMoDataset

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, required=True, help="Path to data manifest json file")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to directory containing .dat and .csv files")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    parser.add_argument("--config-path", type=str, default=os.path.join(BASE_DIR, "..", "configs/dataset-cfg.yaml"), help="Path to config file")
    parser.add_argument("--extract", action='store_true', default=False, help="Force extract features even if they exist")
    args = parser.parse_args()

    return args


def main(args):
    LOGGER.info("Starting feature extraction...")

    os.makedirs(args.work_dir, exist_ok=True)

    with open(args.config_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    LOGGER.info("Downloading raw input data")

    dataset = FeMoDataset(args.data_dir,
                          args.data_manifest,
                          False,
                          dataset_cfg.get('data_pipeline'))

    df = dataset.build(force_extract=args.extract)
    df.to_csv(os.path.join(args.work_dir, 'features.csv'), header=True, index=False)
    LOGGER.info(f"Features saved to {os.path.abspath(args.work_dir)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
