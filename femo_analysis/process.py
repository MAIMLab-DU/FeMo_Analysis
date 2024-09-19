import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..', 'src')))
import yaml
import joblib
import argparse
from logger import LOGGER
from data.dataset import (
    FeMoDataset,
    DataProcessor
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, required=True, help="Path to data manifest json file")
    parser.add_argument("--data-dir", type=str, default="/opt/ml/processing", help="Path to directory containing .dat and .csv files")
    parser.add_argument("--inference", action='store_true', help="Flag enable data processing for inference")
    args = parser.parse_args()

    return args


def main():
    LOGGER.info("Starting data processing...")
    args = parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'dataproc-cfg.yaml')
    with open(config_path, "r") as f:
        dataproc_cfg = yaml.safe_load(f)

    LOGGER.info("Downloading raw input data")
    data_dir = args.data_dir

    dataset = FeMoDataset(data_dir,
                          args.data_manifest,
                          args.inference,
                          dataproc_cfg.get('data_pipeline'))
    df = dataset.build()
    if args.inference:
        LOGGER.info("Dataset built for inference.")
        return

    LOGGER.info("Preprocessing raw input data")
    data_processor = DataProcessor(feat_rank_cfg=dataproc_cfg.get('feature_ranking'))
    data_output = data_processor.process(input_data=df)

    len_data_output = len(data_output)
    LOGGER.info(f"Splitting {len_data_output} rows of data into train, test datasets.")
    split_dict = data_processor.split_data(
        data=data_output,
        **dataproc_cfg.get('data_processor', {})
    )
    train = split_dict['train']
    test = split_dict['test']

    joblib.dump(train, os.path.join(data_dir, 'train.pkl'), compress=True)
    joblib.dump(test, os.path.join(data_dir, 'test.pkl'), compress=True)
    LOGGER.info(f"Saved datasets to {os.path.abspath(data_dir)}")


if __name__ == "__main__":
    main()
