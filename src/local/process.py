"""Processes the dataset for training"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import yaml
import joblib
import argparse
from logger import logger
from data.dataset import (
    FeMoDataset,
    DataProcessor
)


def main():
    logger.info("Starting data processing...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/opt/ml/processing")
    args = parser.parse_args()

    with open("configs/dataproc-cfg.yaml", "r") as f:
        dataproc_cfg = yaml.safe_load(f)

    logger.info("Downloading raw input data")
    data_dir = args.data_dir

    dataset = FeMoDataset(data_dir,
                          args.data_manifest,
                          dataproc_cfg.get('data_pipeline'))
    df = dataset.build()

    logger.info("Preprocessing raw input data")
    data_processor = DataProcessor(feat_rank_cfg=dataproc_cfg.get('feature_ranking'))
    data_output = data_processor.process(input_data=df)

    len_data_output = len(data_output)
    logger.info(f"Splitting {len_data_output} rows of data into train, test datasets.")
    split_dict = data_processor.split_data(
        data=data_output,
        **dataproc_cfg.get('data_processor', {})
    )
    train = split_dict['train']
    test = split_dict['test']

    joblib.dump(train, os.path.join(data_dir, 'train.pkl'), compress=True)
    joblib.dump(test, os.path.join(data_dir, 'test.pkl'), compress=True)
    logger.info(f"Saved datasets to {os.path.abspath(data_dir)}")


if __name__ == "__main__":
    main()