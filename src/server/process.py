"""Processes the dataset for training"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import yaml
import joblib
import argparse
import logging
from data.dataset import (
    FeMoDataset,
    DataProcessor
)


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.debug("Starting data processing...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, required=True)
    args = parser.parse_args()
    with open("configs/dataproc-cfg.yaml", "r") as f:
        dataproc_cfg = yaml.safe_load(f)

    logger.debug("Downloading raw input data")
    base_dir = "/opt/ml/processing"
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    dataset = FeMoDataset(base_dir,
                          args.data_manifest,
                          dataproc_cfg.get('data_pipeline'))
    df = dataset.build()

    logger.debug("Preprocessing raw input data")
    data_processor = DataProcessor(feat_rank_cfg=dataproc_cfg.get('feature_ranking'))
    data_output = data_processor.process(input_data=df)

    len_data_output = len(data_output)
    logger.info(f"Splitting {len_data_output} rows of data into train, test datasets.")
    split_dict = data_processor.split_data(
        data_output=data_output,
        **dataproc_cfg.get('data_processor', {})
    )
    train = split_dict['train']
    test = split_dict['test']

    logger.info(f"Saving datasets to {data_dir}.")
    joblib.dump(train, os.path.join(data_dir, 'train.pkl'), compress=True)
    joblib.dump(test, os.path.join(data_dir, 'test.pkl'), compress=True)


if __name__ == "__main__":
    main()