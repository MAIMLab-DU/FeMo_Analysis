# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                                                              *

"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import json
import yaml
import boto3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data.dataset import FeMoDataset, DataProcessor
import joblib
import tarfile



# TODO: implement
def run_main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.debug("Starting data processing...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, required=True)
    args = parser.parse_args()
    with open("configs/pipeline-cfg.yaml", "r") as f:
        feat_rank_cfg = yaml.safe_load(f)

    logger.debug("Downloading raw input data")
    base_dir = "/opt/ml/processing"
    dataset = FeMoDataset(base_dir, args.data_manifest)
    df = dataset.build()

    logger.debug("Preprocessing raw input data")
    data_processor = DataProcessor(feat_rank_cfg=feat_rank_cfg)
    data_output = data_processor.process(input_data=df)

    len_data_output = len(data_output)
    logger.info(f"Splitting {len_data_output} rows of data into train, validation, test datasets.")
    np.random.shuffle(data_output)
    train, validation, test = np.split(
        data_output, [int(0.7 * len_data_output), int(0.85 * len_data_output)]
    )

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

    logger.info("Saving the preprocessing model to %s", base_dir)
    data_processor.save_model(os.path.join(base_dir, "model"))

if __name__ == "__main__":
    run_main()